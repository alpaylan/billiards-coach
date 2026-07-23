//! Multi-ball event scheduler — the piece that turns per-ball motion, cushions,
//! and collisions into a complete shot.
//!
//! Between events every ball follows its own constant-acceleration phase, so the
//! world evolves analytically until the *soonest* of:
//!
//! - a ball's phase transition (slide→roll, roll→stop),
//! - a ball–cushion contact,
//! - a ball–ball contact.
//!
//! Ball–ball timing is the interesting one: the separation of two accelerating
//! balls is a quartic in `t`, so contact (`|Δp| = 2R`) is the smallest positive
//! root of a quartic — the classic event-based-billiards root find. We solve it
//! only over the window in which both balls' motions stay constant (until the
//! earlier of their transitions); anything later is naturally re-evaluated after
//! that transition fires.

use billiards_core::math::DVec3;
use billiards_core::{
    BallId, BallSpec, BallState, ContactEvent, ContactKind, MotionPhase, MotionSegment,
    PhysicsParams, Simulation, TableSpec, Trajectory,
};

use crate::collision::ball_ball_collision;
use crate::cushion::{CHAIN_RECOVERY_T, Rail, cushion_rebound};
use crate::sim::{CONTACT_EPS, Motion, apply_transition, cushion_contact, motion_of, segment_for};

/// Guard against pathological non-termination.
const MAX_EVENTS: usize = 500_000;

/// The soonest event across the whole world.
enum Event {
    Transition(usize, MotionPhase),
    Cushion(usize, Rail),
    BallBall(usize, usize),
}

/// Simulate a set of balls on a bounded table until all come to rest. Trajectory
/// `i` in the result corresponds to `initial[i]` (i.e. `BallId(i)`).
pub fn simulate(
    initial: &[BallState],
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> Simulation {
    let n = initial.len();
    let mut states = initial.to_vec();
    let mut raw: Vec<Vec<MotionSegment>> = vec![Vec::new(); n];
    let mut events: Vec<ContactEvent> = Vec::new();
    let mut t = 0.0;
    // Per-ball time of the last cushion contact, for the chain-restitution
    // recovery factor (see cushion_rebound).
    let mut last_cushion = vec![f64::NEG_INFINITY; n];

    for _ in 0..MAX_EVENTS {
        let motions: Vec<Motion> = states.iter().map(|s| motion_of(*s, ball, phys)).collect();

        // Find the soonest event (smallest positive dt).
        let mut best: Option<(f64, Event)> = None;
        let consider = |dt: f64, ev: Event, best: &mut Option<(f64, Event)>| {
            if dt > 0.0 && dt.is_finite() && best.as_ref().is_none_or(|(bd, _)| dt < *bd) {
                *best = Some((dt, ev));
            }
        };

        for i in 0..n {
            let m = &motions[i];
            // A resting ball can still carry decaying english: its spin-stop
            // is a real (finite-dur) transition. Only a spin-free rest is inert.
            if m.phase == MotionPhase::Stationary && m.dur.is_infinite() {
                continue;
            }
            consider(m.dur, Event::Transition(i, m.phase), &mut best);
            if let Some((dt, rail)) = cushion_contact(states[i], m.lin_acc, m.dur, table, ball.radius) {
                consider(dt, Event::Cushion(i, rail), &mut best);
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if motions[i].phase == MotionPhase::Stationary
                    && motions[j].phase == MotionPhase::Stationary
                {
                    continue;
                }
                // Both motions are valid only until the earlier transition.
                let horizon = motions[i].dur.min(motions[j].dur);
                if let Some(dt) = ball_ball_contact_dt(
                    (states[i], motions[i].lin_acc),
                    (states[j], motions[j].lin_acc),
                    ball.radius,
                    horizon,
                ) {
                    consider(dt, Event::BallBall(i, j), &mut best);
                }
            }
        }

        let Some((dt, event)) = best else { break };

        // Advance every ball by dt, recording a segment.
        for i in 0..n {
            let seg = {
                let mut s = segment_for(states[i], t, &motions[i]);
                s.t_end = t + dt;
                s
            };
            states[i] = seg.state_at(t + dt);
            raw[i].push(seg);
        }
        t += dt;

        // Resolve.
        match event {
            Event::Transition(i, phase) => apply_transition(&mut states[i], phase, ball.radius),
            Event::Cushion(i, rail) => {
                let recovery = (1.0 - (t - last_cushion[i]) / CHAIN_RECOVERY_T).clamp(0.0, 1.0);
                states[i] = cushion_rebound(states[i], rail.outward_normal(), ball, phys, table.cushion_height, recovery);
                last_cushion[i] = t;
                events.push(ContactEvent { time: t, kind: ContactKind::Cushion { ball: BallId(i as u8) } });
            }
            Event::BallBall(i, j) => {
                let (a, b) = ball_ball_collision(states[i], states[j], ball, phys);
                states[i] = a;
                states[j] = b;
                events.push(ContactEvent {
                    time: t,
                    kind: ContactKind::BallBall { a: BallId(i as u8), b: BallId(j as u8) },
                });
            }
        }
    }

    let trajectories = raw
        .into_iter()
        .zip(states)
        .map(|(segs, rest)| finalize(segs, rest, t))
        .collect();

    Simulation { trajectories, events }
}

/// Smallest positive time (within `horizon`) at which two balls, each following
/// constant acceleration, come into contact while approaching one another.
fn ball_ball_contact_dt(
    (a_state, a_acc): (BallState, DVec3),
    (b_state, b_acc): (BallState, DVec3),
    radius: f64,
    horizon: f64,
) -> Option<f64> {
    // Relative motion r(τ) = d0 + dv·τ + ½·da·τ²; contact at |r| = 2R.
    let d0 = a_state.pos - b_state.pos;
    let dv = a_state.vel - b_state.vel;
    let da = a_acc - b_acc;
    let half_da = da * 0.5;

    let c4 = half_da.dot(half_da);
    let c3 = 2.0 * half_da.dot(dv);
    let c2 = 2.0 * half_da.dot(d0) + dv.dot(dv);
    let c1 = 2.0 * dv.dot(d0);
    let c0 = d0.dot(d0) - 4.0 * radius * radius;

    for root in real_roots(c4, c3, c2, c1, c0) {
        if root <= CONTACT_EPS || root > horizon {
            continue;
        }
        let r = d0 + dv * root + half_da * (root * root);
        let r_dot = dv + da * root;
        if r.dot(r_dot) < 0.0 {
            // approaching
            return Some(root);
        }
    }
    None
}

/// Real roots of `c4·x⁴ + c3·x³ + c2·x² + c1·x + c0`, ascending. Dispatches by
/// the leading nonzero coefficient so a degenerate (e.g. equal-acceleration)
/// case still solves cleanly.
fn real_roots(c4: f64, c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    use roots::{find_roots_cubic, find_roots_linear, find_roots_quadratic, find_roots_quartic};
    const TINY: f64 = 1e-12;
    let roots = if c4.abs() > TINY {
        find_roots_quartic(c4, c3, c2, c1, c0)
    } else if c3.abs() > TINY {
        find_roots_cubic(c3, c2, c1, c0)
    } else if c2.abs() > TINY {
        find_roots_quadratic(c2, c1, c0)
    } else if c1.abs() > TINY {
        find_roots_linear(c1, c0)
    } else {
        return Vec::new();
    };
    let mut v = roots.as_ref().to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Turn a ball's raw per-event segments into a tidy [`Trajectory`]: append the
/// terminal rest, then merge consecutive stationary segments.
fn finalize(mut segs: Vec<MotionSegment>, rest: BallState, t_end: f64) -> Trajectory {
    segs.push(MotionSegment {
        phase: MotionPhase::Stationary,
        t_start: t_end,
        t_end: f64::INFINITY,
        state0: BallState { vel: DVec3::ZERO, ..rest },
        lin_acc: DVec3::ZERO,
        ang_acc: DVec3::ZERO,
    });

    let mut out: Vec<MotionSegment> = Vec::with_capacity(segs.len());
    for seg in segs {
        match out.last_mut() {
            Some(last) if last.phase == MotionPhase::Stationary && seg.phase == MotionPhase::Stationary => {
                last.t_end = seg.t_end;
            }
            _ => out.push(seg),
        }
    }
    Trajectory::new(out)
}
