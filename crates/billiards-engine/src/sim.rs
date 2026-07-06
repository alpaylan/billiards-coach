//! Single-ball motion solver and the shared motion/cushion primitives that the
//! multi-ball scheduler ([`crate::world`]) also builds on.
//!
//! # Physics
//!
//! With the ball center at height `radius`, the contact point's slip velocity
//! (see [`BallState::contact_slip`]) governs which phase we are in.
//!
//! **Sliding.** Kinetic friction of magnitude `μ_slide · m · g` acts at the
//! contact point, opposite the slip direction `û` (constant during a slide).
//! The center decelerates at `a = μ_slide · g` and friction torques the ball,
//! so the slip velocity itself decays at `(7/2)·a` — reaching zero after
//! `t = |u₀| / ((7/2)·a)`. A ball struck flat (`ω₀ = 0`) therefore enters
//! natural roll at exactly `(5/7)|v₀|`, the classic result.
//!
//! **Rolling.** Rolling resistance `μ_roll · g` decelerates the center along
//! `v̂` until it stops, the angular velocity kept consistent with rolling.

use billiards_core::math::{DVec3, SLIP_EPSILON, V_EPSILON};
use billiards_core::{
    BallSpec, BallState, MotionPhase, MotionSegment, PhysicsParams, TableSpec, Trajectory,
};

use crate::cushion::{Rail, han2005_rebound};

/// Guard against pathological non-termination.
pub(crate) const MAX_SEGMENTS: usize = 100_000;

/// Time margin used to avoid re-detecting a contact the ball is already on.
pub(crate) const CONTACT_EPS: f64 = 1e-9;

/// The constant-acceleration motion a ball follows during its current phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Motion {
    pub phase: MotionPhase,
    pub lin_acc: DVec3,
    pub ang_acc: DVec3,
    /// Time until this phase ends on its own (slide→roll or roll→stop);
    /// `INFINITY` when stationary.
    pub dur: f64,
}

/// Classify a ball's current phase and the analytic motion it implies.
pub(crate) fn motion_of(state: BallState, ball: &BallSpec, phys: &PhysicsParams) -> Motion {
    let slip = state.contact_slip(ball.radius);
    let speed = state.vel.length();

    if slip.length() > SLIP_EPSILON {
        let a = phys.mu_slide * phys.g;
        let uhat = slip.normalize();
        Motion {
            phase: MotionPhase::Sliding,
            lin_acc: uhat * -a,
            ang_acc: DVec3::new(-uhat.y, uhat.x, 0.0) * (5.0 * a / (2.0 * ball.radius)),
            dur: slip.length() / (3.5 * a),
        }
    } else if speed > V_EPSILON {
        let a = phys.mu_roll * phys.g;
        let vhat = state.vel.normalize();
        Motion {
            phase: MotionPhase::Rolling,
            lin_acc: vhat * -a,
            ang_acc: DVec3::new(vhat.y, -vhat.x, 0.0) * (a / ball.radius),
            dur: speed / a,
        }
    } else {
        Motion { phase: MotionPhase::Stationary, lin_acc: DVec3::ZERO, ang_acc: DVec3::ZERO, dur: f64::INFINITY }
    }
}

/// Apply the state change that occurs when a phase ends naturally: a sliding
/// ball snaps onto the rolling manifold; a rolling ball stops.
pub(crate) fn apply_transition(state: &mut BallState, phase: MotionPhase, radius: f64) {
    match phase {
        MotionPhase::Sliding => state.angular_vel = state.rolling_angular_vel(radius),
        MotionPhase::Rolling => state.vel = DVec3::ZERO,
        MotionPhase::Stationary => {}
    }
}

/// Build the motion segment for `motion` starting at `t` from `state`.
pub(crate) fn segment_for(state: BallState, t: f64, motion: &Motion) -> MotionSegment {
    MotionSegment {
        phase: motion.phase,
        t_start: t,
        t_end: t + motion.dur,
        state0: state,
        lin_acc: motion.lin_acc,
        ang_acc: motion.ang_acc,
    }
}

/// Earliest cushion contact within `horizon` seconds of a ball following
/// `lin_acc` from `state0`, as `(dt, rail)` with `dt ∈ (CONTACT_EPS, horizon]`.
/// Only counts rails the ball is actually moving into.
pub(crate) fn cushion_contact(
    state0: BallState,
    lin_acc: DVec3,
    horizon: f64,
    table: &TableSpec,
    radius: f64,
) -> Option<(f64, Rail)> {
    let mut best: Option<(f64, Rail)> = None;
    for rail in Rail::ALL {
        let (is_x, bound) = rail.bound(table, radius);
        let (p, v, a) = if is_x {
            (state0.pos.x, state0.vel.x, lin_acc.x)
        } else {
            (state0.pos.y, state0.vel.y, lin_acc.y)
        };
        if let Some(dt) = smallest_root_in(0.5 * a, v, p - bound, CONTACT_EPS, horizon) {
            let vel_at = state0.vel + lin_acc * dt;
            if vel_at.dot(rail.outward_normal()) > 0.0 && best.is_none_or(|(bd, _)| dt < bd) {
                best = Some((dt, rail));
            }
        }
    }
    best
}

/// Smallest root of `a·x² + b·x + c = 0` in `(lo, hi]`, if any.
pub(crate) fn smallest_root_in(a: f64, b: f64, c: f64, lo: f64, hi: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut consider = |x: f64| {
        if x > lo && x <= hi {
            best = Some(best.map_or(x, |b: f64| b.min(x)));
        }
    };
    if a.abs() < 1e-12 {
        if b.abs() > 1e-12 {
            consider(-c / b);
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            consider((-b - sq) / (2.0 * a));
            consider((-b + sq) / (2.0 * a));
        }
    }
    best
}

/// Simulate a single ball moving freely on the cloth (no cushions, no other
/// balls) until it comes to rest.
pub fn simulate_free(initial: BallState, ball: &BallSpec, phys: &PhysicsParams) -> Trajectory {
    let mut segments = Vec::new();
    let mut state = initial;
    let mut t = 0.0;

    for _ in 0..MAX_SEGMENTS {
        let motion = motion_of(state, ball, phys);
        let segment = segment_for(state, t, &motion);
        if motion.phase == MotionPhase::Stationary {
            segments.push(segment);
            break;
        }
        state = segment.state_at(segment.t_end);
        apply_transition(&mut state, motion.phase, ball.radius);
        t = segment.t_end;
        segments.push(segment);
    }
    Trajectory::new(segments)
}

/// Simulate a single ball on a bounded table, rebounding off cushions (Han 2005)
/// until it comes to rest.
pub fn simulate_table(
    initial: BallState,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> Trajectory {
    let mut segments = Vec::new();
    let mut state = initial;
    let mut t = 0.0;

    for _ in 0..MAX_SEGMENTS {
        let motion = motion_of(state, ball, phys);
        if motion.phase == MotionPhase::Stationary {
            segments.push(segment_for(state, t, &motion));
            break;
        }

        match cushion_contact(state, motion.lin_acc, motion.dur, table, ball.radius) {
            Some((dt, rail)) => {
                let mut seg = segment_for(state, t, &motion);
                seg.t_end = t + dt;
                let contact = seg.state_at(seg.t_end);
                segments.push(seg);
                state = han2005_rebound(contact, rail.outward_normal(), ball, phys, table.cushion_height);
                t += dt;
            }
            None => {
                let seg = segment_for(state, t, &motion);
                state = seg.state_at(seg.t_end);
                apply_transition(&mut state, motion.phase, ball.radius);
                t = seg.t_end;
                segments.push(seg);
            }
        }
    }
    Trajectory::new(segments)
}
