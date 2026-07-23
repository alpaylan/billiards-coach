//! Cushion→cushion CHAIN calibration: the regime every other event instrument
//! was blind to, by construction.
//!
//! cushion_events gates the incoming ball to ≥0.35 s of free rolling, so it
//! only ever measures ISOLATED bounces — and on those, the Han map with
//! e_c ≈ 0.84 is accurate. Yet whole-shot reconstruction demands e_c ≈ 0.91,
//! and every energy-side explanation has been eliminated (launch measurement,
//! ball-ball, free roll, regime dependence, pin boxes). What no instrument has
//! tested is the EXIT STATE: a real shot crosses cushions in quick succession
//! (0.15–0.35 s apart), so bounce N's exit spin — which the isolated fit never
//! validates, because its same-operator absorbs exit-spin error into the
//! fitted e — becomes bounce N+1's input. If the model's exit spin is wrong,
//! chains compound the error while singles look perfect.
//!
//! The test: for chains of two bounces of the same ball < 0.35 s apart, with a
//! KNOWN incoming state at bounce 1 (≥0.35 s rolling before it; english as the
//! usual per-event nuisance), push the model through BOTH bounces — rebound at
//! rail 1, free motion for the observed gap, rebound at rail 2, free motion —
//! and compare, in measurement space, against the observed post-chain window.
//! If the model's chain residuals match its isolated-bounce residuals
//! (~0.07–0.15 m/s), the exit state is fine and the hypothesis dies. If they
//! blow up — and if a different contact model (Mathavan) or different
//! parameters fix chains specifically — the compensation is finally localized.
//!
//!   cargo run -p billiards-solver --example chain_events --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams, TableSpec};
use billiards_engine::{cushion_rebound, simulate_free};
use billiards_solver::fit::{FirstMove, ObservedEvents};
use billiards_solver::measure::window_velocity;
use billiards_solver::shotfile;
use rayon::prelude::*;

struct ChainEvent {
    game: usize,
    rail1: u8,
    rail2: u8,
    /// Observed time between the two bounce Vs.
    gap: f64,
    /// Approach velocity at bounce 1 (rolling-corrected to the contact).
    v_in: DVec3,
    /// Measured post-chain velocity (after bounce 2) and its sample times
    /// relative to bounce 2 — the same-operator target.
    v_out_meas: DVec3,
    post_ts: Vec<f64>,
    /// English before bounce 1 unknown → per-event nuisance.
    nuisance: bool,
}

fn rail_normal(rail: u8) -> DVec3 {
    match rail {
        0 => DVec3::new(-1.0, 0.0, 0.0),
        1 => DVec3::new(1.0, 0.0, 0.0),
        2 => DVec3::new(0.0, -1.0, 0.0),
        _ => DVec3::new(0.0, 1.0, 0.0),
    }
}

fn rail_dist(p: DVec3, rail: u8, b: [f64; 4]) -> f64 {
    match rail {
        0 => p.x - b[0],
        1 => b[1] - p.x,
        2 => p.y - b[2],
        _ => b[3] - p.y,
    }
}

fn load_field(dir: &str, key: &str) -> Option<f64> {
    let text = fs::read_to_string(format!("{}/calibration.json", dir.trim_end_matches('/'))).ok()?;
    let i = text.find(&format!("\"{key}\""))?;
    let a = &text[i + key.len() + 2..];
    let a = &a[a.find(':')? + 1..];
    a.chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
        .collect::<String>()
        .parse()
        .ok()
}

fn game_base(dir: &str) -> PhysicsParams {
    let mut p = PhysicsParams::carom_calibrated();
    if let Some(v) = load_field(dir, "mu_slide") {
        p.mu_slide = v;
    }
    if let Some(v) = load_field(dir, "mu_roll") {
        p.mu_roll = v;
    }
    p
}

/// Cushion-map params under test, applied over the event's game cloth.
fn with_game(ev: &ChainEvent, bases: &[PhysicsParams], tune: &PhysicsParams) -> PhysicsParams {
    let base = &bases[ev.game];
    PhysicsParams { mu_slide: base.mu_slide, mu_roll: base.mu_roll, ..tune.clone() }
}

/// The model's post-chain measured velocity: bounce 1 → free motion for the
/// observed gap → bounce 2 → free motion → the event's own sample times →
/// same linear fit. `None` when the model's mid-chain ball isn't even
/// approaching rail 2 (exit direction/speed wrong enough to break the chain).
fn predict_chain(
    ev: &ChainEvent,
    omega_z: f64,
    ball: &BallSpec,
    phys: &PhysicsParams,
    table: &TableSpec,
) -> Option<DVec3> {
    let r = ball.radius;
    let state = BallState {
        pos: DVec3::ZERO,
        vel: ev.v_in,
        angular_vel: DVec3::new(-ev.v_in.y, ev.v_in.x, 0.0) / r + DVec3::new(0.0, 0.0, omega_z),
    };
    let a = cushion_rebound(state, rail_normal(ev.rail1), ball, phys, table.cushion_height, 0.0);
    let mid = simulate_free(a, ball, phys).state_at(ev.gap);
    if mid.vel.dot(rail_normal(ev.rail2)) < 0.02 {
        return None; // model ball not approaching the observed second rail
    }
    let b_in = BallState { pos: DVec3::ZERO, vel: mid.vel, angular_vel: mid.angular_vel };
    let recovery = (1.0 - ev.gap / billiards_engine::CHAIN_RECOVERY_T).clamp(0.0, 1.0);
    let b = cushion_rebound(b_in, rail_normal(ev.rail2), ball, phys, table.cushion_height, recovery);
    let traj = simulate_free(b, ball, phys);
    let pts: Vec<(f64, DVec3)> = ev.post_ts.iter().map(|&dt| (dt, traj.state_at(dt).pos)).collect();
    window_velocity(&pts).map(|(v, _, _)| v)
}

/// (capped sq err, angle err, ω_z, reached) — nuisance-optimized like the
/// other instruments; an unreachable chain contributes the cap.
fn event_residual(
    ev: &ChainEvent,
    ball: &BallSpec,
    phys: &PhysicsParams,
    table: &TableSpec,
) -> (f64, f64, bool) {
    const CAP: f64 = 0.36; // (0.6 m/s)² — chains legitimately err bigger than singles
    let score = |wz: f64| -> Option<f64> {
        let p = predict_chain(ev, wz, ball, phys, table)?;
        Some(
            (DVec3::new(p.x, p.y, 0.0) - DVec3::new(ev.v_out_meas.x, ev.v_out_meas.y, 0.0))
                .length_squared(),
        )
    };
    let mut best: Option<(f64, f64)> = None;
    if ev.nuisance {
        let mut wz = -72.0;
        while wz <= 72.0 {
            if let Some(se) = score(wz) {
                if best.is_none_or(|(_, b)| se < b) {
                    best = Some((wz, se));
                }
            }
            wz += 8.0;
        }
        if let Some((center, _)) = best {
            let mut wz = center - 8.0;
            while wz <= center + 8.0 {
                if let Some(se) = score(wz) {
                    if best.is_none_or(|(_, b)| se < b) {
                        best = Some((wz, se));
                    }
                }
                wz += 1.0;
            }
        }
    } else if let Some(se) = score(0.0) {
        best = Some((0.0, se));
    }
    match best {
        Some((wz, _)) => {
            let p = predict_chain(ev, wz, ball, phys, table).unwrap();
            let se = (DVec3::new(p.x, p.y, 0.0)
                - DVec3::new(ev.v_out_meas.x, ev.v_out_meas.y, 0.0))
            .length_squared();
            let mut d = (p.y.atan2(p.x) - ev.v_out_meas.y.atan2(ev.v_out_meas.x)).abs();
            if d > std::f64::consts::PI {
                d = std::f64::consts::TAU - d;
            }
            (se.min(CAP), d, true)
        }
        None => (CAP, 0.6, false), // never reaches rail 2 at any english
    }
}

/// (rms m/s over capped errors, mean angle deg over reached, unreachable count)
fn residuals(
    events: &[&ChainEvent],
    ball: &BallSpec,
    bases: &[PhysicsParams],
    tune: &PhysicsParams,
    table: &TableSpec,
) -> (f64, f64, usize) {
    let per: Vec<(f64, f64, bool)> = events
        .par_iter()
        .map(|ev| event_residual(ev, ball, &with_game(ev, bases, tune), table))
        .collect();
    let n = per.len().max(1) as f64;
    let se = per.iter().map(|p| p.0).sum::<f64>() / n;
    let reached: Vec<&(f64, f64, bool)> = per.iter().filter(|p| p.2).collect();
    let ae = reached.iter().map(|p| p.1).sum::<f64>() / reached.len().max(1) as f64;
    (se.sqrt(), ae.to_degrees(), per.iter().filter(|p| !p.2).count())
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let bounds = table.center_bounds(ball.radius);
    const G: f64 = 9.81;

    let mut events: Vec<ChainEvent> = Vec::new();
    let mut bases: Vec<PhysicsParams> = Vec::new();
    let mut n_pairs = 0usize;

    for (gi, dir) in dirs.iter().enumerate() {
        bases.push(game_base(dir));
        let decel = load_field(dir, "mu_roll").unwrap_or(0.0055) * G;
        let mut paths: Vec<String> = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".shot"))
            .collect();
        paths.sort();
        for p in &paths {
            let Ok(text) = fs::read_to_string(p) else { continue };
            let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };
            let obs = ObservedEvents::from_tracks(&shot.observed, bounds);
            let strikes: Vec<f64> = obs
                .first_move
                .iter()
                .filter_map(|fm| match fm {
                    FirstMove::At(t) => Some(*t),
                    _ => None,
                })
                .collect();
            for (ti, track) in shot.observed.iter().enumerate() {
                let bounces = &obs.cushions[ti];
                let is_cue = ti == 0;
                let t_move = if is_cue {
                    let Some(&(_, p0)) = track.first() else { continue };
                    match track.iter().find(|(_, p)| (*p - p0).length() > 0.02) {
                        Some(&(t, _)) => t,
                        None => continue,
                    }
                } else {
                    match obs.first_move.get(ti - 1) {
                        Some(FirstMove::At(t)) => *t,
                        _ => continue,
                    }
                };
                for i in 0..bounces.len().saturating_sub(1) {
                    let (t1, r1) = bounces[i];
                    let (t2, r2) = bounces[i + 1];
                    let gap = t2 - t1;
                    if !(0.12..0.35).contains(&gap) || r1 == r2 {
                        continue;
                    }
                    n_pairs += 1;
                    // Known incoming state at bounce 1: free roll ≥0.35 s.
                    let t_prev = bounces[..i]
                        .iter()
                        .map(|&(t, _)| t)
                        .chain(strikes.iter().copied().filter(|&ts| ts < t1))
                        .fold(t_move, f64::max);
                    if t1 - t_prev < 0.35 {
                        continue;
                    }
                    // Clean post-chain window before the next event.
                    let t_next = bounces
                        .get(i + 2)
                        .map(|&(t, _)| t)
                        .into_iter()
                        .chain(strikes.iter().copied().filter(|&ts| ts > t2))
                        .fold(f64::INFINITY, f64::min);
                    if t_next - t2 < 0.18 {
                        continue;
                    }
                    if strikes.iter().any(|&ts| ts > t1 - 0.45 && ts < t2 + 0.45) {
                        continue;
                    }
                    let side = |lo: f64, hi: f64, rail: u8| -> Vec<(f64, DVec3)> {
                        track
                            .iter()
                            .filter(|&&(t, pos)| {
                                t >= lo && t <= hi && rail_dist(pos, rail, bounds) > 0.08
                            })
                            .cloned()
                            .collect()
                    };
                    let pre = side((t1 - 0.45).max(t_prev + 0.03), t1 - 0.05, r1);
                    let post = side(t2 + 0.04, (t2 + 0.34).min(t_next - 0.04), r2);
                    let (Some((v_pre, tm_pre, x_pre)), Some((v_post, _, x_post))) =
                        (window_velocity(&pre), window_velocity(&post))
                    else {
                        continue;
                    };
                    if x_pre > 0.02 || x_post > 0.02 {
                        continue;
                    }
                    let speed = v_pre.length();
                    let v_in = v_pre * ((speed - decel * (t1 - tm_pre)).max(0.05) / speed);
                    if v_in.dot(rail_normal(r1)) < 0.4 || v_in.length() > 4.5 {
                        continue;
                    }
                    if v_post.length() > 1.1 * v_in.length() {
                        continue;
                    }
                    events.push(ChainEvent {
                        game: gi,
                        rail1: r1,
                        rail2: r2,
                        gap,
                        v_in,
                        v_out_meas: v_post,
                        post_ts: post.iter().map(|&(t, _)| t - t2).collect(),
                        // Cue carries english; an object that already bounced
                        // picked some up from cushion friction. A virgin
                        // object striker's first chain is spin-clean.
                        nuisance: is_cue || i > 0,
                    });
                }
            }
        }
    }

    let all: Vec<&ChainEvent> = events.iter().collect();
    println!(
        "{} chain candidates → {} clean two-bounce chains (median gap {:.2} s)\n",
        n_pairs,
        events.len(),
        {
            let mut g: Vec<f64> = events.iter().map(|e| e.gap).collect();
            g.sort_by(|a, b| a.total_cmp(b));
            g.get(g.len() / 2).copied().unwrap_or(0.0)
        }
    );
    if events.len() < 30 {
        println!("too few chains to conclude anything");
        return;
    }

    // The four configs that decide the hypothesis. Isolated-bounce reference
    // scale (cushion_events, same operator): rms ≈ 0.07–0.15 m/s.
    let mk = |e: f64, f: f64, s: f64, steps: u32| PhysicsParams {
        cushion_restitution: e,
        cushion_friction: f,
        cushion_restitution_slope: s,
        cushion_contact_steps: steps,
        ..PhysicsParams::carom_calibrated()
    };
    println!("two-bounce chain residuals (rms m/s · angle° · model-never-reaches-rail-2):");
    for (label, tune) in [
        ("Han  event-true e 0.843 f 0.42", mk(0.843, 0.42, -0.038, 0)),
        ("Han  production e 0.910 f 0.30", mk(0.910, 0.30, 0.0, 0)),
        ("Mathavan(60) event-true       ", mk(0.843, 0.42, -0.038, 60)),
        ("Mathavan(60) production       ", mk(0.910, 0.30, 0.0, 60)),
    ] {
        let (rms, ang, unreached) = residuals(&all, &ball, &bases, &tune, &table);
        println!("  {label}: {rms:.3} · {ang:.1}° · {unreached}/{}", all.len());
    }

    // Where do CHAINS want the Han map? If this lands near the isolated-fit
    // values, exit state is fine; if it runs to the bilevel values (or past
    // them), the compensation is localized here.
    let mut best = (0.85, 0.3, f64::INFINITY);
    let mut e = 0.70;
    while e <= 0.99 {
        let mut f = 0.10;
        while f <= 0.55 {
            let (rms, _, _) = residuals(&all, &ball, &bases, &mk(e, f, 0.0, 0), &table);
            if rms < best.2 {
                best = (e, f, rms);
            }
            f += 0.03;
        }
        e += 0.01;
    }
    println!(
        "\nchain-fit (Han, slope 0): e_c {:.3} f_c {:.2} · rms {:.3} m/s  (isolated-bounce fit was e_c ≈ 0.84)",
        best.0, best.1, best.2
    );

    // Fit the chain delta with the isolated-bounce truth pinned: the smallest
    // model that lets BOTH measurements be right at once.
    let mut bestd = (0.0, f64::INFINITY);
    let mut d = 0.0;
    while d <= 0.24 {
        let tune = PhysicsParams {
            cushion_restitution_chain: d,
            ..mk(0.843, 0.42, -0.038, 0)
        };
        let (rms, _, _) = residuals(&all, &ball, &bases, &tune, &table);
        if rms < bestd.1 {
            bestd = (d, rms);
        }
        d += 0.01;
    }
    println!(
        "chain-delta fit (e_c pinned 0.843): delta {:.2} · rms {:.3} m/s",
        bestd.0, bestd.1
    );

    // Gap dependence: exit-state error should hurt SHORT gaps (arrives still
    // sliding on the model's wrong spin) more than long ones.
    for (lo, hi) in [(0.12, 0.20), (0.20, 0.27), (0.27, 0.35)] {
        let evs: Vec<&ChainEvent> =
            events.iter().filter(|e| e.gap >= lo && e.gap < hi).collect();
        if evs.len() < 20 {
            continue;
        }
        let (r_ev, _, u_ev) = residuals(&evs, &ball, &bases, &mk(0.843, 0.42, -0.038, 0), &table);
        let (r_pr, _, u_pr) = residuals(&evs, &ball, &bases, &mk(0.910, 0.30, 0.0, 0), &table);
        println!(
            "  gap [{lo:.2},{hi:.2}) s (n {}): event-true rms {r_ev:.3} (unreach {u_ev}) · production rms {r_pr:.3} (unreach {u_pr})",
            evs.len()
        );
    }
}
