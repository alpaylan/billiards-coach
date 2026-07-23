//! Event-local cushion calibration: fit the cushion map to directly observed
//! bounce in/out pairs instead of end-to-end trajectory error.
//!
//! The bilevel calibration (fit actions per shot, tune physics on whole-shot
//! RMS) assigns credit through a chaotic full simulation — noisy, slow, and
//! able to absorb unrelated model error into the cushion parameters. But the
//! corpus *contains the cushion map directly*: every confident bounce V in a
//! track has an approach velocity and a departure velocity. This example
//! harvests the clean subset and fits (restitution, friction, restitution
//! slope) against those pairs alone — no shot simulation in the loop,
//! unambiguous credit.
//!
//! Two populations, one map:
//!
//! - **Object balls** are struck with ~no sidespin and reach natural roll
//!   within ~0.3 s, after which their full state (velocity AND spin) is
//!   determined by velocity — the fully-known-input condition.
//! - **Cue balls** carry english (ω_z), which is invisible in a track: pure
//!   z-spin has zero contact-point slip contribution, so it bends nothing on
//!   the cloth and shows up ONLY in the cushion rebound. After ≥0.35 s of
//!   free rolling ω_xy is natural roll, leaving ω_z as a single per-bounce
//!   nuisance parameter, identified (or not) by the rebound itself: one
//!   unknown against a 2D measured departure.
//!
//! Fitting the shared map on both populations — and on each alone — is the
//! direct test of whether one parameter set can serve both regimes (the
//! bilevel calibration wants e_c ≈ 0.91; object bounces alone want ≈ 0.83).
//!
//! Gates (both populations): moving ≥0.35 s since the ball's last event
//! (rolling, not sliding), bounce isolated from the ball's other bounces by
//! 0.6 s and from ball-ball contacts by 0.5 s, ≥3 straight samples ≥8 cm off
//! the rail per side (cross-track RMS < 2 cm — kinks mean a missed contact),
//! firm approach (v⊥ ≥ 0.4 m/s), physically sane apparent restitution.
//!
//! **Same-operator comparison.** Right after a cushion the ball's spin is
//! wrong for its new direction, so it SLIDES — decelerating ~20× faster than
//! rolling — through the early post-window. A raw velocity fit there
//! understates the true rebound speed, by an amount depending on the very
//! parameters being fit. So the model is pushed through the identical
//! measurement operator: rebound → free motion → sample at the SAME
//! post-window times → same linear fit. Both sides of the residual are then
//! "what a 30 fps tracker would measure", and the slide loss cancels.
//!
//!   cargo run -p billiards-solver --example cushion_events --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams, TableSpec};
use billiards_engine::{cushion_rebound, simulate_free};
use billiards_solver::fit::{FirstMove, ObservedEvents};
use billiards_solver::measure::window_velocity;
use billiards_solver::shotfile;
use rayon::prelude::*;

/// One harvested bounce: the model inputs (approach state at contact), the
/// measured departure (in measurement space), and the post-window sample times
/// (relative to the bounce) the measurement operator must reproduce.
struct BounceEvent {
    rail: u8, // 0 = -x, 1 = +x, 2 = -y, 3 = +y (fit.rs encoding)
    v_in: DVec3,
    /// Linear-fit velocity over the post window — includes post-bounce slide
    /// loss; compare only against the model pushed through the same operator.
    v_out_meas: DVec3,
    /// Post-window sample times relative to the bounce.
    post_ts: Vec<f64>,
    /// Cue ball: unknown english → ω_z is fitted per bounce (nuisance).
    is_cue: bool,
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

fn load_json_field(dir: &str, key: &str) -> Option<f64> {
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

/// The model's prediction of what the tracker would MEASURE after this bounce:
/// rebound under `phys` with the given english, free motion (slide → roll),
/// sampled at the event's own post-window times, same linear fit.
fn predict_measured(
    ev: &BounceEvent,
    omega_z: f64,
    ball: &BallSpec,
    phys: &PhysicsParams,
    table: &TableSpec,
) -> Option<DVec3> {
    let state = BallState {
        pos: DVec3::ZERO,
        vel: ev.v_in,
        angular_vel: DVec3::new(-ev.v_in.y, ev.v_in.x, 0.0) / ball.radius
            + DVec3::new(0.0, 0.0, omega_z),
    };
    let out = cushion_rebound(state, rail_normal(ev.rail), ball, phys, table.cushion_height, 0.0);
    let traj = simulate_free(out, ball, phys);
    let pts: Vec<(f64, DVec3)> = ev.post_ts.iter().map(|&dt| (dt, traj.state_at(dt).pos)).collect();
    window_velocity(&pts).map(|(v, _, _)| v)
}

/// Squared measured-velocity error and absolute departure-angle error of the
/// event under `phys` — for cue events, minimized over the per-bounce english
/// nuisance (coarse ±72 rad/s then a 1 rad/s refinement). Returns
/// (sq_err, angle_err, omega_z).
fn event_residual(
    ev: &BounceEvent,
    ball: &BallSpec,
    phys: &PhysicsParams,
    table: &TableSpec,
) -> Option<(f64, f64, f64)> {
    let score = |wz: f64| -> Option<f64> {
        let p = predict_measured(ev, wz, ball, phys, table)?;
        Some((DVec3::new(p.x, p.y, 0.0) - DVec3::new(ev.v_out_meas.x, ev.v_out_meas.y, 0.0)).length_squared())
    };
    let best_wz = if ev.is_cue {
        let mut best = (0.0, f64::INFINITY);
        let mut wz = -72.0;
        while wz <= 72.0 {
            if let Some(se) = score(wz) {
                if se < best.1 {
                    best = (wz, se);
                }
            }
            wz += 8.0;
        }
        let center = best.0;
        let mut wz = center - 8.0;
        while wz <= center + 8.0 {
            if let Some(se) = score(wz) {
                if se < best.1 {
                    best = (wz, se);
                }
            }
            wz += 1.0;
        }
        best.0
    } else {
        0.0
    };
    let p = predict_measured(ev, best_wz, ball, phys, table)?;
    let se = (DVec3::new(p.x, p.y, 0.0) - DVec3::new(ev.v_out_meas.x, ev.v_out_meas.y, 0.0)).length_squared();
    let mut d = (p.y.atan2(p.x) - ev.v_out_meas.y.atan2(ev.v_out_meas.x)).abs();
    if d > std::f64::consts::PI {
        d = std::f64::consts::TAU - d;
    }
    Some((se, d, best_wz))
}

/// Mean squared measured-velocity error (m²/s²) and mean absolute departure
/// angle error (radians) of `phys` over `events`.
fn residuals(
    events: &[&BounceEvent],
    ball: &BallSpec,
    phys: &PhysicsParams,
    table: &TableSpec,
) -> (f64, f64) {
    // Per-event squared error capped at (0.3 m/s)²: one corrupted harvest
    // must not own the fit (the collision instrument's hard-won lesson).
    let (se, ae) = events
        .par_iter()
        .filter_map(|ev| event_residual(ev, ball, phys, table).map(|(s, a, _)| (s.min(0.09), a)))
        .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
    let n = events.len().max(1) as f64;
    (se / n, ae / n)
}

/// Two-stage grid fit of (e_c, f_c, slope) under the Han map: coarse sweep,
/// then a fine local refinement around the coarse winner.
fn fit_map(events: &[&BounceEvent], ball: &BallSpec, table: &TableSpec) -> (f64, f64, f64, f64, f64) {
    let eval = |e_c: f64, f_c: f64, slope: f64| -> (f64, f64) {
        let phys = PhysicsParams {
            cushion_restitution: e_c,
            cushion_friction: f_c,
            cushion_restitution_slope: slope,
            ..PhysicsParams::carom_calibrated()
        };
        residuals(events, ball, &phys, table)
    };
    let mut best = (0.9, 0.2, 0.0, f64::INFINITY, 0.0);
    let consider = |e: f64, f: f64, s: f64, best: &mut (f64, f64, f64, f64, f64)| {
        let (se, ae) = eval(e, f, s);
        if se < best.3 {
            *best = (e, f, s, se, ae);
        }
    };
    for ei in 0..=39 {
        let e_c = 0.60 + 0.01 * ei as f64;
        for fi in 0..=19 {
            let f_c = 0.02 + 0.02 * fi as f64;
            for slope in [-0.10, -0.05, 0.0, 0.05] {
                consider(e_c, f_c, slope, &mut best);
            }
        }
    }
    let coarse = best;
    for ei in -4i32..=4 {
        let e_c = (coarse.0 + 0.0025 * ei as f64).clamp(0.5, 0.99);
        for fi in -4i32..=4 {
            let f_c = (coarse.1 + 0.005 * fi as f64).max(0.0);
            for si in -2i32..=2 {
                consider(e_c, f_c, coarse.2 + 0.0125 * si as f64, &mut best);
            }
        }
    }
    best
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let bounds = table.center_bounds(ball.radius);
    const G: f64 = 9.81;

    let mut events: Vec<BounceEvent> = Vec::new();
    let mut n_candidates = 0usize;

    for dir in &dirs {
        let decel = load_json_field(dir, "mu_roll").unwrap_or(PhysicsParams::carom_calibrated().mu_roll) * G;
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
            // Ball-ball contact proxies: every object's verified first move.
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
                // The ball's own motion start: launch for the cue, the strike
                // for an object (which must have verifiably moved).
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
                for (bi, &(t_v, rail)) in bounces.iter().enumerate() {
                    n_candidates += 1;
                    // ASYMMETRIC isolation. The rolling assumption is only
                    // needed on the INCOMING side (the model produces the
                    // outgoing state, and the same-operator pipeline handles
                    // its slide) — so demand a long free approach but only
                    // enough outgoing samples before the next event. This is
                    // what admits the fast mid-rally bounces the symmetric
                    // ±0.6 s harvest excluded, i.e. the regime that decides
                    // rulings.
                    let t_prev = bounces
                        .iter()
                        .enumerate()
                        .filter(|&(j, &(t2, _))| j != bi && t2 < t_v)
                        .map(|(_, &(t2, _))| t2)
                        .chain(strikes.iter().copied().filter(|&ts| ts < t_v))
                        .fold(t_move, f64::max);
                    let t_next = bounces
                        .iter()
                        .enumerate()
                        .filter(|&(j, &(t2, _))| j != bi && t2 > t_v)
                        .map(|(_, &(t2, _))| t2)
                        .chain(strikes.iter().copied().filter(|&ts| ts > t_v))
                        .fold(f64::INFINITY, f64::min);
                    if t_v - t_prev < 0.35 || t_next - t_v < 0.18 {
                        continue;
                    }
                    let side = |lo: f64, hi: f64| -> Vec<(f64, DVec3)> {
                        track
                            .iter()
                            .filter(|&&(t, pos)| {
                                t >= lo && t <= hi && rail_dist(pos, rail, bounds) > 0.08
                            })
                            .cloned()
                            .collect()
                    };
                    let pre = side((t_v - 0.45).max(t_prev + 0.03), t_v - 0.05);
                    let post = side(t_v + 0.04, (t_v + 0.34).min(t_next - 0.04));
                    let (Some((v_pre, tm_pre, x_pre)), Some((v_post, _, x_post))) =
                        (window_velocity(&pre), window_velocity(&post))
                    else {
                        continue;
                    };
                    // Straight-line windows only: a kink means an unnoticed
                    // contact inside the window, not a clean bounce pair.
                    if x_pre > 0.02 || x_post > 0.02 {
                        continue;
                    }
                    // Approach velocity AT the contact: the pre-side ball is
                    // rolling (gated), so a rolling-deceleration correction
                    // from the window midpoint is nearly exact.
                    let speed = v_pre.length();
                    let v_in = v_pre * ((speed - decel * (t_v - tm_pre)).max(0.05) / speed);
                    let n = rail_normal(rail);
                    let perp_in = v_in.dot(n);
                    let perp_out = -v_post.dot(n);
                    let e_apparent = perp_out / perp_in.max(1e-6);
                    if perp_in < 0.4 || v_in.length() > 4.5 {
                        continue;
                    }
                    // Unphysical pairs are extraction failures (mis-paired V,
                    // identity swap), not data. English can add tangential
                    // speed, so the cue's speed-ratio bound is looser.
                    let ratio_max = if is_cue { 1.25 } else { 1.05 };
                    if !(0.15..=1.05).contains(&e_apparent)
                        || v_post.length() > ratio_max * v_in.length()
                    {
                        continue;
                    }
                    events.push(BounceEvent {
                        rail,
                        v_in,
                        v_out_meas: v_post,
                        post_ts: post.iter().map(|&(t, _)| t - t_v).collect(),
                        is_cue,
                    });
                }
            }
        }
    }

    let objects: Vec<&BounceEvent> = events.iter().filter(|e| !e.is_cue).collect();
    let cues: Vec<&BounceEvent> = events.iter().filter(|e| e.is_cue).collect();
    println!(
        "{} bounce candidates → {} clean events ({} object-ball, {} cue-ball)\n",
        n_candidates,
        events.len(),
        objects.len(),
        cues.len()
    );

    // Apparent restitution vs approach speed, straight off the measurements
    // (object balls — no english confound). "Apparent" = includes post-bounce
    // slide loss, so it sits well below the map's e_c; its SPEED TREND is
    // still the model-free EC_SLOPE readout.
    println!("object-ball apparent restitution (incl. slide loss) vs approach speed:");
    let bins = [(0.4, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.5)];
    for (lo, hi) in bins {
        let es: Vec<f64> = objects
            .iter()
            .filter(|e| {
                let pi = e.v_in.dot(rail_normal(e.rail));
                pi >= lo && pi < hi
            })
            .map(|e| -e.v_out_meas.dot(rail_normal(e.rail)) / e.v_in.dot(rail_normal(e.rail)))
            .collect();
        if es.is_empty() {
            continue;
        }
        let n = es.len() as f64;
        let m = es.iter().sum::<f64>() / n;
        let sd = (es.iter().map(|e| (e - m) * (e - m)).sum::<f64>() / n).sqrt();
        println!("  v⊥ [{lo:.2},{hi:.2}) m/s: e_app = {m:.3} ± {sd:.3}  (n {})", es.len());
    }

    // Population fits: objects alone, cues alone (english as nuisance), and
    // the joint compromise — the direct one-map-two-regimes test.
    let report = |label: &str, evs: &[&BounceEvent]| -> (f64, f64, f64) {
        let (e, f, s, se, ae) = fit_map(evs, &ball, &table);
        println!(
            "{label} ({} events): e_c {e:.3} f_c {f:.3} slope {s:+.3} · rms {:.3} m/s · angle err {:.1}°",
            evs.len(),
            se.sqrt(),
            ae.to_degrees()
        );
        (e, f, s)
    };
    println!();
    let (eo, fo, so) = report("object-fit", &objects);
    let (ec_cue, fc_cue, s_cue) = report("cue-fit   ", &cues);
    let (ej, fj, sj) = report("joint-fit ", &events.iter().collect::<Vec<_>>());

    // Cross-population residuals: each population under its own fit, the other
    // population's fit, and the joint fit — the misfit, quantified.
    let phys_of = |e: f64, f: f64, s: f64| PhysicsParams {
        cushion_restitution: e,
        cushion_friction: f,
        cushion_restitution_slope: s,
        ..PhysicsParams::carom_calibrated()
    };
    println!("\ncross-population residuals (rms m/s · angle°):");
    for (label, e, f, s) in [
        ("object-fit params", eo, fo, so),
        ("cue-fit params   ", ec_cue, fc_cue, s_cue),
        ("joint-fit params ", ej, fj, sj),
        ("bilevel e_c 0.91 ", 0.910, 0.25, 0.0),
    ] {
        let ph = phys_of(e, f, s);
        let (so_, ao) = residuals(&objects, &ball, &ph, &table);
        let (sc, ac) = residuals(&cues, &ball, &ph, &table);
        println!(
            "  {label}: objects {:.3} · {:.1}°   cues {:.3} · {:.1}°",
            so_.sqrt(),
            ao.to_degrees(),
            sc.sqrt(),
            ac.to_degrees()
        );
    }

    // REGIME MAP: effective e fitted per speed / obliquity bin (f_c and the
    // slopes held at the joint values) — the direct answer to "is the
    // fast/oblique regime more elastic than the slow perpendicular one?".
    let fit_e_only = |evs: &[&BounceEvent], f_c: f64| -> (f64, f64) {
        let mut best = (0.9, f64::INFINITY);
        let mut e = 0.60;
        while e <= 0.99 {
            let phys = PhysicsParams {
                cushion_restitution: e,
                cushion_friction: f_c,
                ..PhysicsParams::carom_calibrated()
            };
            let (se, _) = residuals(evs, &ball, &phys, &table);
            if se < best.1 {
                best = (e, se);
            }
            e += 0.005;
        }
        best
    };
    println!("\nregime map (e fitted per bin, f_c {fj:.2}):", fj = fj);
    for (lo, hi) in [(0.4, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 4.5)] {
        let evs: Vec<&BounceEvent> = events
            .iter()
            .filter(|e| {
                let vn = e.v_in.dot(rail_normal(e.rail));
                vn >= lo && vn < hi
            })
            .collect();
        if evs.len() < 25 {
            continue;
        }
        let (e, se) = fit_e_only(&evs, fj);
        println!("  v_n [{lo:.1},{hi:.1}) m/s: e {e:.3}  (n {}, rms {:.3})", evs.len(), se.sqrt());
    }
    for (lo, hi, label) in [(0.0, 25.0, "θ <25°"), (25.0, 50.0, "θ 25-50°"), (50.0, 90.0, "θ >50°")] {
        let evs: Vec<&BounceEvent> = events
            .iter()
            .filter(|e| {
                let n = rail_normal(e.rail);
                let vn = e.v_in.dot(n);
                let vt = (e.v_in - n * vn).length();
                let th = vt.atan2(vn).to_degrees();
                th >= lo && th < hi && vn >= 0.4
            })
            .collect();
        if evs.len() < 25 {
            continue;
        }
        let (e, se) = fit_e_only(&evs, fj);
        println!("  {label}: e {e:.3}  (n {}, rms {:.3})", evs.len(), se.sqrt());
    }

    // Parametric regime fit: e_eff = e_c + s_n·(v_n−1) + s_t·|v_t| — does the
    // extended model beat the flat/slope-only fits on the same events?
    let mut bestp = (0.85, 0.0, 0.0, 0.42, f64::INFINITY);
    for f_c in [0.30, 0.42] {
        let mut e = 0.70;
        while e <= 0.99 {
            for s_n in [-0.06, -0.03, 0.0, 0.03] {
                for s_t in [0.0, 0.02, 0.04, 0.06, 0.08] {
                    let phys = PhysicsParams {
                        cushion_restitution: e,
                        cushion_friction: f_c,
                        cushion_restitution_slope: s_n,
                        cushion_restitution_slope_t: s_t,
                        ..PhysicsParams::carom_calibrated()
                    };
                    let (se, _) = residuals(&events.iter().collect::<Vec<_>>(), &ball, &phys, &table);
                    if se < bestp.4 {
                        bestp = (e, s_n, s_t, f_c, se);
                    }
                }
            }
            e += 0.01;
        }
    }
    println!(
        "\nparametric regime fit: e_c {:.3} s_n {:+.3} s_t {:+.3} f_c {:.2} · rms {:.3} m/s",
        bestp.0,
        bestp.1,
        bestp.2,
        bestp.3,
        bestp.4.sqrt()
    );

    // English sanity under the cue-fit: the fitted nuisance should look like
    // real english (roughly centered, mostly |ω_z| < 60 rad/s), not like a
    // parameter soaking up model error at the search bounds.
    let ph = phys_of(ec_cue, fc_cue, s_cue);
    let wzs: Vec<f64> = cues
        .par_iter()
        .filter_map(|ev| event_residual(ev, &ball, &ph, &table).map(|(_, _, w)| w))
        .collect();
    if !wzs.is_empty() {
        let n = wzs.len() as f64;
        let m = wzs.iter().sum::<f64>() / n;
        let sd = (wzs.iter().map(|w| (w - m) * (w - m)).sum::<f64>() / n).sqrt();
        let at_edge = wzs.iter().filter(|w| w.abs() > 64.0).count();
        println!(
            "\nfitted english over cue bounces: mean {m:+.1} rad/s · sd {sd:.1} · |ω_z|>64: {at_edge}/{}",
            wzs.len()
        );
    }
}
