//! Event-local BALL-BALL calibration: fit the collision model to directly
//! observed strike in/out pairs — the instrument aimed at the energy leak the
//! inflated bilevel cushion restitution compensates for (see cushion_events).
//!
//! A usable collision: a **striker** that rolled freely ≥0.35 s (state known
//! up to english — ω_xy is natural roll; ω_z is a per-event nuisance for cue
//! strikers and post-cushion objects, zero for virgin object strikers) hits a
//! ball **verifiably at rest** (position known from its own pre-hit samples).
//! Observables: the striker's measured departure AND the struck ball's
//! measured departure — four numbers against (e_b, μ_b) + at most one
//! nuisance. Contact geometry (line of centers) comes from the striker's
//! pre-window line extrapolated to center distance 2R from the rest.
//!
//! **Same-operator comparison** (as in cushion_events): both balls SLIDE
//! immediately after impact (the struck ball starts with no spin at all), so
//! the model's post-collision states are run through free motion, sampled at
//! each ball's real frame times, and estimated with the same linear fit the
//! measurement used. Residuals live entirely in measurement space.
//!
//! Gates: exactly one attributable striker, impact parameter ≤ 0.85·2R (thin
//! grazes are mm-sensitive), firm normal approach (v_n ≥ 0.35 m/s), both
//! post-windows straight (cross-track RMS < 2 cm), no cushion of either ball
//! and no other strike within ±0.5 s, third ball ≥ 20 cm away throughout.
//!
//!   cargo run -p billiards-solver --example collision_events --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams};
use billiards_engine::{ball_ball_collision, simulate_free};
use billiards_solver::fit::{FirstMove, ObservedEvents};
use billiards_solver::measure::{window_line, window_velocity};
use billiards_solver::shotfile;
use rayon::prelude::*;

/// One harvested collision, everything in measurement space.
struct StrikeEvent {
    /// Index into the per-game physics bases (operator cloth params).
    game: usize,
    /// Striker center at contact and its velocity there (rolling-corrected).
    contact: DVec3,
    v_in: DVec3,
    /// The struck ball's rest position.
    rest: DVec3,
    /// Measured post-window velocities (include post-impact slide loss).
    v_striker_meas: DVec3,
    v_struck_meas: DVec3,
    /// Post-window sample times relative to contact, per ball.
    striker_ts: Vec<f64>,
    struck_ts: Vec<f64>,
    /// English unknown → fit ω_z per event (cue striker or post-cushion object).
    nuisance: bool,
    is_cue_striker: bool,
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

fn load_mu_roll(dir: &str) -> f64 {
    load_field(dir, "mu_roll").unwrap_or(PhysicsParams::carom_calibrated().mu_roll)
}

/// Per-game operator physics: the game's own cloth friction (the measurement
/// operator simulates post-impact slide/roll, so wrong cloth biases every
/// prediction the same direction — this was the source of a uniform ~20%
/// over-prediction when a single global cloth was used).
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

/// Predicted measured departures (striker, struck) under `phys` and english
/// `omega_z`, pushed through the same measurement operator.
fn predict(
    ev: &StrikeEvent,
    omega_z: f64,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> Option<(DVec3, DVec3)> {
    // `phys` must already carry this event's game cloth (see with_game).
    let r = ball.radius;
    let striker = BallState {
        pos: DVec3::new(ev.contact.x, ev.contact.y, r),
        vel: DVec3::new(ev.v_in.x, ev.v_in.y, 0.0),
        angular_vel: DVec3::new(-ev.v_in.y, ev.v_in.x, 0.0) / r + DVec3::new(0.0, 0.0, omega_z),
    };
    let struck = BallState {
        pos: DVec3::new(ev.rest.x, ev.rest.y, r),
        vel: DVec3::ZERO,
        angular_vel: DVec3::ZERO,
    };
    let (a, b) = ball_ball_collision(striker, struck, ball, phys);
    let ta = simulate_free(a, ball, phys);
    let tb = simulate_free(b, ball, phys);
    let sample = |traj: &billiards_core::Trajectory, ts: &[f64]| -> Option<DVec3> {
        let pts: Vec<(f64, DVec3)> = ts.iter().map(|&dt| (dt, traj.state_at(dt).pos)).collect();
        window_velocity(&pts).map(|(v, _, _)| v)
    };
    Some((sample(&ta, &ev.striker_ts)?, sample(&tb, &ev.struck_ts)?))
}

/// Event residual under `phys`: summed squared measured-velocity error over
/// both balls, minimized over the english nuisance where flagged. Returns
/// (sq_err, struck_speed_ratio_pred_over_meas, omega_z).
fn event_residual(
    ev: &StrikeEvent,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> Option<(f64, f64, f64, f64)> {
    let score = |wz: f64| -> Option<f64> {
        let (ps, pk) = predict(ev, wz, ball, phys)?;
        Some(
            (DVec3::new(ps.x - ev.v_striker_meas.x, ps.y - ev.v_striker_meas.y, 0.0))
                .length_squared()
                + (DVec3::new(pk.x - ev.v_struck_meas.x, pk.y - ev.v_struck_meas.y, 0.0))
                    .length_squared(),
        )
    };
    let best_wz = if ev.nuisance {
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
    let (ps, pk) = predict(ev, best_wz, ball, phys)?;
    let se = (DVec3::new(ps.x - ev.v_striker_meas.x, ps.y - ev.v_striker_meas.y, 0.0))
        .length_squared()
        + (DVec3::new(pk.x - ev.v_struck_meas.x, pk.y - ev.v_struck_meas.y, 0.0)).length_squared();
    let ratio = pk.length() / ev.v_struck_meas.length().max(1e-6);
    let ratio_s = ps.length() / ev.v_striker_meas.length().max(1e-6);
    Some((se, ratio, ratio_s, best_wz))
}

/// Event's effective physics: collision params from `tune`, cloth from the
/// event's game base scaled by `mu_scale` (shared cloth-error correction).
fn with_game(ev: &StrikeEvent, bases: &[PhysicsParams], tune: &PhysicsParams, mu_scale: f64) -> PhysicsParams {
    let base = &bases[ev.game];
    PhysicsParams {
        mu_slide: base.mu_slide * mu_scale,
        mu_roll: base.mu_roll,
        ..tune.clone()
    }
}

/// Robust residuals: per-event squared error CAPPED at (0.3 m/s)² so a single
/// corrupted harvest (a mis-attributed strike, a blurred slam) cannot own the
/// fit, and MEDIAN pred/meas speed ratios (a mean over ratios is a tail
/// statistic). Returns (mean capped sq err, median struck ratio, median
/// striker ratio).
fn residuals(
    events: &[&StrikeEvent],
    ball: &BallSpec,
    bases: &[PhysicsParams],
    tune: &PhysicsParams,
    mu_scale: f64,
) -> (f64, f64, f64) {
    const CAP: f64 = 0.09;
    let per: Vec<(f64, f64, f64)> = events
        .par_iter()
        .filter_map(|ev| {
            let phys = with_game(ev, bases, tune, mu_scale);
            event_residual(ev, ball, &phys).map(|(s, r, rs, _)| (s.min(CAP), r, rs))
        })
        .collect();
    if per.is_empty() {
        return (f64::INFINITY, 1.0, 1.0);
    }
    let med = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let se = per.iter().map(|p| p.0).sum::<f64>() / per.len() as f64;
    (se, med(per.iter().map(|p| p.1).collect()), med(per.iter().map(|p| p.2).collect()))
}

/// Two-stage grid fit of (e_b, μ_b, μ_s-scale) with the constant friction
/// curve. The μ_s scale multiplies each game's own cloth sliding friction in
/// the measurement operator — a shared correction for the cloth the bilevel
/// calibration itself may have mis-fit.
fn fit_map(
    events: &[&StrikeEvent],
    ball: &BallSpec,
    bases: &[PhysicsParams],
    steps: u32,
) -> (f64, f64, f64, f64, f64) {
    let eval = |e_b: f64, mu: f64, ms: f64| -> (f64, f64) {
        let tune = PhysicsParams {
            ball_restitution: e_b,
            ball_friction: mu,
            ball_friction_b: 0.0,
            ball_contact_steps: steps,
            ..PhysicsParams::carom_calibrated()
        };
        let (se, ratio, _) = residuals(events, ball, bases, &tune, ms);
        (se, ratio)
    };
    let mut best = (0.95, 0.06, 1.0, f64::INFINITY, 1.0);
    let consider = |e: f64, m: f64, ms: f64, best: &mut (f64, f64, f64, f64, f64)| {
        let (se, ratio) = eval(e, m, ms);
        if se < best.3 {
            *best = (e, m, ms, se, ratio);
        }
    };
    for ei in 0..=20 {
        let e_b = 0.80 + 0.01 * ei as f64;
        for mi in 0..=20 {
            for ms in [0.8, 1.0, 1.2, 1.4] {
                consider(e_b, 0.02 * mi as f64, ms, &mut best);
            }
        }
    }
    let coarse = best;
    for ei in -4i32..=4 {
        let e_b = (coarse.0 + 0.0025 * ei as f64).clamp(0.5, 1.0);
        for mi in -4i32..=4 {
            for si in -2i32..=2 {
                consider(
                    e_b,
                    (coarse.1 + 0.005 * mi as f64).max(0.0),
                    (coarse.2 + 0.1 * si as f64).max(0.5),
                    &mut best,
                );
            }
        }
    }
    best
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let ball = BallSpec::carom();
    let table = billiards_core::TableSpec::carom_match();
    let bounds = table.center_bounds(ball.radius);
    let r = ball.radius;
    const G: f64 = 9.81;

    let mut events: Vec<StrikeEvent> = Vec::new();
    let mut n_candidates = 0usize;
    // Where candidates die, in gate order — the harvest's own diagnostics.
    let (mut d_rest, mut d_strk, mut d_geom, mut d_roll, mut d_iso, mut d_win, mut d_sane) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);

    let mut bases: Vec<PhysicsParams> = Vec::new();
    for (gi, dir) in dirs.iter().enumerate() {
        let decel = load_mu_roll(dir) * G;
        bases.push(game_base(dir));
        let _ = gi;
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
            let first_moves: Vec<Option<f64>> = obs
                .first_move
                .iter()
                .map(|fm| match fm {
                    FirstMove::At(t) => Some(*t),
                    _ => None,
                })
                .collect();
            for (k, t_hit) in first_moves.iter().enumerate() {
                let Some(t_hit) = *t_hit else { continue };
                n_candidates += 1;
                let struck_ti = k + 1;
                let struck_track = &shot.observed[struck_ti];

                // The struck ball's rest: stationary pre-hit samples.
                let pre_rest: Vec<DVec3> = struck_track
                    .iter()
                    .filter(|(t, _)| *t >= t_hit - 0.50 && *t <= t_hit - 0.05)
                    .map(|&(_, p)| p)
                    .collect();
                if pre_rest.len() < 3 {
                    d_rest += 1;
                    continue;
                }
                let rest = pre_rest.iter().fold(DVec3::ZERO, |a, p| a + *p) / pre_rest.len() as f64;
                if pre_rest.iter().any(|p| (*p - rest).length() > 0.02) {
                    d_rest += 1;
                    continue; // not verifiably still
                }

                // Attribute the striker: exactly one other ball whose pre-hit
                // line reaches center distance 2R from the rest near t_hit.
                let mut striker: Option<(usize, DVec3, DVec3, f64)> = None; // (ti, contact, v_at_contact, t*)
                let mut ambiguous = false;
                for (j, tr) in shot.observed.iter().enumerate() {
                    if j == struck_ti {
                        continue;
                    }
                    let pre: Vec<(f64, DVec3)> = tr
                        .iter()
                        .filter(|(t, _)| *t >= t_hit - 0.45 && *t <= t_hit - 0.05)
                        .cloned()
                        .collect();
                    let Some((v, tm, cross)) = window_velocity(&pre) else { continue };
                    let Some((_, pm, _)) = window_line(&pre) else { continue };
                    if cross > 0.02 || v.length() < 0.4 {
                        continue;
                    }
                    // |pm + v(t−tm) − rest| = 2R: quadratic in τ = t − tm.
                    let d0 = pm - rest;
                    let (a2, a1, a0) =
                        (v.length_squared(), 2.0 * v.dot(d0), d0.length_squared() - 4.0 * r * r);
                    let disc = a1 * a1 - 4.0 * a2 * a0;
                    if disc < 0.0 {
                        continue; // line never reaches contact distance
                    }
                    let tau = (-a1 - disc.sqrt()) / (2.0 * a2); // approaching root
                    let t_star = tm + tau;
                    if (t_star - t_hit).abs() > 0.12 {
                        continue;
                    }
                    if striker.is_some() {
                        ambiguous = true;
                        break;
                    }
                    // Rolling-deceleration correction of speed to contact time.
                    let speed = v.length();
                    let v_c = v * ((speed - decel * (t_star - tm)).max(0.05) / speed);
                    striker = Some((j, pm + v * tau, v_c, t_star));
                }
                let Some((sj, contact, v_in, t_star)) = striker else {
                    d_strk += 1;
                    continue;
                };
                if ambiguous {
                    d_strk += 1;
                    continue;
                }

                // Impact geometry gates: not too thin, firm normal approach.
                let n_hat = (rest - contact) / (rest - contact).length();
                let v_n = v_in.dot(n_hat);
                let b_impact = (v_in - n_hat * v_n).length() / v_in.length() * 2.0 * r;
                if v_n < 0.35 || b_impact > 0.85 * 2.0 * r || v_in.length() > 3.0 {
                    d_geom += 1;
                    continue;
                }

                // Striker rolled freely ≥0.35 s: no own cushion, no strike, and
                // its own motion start sufficiently long ago.
                let striker_track = &shot.observed[sj];
                let Some(&(_, sp0)) = striker_track.first() else { continue };
                let move_start = striker_track
                    .iter()
                    .find(|(_, p)| (*p - sp0).length() > 0.02)
                    .map(|&(t, _)| t)
                    .unwrap_or(t_hit);
                let mut last_event: f64 = move_start;
                for &(tb, _) in &obs.cushions[sj] {
                    if tb < t_hit - 0.02 {
                        last_event = last_event.max(tb);
                    }
                }
                for (kk, tm2) in first_moves.iter().enumerate() {
                    if kk == k {
                        continue;
                    }
                    if let Some(t2) = tm2 {
                        if *t2 < t_hit - 0.02 {
                            last_event = last_event.max(*t2);
                        }
                    }
                }
                if t_hit - last_event < 0.35 {
                    d_roll += 1;
                    continue;
                }

                // Isolation: no cushion of either ball and no other strike
                // within ±0.5 s; third ball ≥ 20 cm away around the contact.
                let near = |ts: &[(f64, u8)]| ts.iter().any(|&(t, _)| (t - t_hit).abs() < 0.5);
                if near(&obs.cushions[sj]) || near(&obs.cushions[struck_ti]) {
                    d_iso += 1;
                    continue;
                }
                if first_moves
                    .iter()
                    .enumerate()
                    .any(|(kk, t2)| kk != k && t2.is_some_and(|t2| (t2 - t_hit).abs() < 0.5))
                {
                    d_iso += 1;
                    continue;
                }
                let third_near = shot.observed.iter().enumerate().any(|(m, tr)| {
                    m != sj
                        && m != struck_ti
                        && tr.iter().any(|&(t, p)| {
                            (t - t_hit).abs() < 0.45
                                && ((p - rest).length() < 0.20 || (p - contact).length() < 0.20)
                        })
                });
                if third_near {
                    d_iso += 1;
                    continue;
                }

                // Post windows for both balls (same-operator measurement),
                // relative to the GEOMETRIC contact time. POST_LO (default
                // 0.05) sets the window start after contact — raise it to test
                // sensitivity to detection lag on the suddenly-moving struck
                // ball (a stationary ball's tracker may need frames to re-lock
                // once it flies, understating its early velocity).
                let post_lo: f64 =
                    env::var("POST_LO").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);
                let post_of = |tr: &[(f64, DVec3)]| -> Vec<(f64, DVec3)> {
                    tr.iter()
                        .filter(|(t, _)| *t >= t_star + post_lo && *t <= t_star + post_lo + 0.40)
                        .cloned()
                        .collect()
                };
                let (sp, kp) = (post_of(striker_track), post_of(struck_track));
                let (Some((vs, _, xs)), Some((vk, _, xk))) =
                    (window_velocity(&sp), window_velocity(&kp))
                else {
                    d_win += 1;
                    continue;
                };
                if xs > 0.02 || xk > 0.02 {
                    d_win += 1;
                    continue;
                }
                // Physical sanity (equal masses can't speed anything up).
                if vk.length() > 1.1 * v_in.length() || vs.length() > 1.1 * v_in.length() {
                    d_sane += 1;
                    continue;
                }

                let striker_had_cushions = obs.cushions[sj].iter().any(|&(t, _)| t < t_hit);
                events.push(StrikeEvent {
                    game: gi,
                    contact,
                    v_in,
                    rest,
                    v_striker_meas: vs,
                    v_struck_meas: vk,
                    striker_ts: sp.iter().map(|&(t, _)| t - t_star).collect(),
                    struck_ts: kp.iter().map(|&(t, _)| t - t_star).collect(),
                    nuisance: sj == 0 || striker_had_cushions,
                    is_cue_striker: sj == 0,
                });
            }
        }
    }

    let all: Vec<&StrikeEvent> = events.iter().collect();
    let gold: Vec<&StrikeEvent> = events.iter().filter(|e| !e.nuisance).collect();
    let cue: Vec<&StrikeEvent> = events.iter().filter(|e| e.is_cue_striker).collect();
    println!(
        "{} struck-ball candidates → {} clean collisions ({} cue-striker, {} nuisance-free object strikers)\n",
        n_candidates,
        events.len(),
        cue.len(),
        gold.len()
    );
    println!(
        "dropped: rest {d_rest} · striker-attrib {d_strk} · geometry {d_geom} · not-rolling {d_roll} · isolation {d_iso} · windows {d_win} · sanity {d_sane}\n"
    );

    // Model-free audit: apparent e_b from the struck ball's normal departure
    // (equal masses: v_struck·n̂ = (1+e)/2 · v_n) — biased low by post-impact
    // slide, so read the TREND, not the level.
    println!("apparent ball restitution (incl. slide loss) vs normal approach speed:");
    for (lo, hi) in [(0.35, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 3.5)] {
        let es: Vec<f64> = events
            .iter()
            .filter_map(|e| {
                let n_hat = (e.rest - e.contact).normalize();
                let vn = e.v_in.dot(n_hat);
                (vn >= lo && vn < hi).then(|| 2.0 * e.v_struck_meas.dot(n_hat) / vn - 1.0)
            })
            .collect();
        if es.is_empty() {
            continue;
        }
        let n = es.len() as f64;
        let m = es.iter().sum::<f64>() / n;
        let sd = (es.iter().map(|e| (e - m) * (e - m)).sum::<f64>() / n).sqrt();
        println!("  v_n [{lo:.2},{hi:.2}) m/s: e_app = {m:.3} ± {sd:.3}  (n {})", es.len());
    }

    // Bias under the current model: does it over- or under-predict how fast
    // the struck ball actually leaves? (ratio > 1 = model too energetic).
    if env::var("DUMP").is_ok() {
        let cur0 = PhysicsParams::carom_calibrated();
        for (i, ev) in events.iter().enumerate().take(18) {
            let n_hat = (ev.rest - ev.contact).normalize();
            let vn = ev.v_in.dot(n_hat);
            let cut = (1.0 - (vn / ev.v_in.length()).powi(2)).max(0.0).sqrt().asin().to_degrees();
            let phys = with_game(ev, &bases, &cur0, 1.0);
            let (pred_s, pred_k) = predict(ev, 0.0, &ball, &phys).unwrap_or((DVec3::ZERO, DVec3::ZERO));
            println!(
                "  ev{i:02} cue={} v_in {:.2} v_n {:.2} cut {:.0}° | struck meas {:.2} pred {:.2} | striker meas {:.2} pred {:.2} | ts_k {:.2}..{:.2} n{}",
                ev.is_cue_striker as u8,
                ev.v_in.length(),
                vn,
                cut,
                ev.v_struck_meas.length(),
                pred_k.length(),
                ev.v_striker_meas.length(),
                pred_s.length(),
                ev.struck_ts.first().copied().unwrap_or(0.0),
                ev.struck_ts.last().copied().unwrap_or(0.0),
                ev.struck_ts.len(),
            );
        }
        println!();
    }

    let cur = PhysicsParams::carom_calibrated();
    let (se_cur, ratio_cur, ratio_s_cur) = residuals(&all, &ball, &bases, &cur, 1.0);
    println!(
        "\ncurrent model (e_b {:.2}, μ_b {:.2}, per-game cloth): rms {:.3} m/s · pred/meas struck {:.3} · striker {:.3}",
        cur.ball_restitution,
        cur.ball_friction,
        se_cur.sqrt(),
        ratio_cur,
        ratio_s_cur
    );

    for (label, evs, steps) in [
        ("closed-form all-fit", &all, 0u32),
        ("closed-form cue-fit", &cue, 0),
        ("integrated  all-fit", &all, 200),
    ] {
        if evs.len() < 20 {
            println!("{label}: only {} events — skipped", evs.len());
            continue;
        }
        let (e_b, mu, ms, se, ratio) = fit_map(evs, &ball, &bases, steps);
        println!(
            "{label} ({} events): e_b {e_b:.3} μ_b {mu:.3} μ_s×{ms:.2} · rms {:.3} m/s · struck pred/meas {:.3}",
            evs.len(),
            se.sqrt(),
            ratio
        );
    }

    // Integrated-contact probe at the all-fit params: gearing transfers
    // follow/draw into the struck ball, changing BOTH post-impact slide
    // phases — exactly what measurement space sees.
    let (e_b, mu, ms, ..) = fit_map(&all, &ball, &bases, 0);
    let tune_int = PhysicsParams {
        ball_restitution: e_b,
        ball_friction: mu,
        ball_friction_b: 0.0,
        ball_contact_steps: 200,
        ..PhysicsParams::carom_calibrated()
    };
    let (se_i, ratio_i, _) = residuals(&all, &ball, &bases, &tune_int, ms);
    let tune_cf = PhysicsParams { ball_contact_steps: 0, ..tune_int.clone() };
    let (se_c, ratio_c, _) = residuals(&all, &ball, &bases, &tune_cf, ms);
    println!(
        "\nintegrated(200) at all-fit params: rms {:.3} m/s · ratio {:.3}  (closed-form: {:.3} · {:.3})",
        se_i.sqrt(),
        ratio_i,
        se_c.sqrt(),
        ratio_c
    );
}
