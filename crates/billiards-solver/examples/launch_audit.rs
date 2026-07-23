//! Launch-speed audit: is the speed the fit pins to — `launch_estimate`, read
//! from the strike-adjacent frames — systematically LOW?
//!
//! The energy audit (cushion_events, collision_events, roll_decel) showed every
//! passive component is correct-or-less-energetic than the bilevel calibration
//! assumes, while whole-shot reconstruction wants MORE energy. The launch speed
//! is the one energy input left unaudited, and it is measured from the worst
//! frames of the entire shot (motion blur at the strike).
//!
//! Independent reference, per shot with a clean free opening: measure the cue's
//! speed in an early but SHARP window (0.18–0.48 s after it leaves rest, past
//! the strike blur), then extrapolate back to the strike:
//!
//!   - rolling-only correction → a hard LOWER BOUND on the true launch speed
//!     (real deceleration is ≥ rolling). If even the lower bound beats
//!     `launch_estimate` on average, under-measurement is proven one-sidedly.
//!   - when a second window fits before the first event, the measured
//!     deceleration between the windows gives a point estimate (slide-aware).
//!
//! Second, zero-machinery check over ALL shots: where does the fit's CHOSEN
//! speed sit inside the ±22% pin window? A pile-up at the ceiling means the
//! optimizer keeps straining against a pin center that is too low.
//!
//!   cargo run -p billiards-solver --example launch_audit --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_solver::fit::{FitConfig, fit_action, launch_estimate};
use billiards_solver::measure::window_velocity;
use billiards_solver::shotfile;

fn load_calibration(dir: &str) -> PhysicsParams {
    let path = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let Ok(text) = fs::read_to_string(path) else { return PhysicsParams::carom_calibrated() };
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>()
            .parse()
            .ok()
    };
    match (get("cushion_restitution"), get("cushion_friction"), get("mu_slide"), get("mu_roll")) {
        (Some(e), Some(f), Some(s), Some(r)) => PhysicsParams {
            cushion_restitution: e,
            cushion_friction: f,
            mu_slide: s,
            mu_roll: r,
            ..PhysicsParams::carom_calibrated()
        },
        _ => PhysicsParams::carom_calibrated(),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let bounds = table.center_bounds(ball.radius);
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    const G: f64 = 9.81;
    // Free-roll deceleration as measured by roll_decel (0.0053–0.0058 across
    // all five games) — the floor of any real launch-window deceleration.
    const MU_R: f64 = 0.0055;

    // Audit records: (v_est, r_lb, Option<r_pt>) for clean-opening shots.
    let mut audit: Vec<(f64, f64, Option<f64>)> = Vec::new();
    // Pin placement of the fit's chosen speed for every labeled shot.
    let (mut at_ceiling, mut at_floor, mut interior, mut beyond_up, mut beyond_dn, mut n_fit) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);

    // Physics overrides (same env contract as ruling_check) so pin pressure
    // can be compared between parameter sets: if event-true physics needs
    // systematically higher speeds to explain the same tracks, the fits pile
    // up at the window ceiling — the "searchability compensation" signature.
    let envf = |k: &str| env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
    let (ec, fc, ecs, eb, bf) =
        (envf("EC"), envf("FC"), envf("EC_SLOPE"), envf("EB"), envf("BF"));
    for dir in &dirs {
        let mut phys = load_calibration(dir);
        if let Some(v) = ec {
            phys.cushion_restitution = v;
        }
        if let Some(v) = fc {
            phys.cushion_friction = v;
        }
        if let Some(v) = ecs {
            phys.cushion_restitution_slope = v;
        }
        if let Some(v) = eb {
            phys.ball_restitution = v;
        }
        if let Some(v) = bf {
            phys.ball_friction = v;
        }
        let mu_s = phys.mu_slide;
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
            let label = text
                .lines()
                .find_map(|l| l.strip_prefix("result "))
                .map(str::trim)
                .unwrap_or("");
            if label != "make" && label != "miss" {
                continue;
            }
            let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };

            // Reproduce launch_estimate's inputs exactly as fit_action does.
            let mut obstacles: Vec<DVec3> = shot.scene.objects.clone();
            obstacles
                .extend(shot.observed.iter().skip(1).filter_map(|t| t.first().map(|s| s.1)));
            let t_stop = shot
                .observed
                .iter()
                .skip(1)
                .filter_map(|tr| {
                    let &(_, p0) = tr.first()?;
                    tr.iter().find(|(_, p)| (*p - p0).length() > 0.02).map(|&(t, _)| t)
                })
                .fold(f64::INFINITY, f64::min);
            let (_, v_est, _) =
                launch_estimate(&shot.observed[0], &obstacles, t_stop, bounds, ball.radius);

            // Fit-chosen speed vs the pin window (over all labeled shots).
            let f = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
            n_fit += 1;
            let chosen = f.action.speed / v_est;
            if chosen > 1.25 {
                beyond_up += 1; // escaped the strict window (relaxed/unpinned pass)
            } else if chosen < 0.75 {
                beyond_dn += 1;
            } else if chosen >= 1.22 * 0.995 {
                at_ceiling += 1;
            } else if chosen <= 0.78 * 1.005 {
                at_floor += 1;
            } else {
                interior += 1;
            }

            // Independent back-extrapolation, clean openings only.
            let cue = &shot.observed[0];
            let Some(&(_, p0)) = cue.first() else { continue };
            // Rest reference: last sample still within jitter of the start.
            let mut t_rest = cue.first().map(|s| s.0).unwrap_or(0.0);
            for &(t, pp) in cue {
                if (pp - p0).length() < 0.015 {
                    t_rest = t;
                } else {
                    break;
                }
            }
            // Free flight ends at the first cue cushion or first object move.
            use billiards_solver::fit::ObservedEvents;
            let obs = ObservedEvents::from_tracks(&shot.observed, bounds);
            let t_end = obs.cushions[0]
                .iter()
                .map(|&(t, _)| t)
                .fold(t_stop, f64::min);
            if !(t_end - t_rest > 0.55) {
                continue; // no clean early window before the first event
            }
            let win = |lo: f64, hi: f64| -> Option<(DVec3, f64, f64)> {
                let pts: Vec<(f64, DVec3)> = cue
                    .iter()
                    .filter(|(t, _)| *t >= lo && *t <= hi.min(t_end - 0.04))
                    .cloned()
                    .collect();
                window_velocity(&pts)
            };
            let Some((v1, tm1, x1)) = win(t_rest + 0.18, t_rest + 0.48) else { continue };
            if x1 > 0.02 || v1.length() < 0.3 {
                continue;
            }
            let r_lb = (v1.length() + MU_R * G * (tm1 - t_rest)) / v_est;
            // Point estimate when a second window fits: measured deceleration
            // between the windows, clamped to the physical range [roll, slide].
            let r_pt = win(t_rest + 0.48, t_rest + 0.78)
                .filter(|&(v2, _, x2)| x2 <= 0.02 && v2.length() > 0.15)
                .map(|(v2, tm2, _)| {
                    let a = ((v1.length() - v2.length()) / (tm2 - tm1))
                        .clamp(MU_R * G, 1.2 * mu_s * G);
                    (v1.length() + a * (tm1 - t_rest)) / v_est
                });
            audit.push((v_est, r_lb, r_pt));
        }
    }

    println!(
        "fit-chosen speed vs its ±22% pin window ({n_fit} shots):\n  \
         at ceiling {at_ceiling} ({:.0}%) · interior {interior} ({:.0}%) · at floor {at_floor} ({:.0}%) · escaped up {beyond_up} · escaped down {beyond_dn}\n",
        100.0 * at_ceiling as f64 / n_fit.max(1) as f64,
        100.0 * interior as f64 / n_fit.max(1) as f64,
        100.0 * at_floor as f64 / n_fit.max(1) as f64,
    );

    println!("independent back-extrapolation ({} clean-opening shots):", audit.len());
    let lbs: Vec<f64> = audit.iter().map(|a| a.1).collect();
    let pts: Vec<f64> = audit.iter().filter_map(|a| a.2).collect();
    let above = lbs.iter().filter(|r| **r > 1.0).count();
    println!(
        "  lower-bound ratio (true ≥ this / launch_estimate): median {:.3} · >1 in {}/{} shots",
        median(lbs.clone()),
        above,
        lbs.len()
    );
    if !pts.is_empty() {
        println!(
            "  point-estimate ratio (slide-aware): median {:.3}  (n {})",
            median(pts.clone()),
            pts.len()
        );
    }
    for (lo, hi) in [(0.0, 1.2), (1.2, 2.0), (2.0, 3.5), (3.5, 9.0)] {
        let sel: Vec<f64> = audit
            .iter()
            .filter(|a| a.0 >= lo && a.0 < hi)
            .map(|a| a.1)
            .collect();
        if sel.len() < 8 {
            continue;
        }
        let selp: Vec<f64> =
            audit.iter().filter(|a| a.0 >= lo && a.0 < hi).filter_map(|a| a.2).collect();
        println!(
            "  v_est [{lo:.1},{hi:.1}) m/s: lb median {:.3} (n {})  pt median {}",
            median(sel.clone()),
            sel.len(),
            if selp.is_empty() { "—".into() } else { format!("{:.3} (n {})", median(selp.clone()), selp.len()) },
        );
    }
}
