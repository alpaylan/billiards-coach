//! Free-roll deceleration, measured straight off the tracks — the third leg of
//! the event-local energy audit (cushions: cushion_events; collisions:
//! collision_events). Those two showed the CONTACT models are, if anything,
//! less energetic than the bilevel calibration uses, while whole-shot ruling
//! quality wants MORE energy — so the leak must be in the free motion between
//! events. This measures it: on every long, event-free, straight stretch of a
//! track, compare the ball's speed in the first and last 0.35 s. The slope is
//! the cloth's real rolling deceleration `μ_r·g`, no model in the loop.
//!
//!   cargo run -p billiards-solver --example roll_decel --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_solver::fit::{FirstMove, ObservedEvents};
use billiards_solver::measure::window_velocity;
use billiards_solver::shotfile;

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

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let bounds = table.center_bounds(ball.radius);
    const G: f64 = 9.81;

    for dir in &dirs {
        let mut decels: Vec<(f64, f64)> = Vec::new(); // (mean speed, decel m/s²)
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
                // Event times that break a stretch: this ball's bounces, any
                // strike, plus 0.4 s after each for slide-out.
                let mut breaks: Vec<f64> = obs.cushions[ti].iter().map(|&(t, _)| t).collect();
                breaks.extend(&strikes);
                let Some(&(t0, p0)) = track.first() else { continue };
                let move_t = track
                    .iter()
                    .find(|(_, p)| (*p - p0).length() > 0.02)
                    .map(|&(t, _)| t)
                    .unwrap_or(t0);
                breaks.push(move_t);
                let end = track.last().map(|s| s.0).unwrap_or(t0);
                // Candidate stretch starts: 0.45 s after each break, ending at
                // the next break. Take stretches ≥ 1.1 s.
                let pts_of = |lo: f64, hi: f64| -> Vec<(f64, DVec3)> {
                    track.iter().filter(|(t, _)| *t >= lo && *t <= hi).cloned().collect()
                };
                let mut starts: Vec<f64> = breaks.iter().map(|t| t + 0.45).collect();
                starts.push(t0);
                for s in starts {
                    let stop = breaks
                        .iter()
                        .map(|&b| b - 0.05)
                        .filter(|&b| b > s + 1.1)
                        .fold(end, f64::min);
                    if stop - s < 1.1 || stop > end {
                        continue;
                    }
                    let a = pts_of(s, s + 0.35);
                    let b = pts_of(stop - 0.35, stop);
                    let (Some((v1, tm1, x1)), Some((v2, tm2, x2))) =
                        (window_velocity(&a), window_velocity(&b))
                    else {
                        continue;
                    };
                    // Straight, genuinely rolling, and still moving at the end.
                    if x1 > 0.015 || x2 > 0.015 || v1.length() < 0.35 || v2.length() < 0.12 {
                        continue;
                    }
                    let dot = v1.dot(v2) / (v1.length() * v2.length());
                    if dot < 0.995 {
                        continue; // bent — hidden contact or curve
                    }
                    let decel = (v1.length() - v2.length()) / (tm2 - tm1);
                    decels.push((0.5 * (v1.length() + v2.length()), decel));
                }
            }
        }
        decels.sort_by(|a, b| a.1.total_cmp(&b.1));
        if decels.is_empty() {
            println!("{dir}: no stretches");
            continue;
        }
        let med = decels[decels.len() / 2].1;
        let mu_cal = load_field(dir, "mu_roll").unwrap_or(PhysicsParams::carom_calibrated().mu_roll);
        // Interquartile spread for scale.
        let (q1, q3) = (decels[decels.len() / 4].1, decels[3 * decels.len() / 4].1);
        println!(
            "{}: {} stretches · measured roll decel median {:.3} m/s² [IQR {:.3}..{:.3}] → μ_r {:.4} · calibration μ_r {:.4} ({:.3} m/s²)",
            dir.trim_end_matches('/').rsplit('/').next().unwrap(),
            decels.len(),
            med,
            q1,
            q3,
            med / G,
            mu_cal,
            mu_cal * G,
        );
    }
}
