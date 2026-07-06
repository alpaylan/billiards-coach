//! Why does the reconstructed cue go the wrong way after the first ball-ball
//! collision? Compare, per shot: the OBSERVED cue deflection (heading change
//! across the first object contact, read from the tracks) vs the SIMULATED
//! deflection under the fitted action. A systematic gap points at the collision
//! model (spin transfer / throw / restitution); scattered per-shot error points
//! at spin recovery (follow/draw is only weakly observable before a cushion).
//!
//!   cargo run -p billiards-solver --example collision_deflection --release -- data/<match>

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::shotfile;

/// Heading (deg) of a track over [t0, t1], from the chord.
fn heading(track: &[(f64, DVec3)], t0: f64, t1: f64) -> Option<f64> {
    let pts: Vec<DVec3> = track.iter().filter(|(t, _)| *t >= t0 && *t <= t1).map(|(_, p)| *p).collect();
    let (a, b) = (pts.first()?, pts.last()?);
    let d = *b - *a;
    if d.length() < 0.03 {
        return None;
    }
    Some(d.y.atan2(d.x).to_degrees())
}

fn wrap(a: f64) -> f64 {
    (a + 540.0).rem_euclid(360.0) - 180.0
}

fn main() {
    let dir = env::args().nth(1).expect("usage: collision_deflection <match_dir>");
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };

    let mut paths: Vec<String> = fs::read_dir(&dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path().to_string_lossy().into_owned()))
        .filter(|p| p.ends_with(".shot")).collect();
    paths.sort();

    println!("{:<16} {:>8} {:>9} {:>9} {:>9}  (cue heading change across 1st contact)",
        "shot", "t_hit", "observed", "simulated", "sim-obs");
    let phys_file = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let phys = load_phys(&phys_file).unwrap_or_else(PhysicsParams::carom_calibrated);
    let mut gaps = Vec::new();
    for p in &paths {
        let Ok(text) = fs::read_to_string(p) else { continue };
        let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };
        let cue_tr = &shot.observed[0];
        // First object motion = the collision moment (same signal the fit uses).
        let t_hit = shot.observed.iter().skip(1)
            .filter_map(|tr| {
                let &(_, p0) = tr.first()?;
                tr.iter().find(|(_, q)| (*q - p0).length() > 0.02).map(|&(t, _)| t)
            })
            .fold(f64::INFINITY, f64::min);
        if !t_hit.is_finite() || t_hit > 3.0 {
            continue; // no early ball-ball contact to study
        }
        // Observed cue heading just before vs just after the contact.
        let (Some(h_in), Some(h_out)) =
            (heading(cue_tr, (t_hit - 0.45).max(0.0), t_hit - 0.05), heading(cue_tr, t_hit + 0.05, t_hit + 0.45))
        else { continue };
        let obs = wrap(h_out - h_in);

        // Same measurement on the simulated trajectory at the fitted action.
        let fit = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
        let sim = simulate(&shot.scene.ball_states(&fit.action), &table, &ball, &phys);
        let sim_tr: Vec<(f64, DVec3)> = cue_tr.iter()
            .map(|&(t, _)| (t, sim.trajectories[0].state_at(t).pos)).collect();
        let (Some(s_in), Some(s_out)) =
            (heading(&sim_tr, (t_hit - 0.45).max(0.0), t_hit - 0.05), heading(&sim_tr, t_hit + 0.05, t_hit + 0.45))
        else { continue };
        let simd = wrap(s_out - s_in);

        let name = p.rsplit('/').next().unwrap();
        println!("{name:<16} {t_hit:>7.2}s {obs:>8.1}° {simd:>8.1}° {:>8.1}°", wrap(simd - obs));
        gaps.push(wrap(simd - obs));
    }
    if !gaps.is_empty() {
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let mad = gaps.iter().map(|g| (g - mean).abs()).sum::<f64>() / gaps.len() as f64;
        println!("\nmean sim-obs {mean:.1}° · mean|dev| {mad:.1}° over {} early collisions", gaps.len());
        println!("(consistent sign => collision model bias; large scatter => per-shot spin mis-fit)");
    }
}

fn load_phys(path: &str) -> Option<PhysicsParams> {
    let text = fs::read_to_string(path).ok()?;
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars().skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>().parse().ok()
    };
    Some(PhysicsParams {
        cushion_restitution: get("cushion_restitution")?,
        cushion_friction: get("cushion_friction")?,
        mu_slide: get("mu_slide")?,
        mu_roll: get("mu_roll")?,
        ..PhysicsParams::carom_calibrated()
    })
}
