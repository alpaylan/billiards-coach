//! Ball-ball collision check: for each cue→object collision the fit produces,
//! compare the ENGINE's throw (how far off the line of centers it sends the
//! object) and speed transfer against the OBSERVED object direction from the
//! tracks. The object leaves from rest, so its outgoing *direction* is the clean,
//! measurable quantity — this isolates the collision/throw model from cushions.
//!
//!   cargo run -p billiards-solver --example collision_check --release -- data/masa4_match/shot_05.shot

use std::collections::HashMap;
use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallId, BallSpec, ContactKind, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};

const COLORS: [&str; 3] = ["white", "yellow", "red"];

fn load_shot(path: &str, r: f64) -> Option<(Vec<Vec<(f64, DVec3)>>, Vec<String>, Scene)> {
    let text = fs::read_to_string(path).ok()?;
    let mut cue = "white".to_string();
    let mut by: HashMap<String, Vec<(f64, DVec3)>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(c) = line.strip_prefix("cue ") {
            cue = c.trim().to_ascii_lowercase();
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 4 {
            continue;
        }
        let (Ok(t), Ok(x), Ok(y)) = (f[1].trim().parse(), f[2].trim().parse(), f[3].trim().parse()) else {
            continue;
        };
        by.entry(f[0].trim().to_string()).or_default().push((t, DVec3::new(x, y, r)));
    }
    let order: Vec<String> = std::iter::once(cue.clone())
        .chain(COLORS.iter().map(|s| s.to_string()).filter(|c| *c != cue))
        .filter(|c| by.contains_key(c))
        .collect();
    let tracks: Vec<Vec<(f64, DVec3)>> = order.iter().map(|c| by[c].clone()).collect();
    if tracks.first().map_or(true, |t| t.len() < 2) {
        return None;
    }
    let cue_pos = tracks[0][0].1;
    let objects = tracks.iter().skip(1).filter_map(|t| t.first().map(|s| s.1)).collect();
    Some((tracks, order, Scene::new(cue_pos, objects)))
}

/// Average velocity from observed samples in [t0, t1] (consecutive frames only).
fn obs_vel(track: &[(f64, DVec3)], t0: f64, t1: f64) -> Option<DVec3> {
    let mut sum = DVec3::new(0.0, 0.0, 0.0);
    let mut n = 0;
    for w in track.windows(2) {
        let (ta, pa) = w[0];
        let (tb, pb) = w[1];
        if ta >= t0 - 1e-9 && tb <= t1 + 1e-9 && tb - ta < 0.05 {
            sum = sum + (pb - pa) / (tb - ta);
            n += 1;
        }
    }
    (n >= 2).then(|| sum / n as f64)
}

fn deg_between(a: DVec3, b: DVec3) -> Option<f64> {
    let (la, lb) = (a.length(), b.length());
    if la < 1e-6 || lb < 1e-6 {
        return None;
    }
    Some((a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees())
}

fn main() {
    let (ball, phys, table) = (BallSpec::carom(), PhysicsParams::carom_calibrated(), TableSpec::carom_match());
    let r = ball.radius;
    println!("engine ball_restitution {:.2}, ball_friction {:.2}\n", phys.ball_restitution, phys.ball_friction);
    for path in env::args().skip(1) {
        let Some((tracks, order, scene)) = load_shot(&path, r) else { continue };
        let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
        let fit = fit_action(&scene, &tracks, &table, &ball, &phys, &cfg);
        let sim = simulate(&scene.ball_states(&fit.action), &table, &ball, &phys);
        println!("{path}");
        for e in &sim.events {
            let ContactKind::BallBall { a, b } = e.kind else { continue };
            let other = if a == BallId(0) { b } else if b == BallId(0) { a } else { continue };
            let oi = other.0 as usize;
            let tc = e.time;
            let cue_in = sim.trajectories[0].state_at(tc - 0.005).vel;
            if cue_in.length() < 0.5 {
                continue;
            }
            let n = (sim.trajectories[oi].state_at(tc).pos - sim.trajectories[0].state_at(tc).pos).normalize();
            let eng_out = sim.trajectories[oi].state_at(tc + 0.01).vel;
            let (Some(eng_throw), Some(cut)) = (deg_between(eng_out, n), deg_between(cue_in, n)) else { continue };
            println!("  {} -> {}  @t={tc:.2}s  cut {cut:.0}°", order[0], order[oi]);
            match obs_vel(&tracks[oi], tc, tc + 0.25) {
                Some(o) => {
                    let ot = deg_between(o, n).unwrap_or(f64::NAN);
                    println!("      throw: engine {eng_throw:.0}°  observed {ot:.0}°  (Δ {:+.0}°)", eng_throw - ot);
                    println!("      speed: cue-in {:.2}  engine-obj {:.2}  observed-obj {:.2} m/s",
                        cue_in.length(), eng_out.length(), o.length());
                }
                None => {
                    println!("      throw: engine {eng_throw:.0}°  observed n/a (gap at contact)");
                    println!("      speed: cue-in {:.2}  engine-obj {:.2}", cue_in.length(), eng_out.length());
                }
            }
        }
    }
}
