//! Where does the reconstruction error come from? Fit a shot, simulate the
//! recovered action, then report the cue-ball error (simulated vs observed) at
//! t=0, before the first contact, and *at each contact* — so we can see whether
//! the divergence is present from the start (initial spin), grows at the cushions
//! (cushion model), or only blows up in the sensitive tail.
//!
//!   cargo run -p billiards-solver --example error_profile --release -- data/masa4_match/shot_07.shot

use std::collections::HashMap;
use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallId, BallSpec, ContactKind, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};

const COLORS: [&str; 3] = ["white", "yellow", "red"];

fn load_shot(path: &str, r: f64) -> Option<(Vec<Vec<(f64, DVec3)>>, Scene)> {
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
    Some((tracks, Scene::new(cue_pos, objects)))
}

fn interp(track: &[(f64, DVec3)], t: f64) -> DVec3 {
    if t <= track[0].0 {
        return track[0].1;
    }
    for w in track.windows(2) {
        if t <= w[1].0 {
            let a = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
            return w[0].1 + (w[1].1 - w[0].1) * a;
        }
    }
    track.last().unwrap().1
}

fn main() {
    let (ball, phys, table) = (BallSpec::carom(), PhysicsParams::carom_calibrated(), TableSpec::carom_match());
    let r = ball.radius;
    for path in env::args().skip(1) {
        let Some((tracks, scene)) = load_shot(&path, r) else {
            eprintln!("skip {path}");
            continue;
        };
        let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
        let fit = fit_action(&scene, &tracks, &table, &ball, &phys, &cfg);
        let sim = simulate(&scene.ball_states(&fit.action), &table, &ball, &phys);

        // cue error (mm) at time t: observed vs simulated cue position
        let cue_err = |t: f64| (interp(&tracks[0], t) - sim.trajectories[0].state_at(t).pos).length() * 1000.0;
        let t_end = tracks[0].last().unwrap().0;

        // cue contacts, in time order
        let mut contacts: Vec<(f64, &str)> = sim
            .events
            .iter()
            .filter_map(|e| match e.kind {
                ContactKind::Cushion { ball: b } if b == BallId(0) => Some((e.time, "cushion")),
                ContactKind::BallBall { a, b } if a == BallId(0) || b == BallId(0) => Some((e.time, "ball")),
                _ => None,
            })
            .filter(|(t, _)| *t <= t_end)
            .collect();
        contacts.sort_by(|a, b| a.0.total_cmp(&b.0));

        let n_cush = contacts.iter().filter(|(_, k)| *k == "cushion").count();
        println!("\n{path}");
        println!("  fit rms {:.0} mm · {} cue cushions · aim {:.1}° speed {:.2} m/s",
            fit.rms_m * 1000.0, n_cush, fit.action.aim.to_degrees(), fit.action.speed);
        let first = contacts.first().map(|(t, _)| *t).unwrap_or(t_end);
        let t_pre = (first * 0.9).max(0.0);
        println!("  t=0.00s (start)            {:5.0} mm", cue_err(0.0));
        println!("  t={:.2}s (before 1st hit)  {:5.0} mm   <- pre-contact (initial conditions)", t_pre, cue_err(t_pre));
        for (i, (t, kind)) in contacts.iter().enumerate() {
            println!("  t={:.2}s  contact {:>2} ({:<7}) {:5.0} mm", t, i + 1, kind, cue_err(*t));
        }
        println!("  t={:.2}s (end)              {:5.0} mm", t_end, cue_err(t_end));
    }
}
