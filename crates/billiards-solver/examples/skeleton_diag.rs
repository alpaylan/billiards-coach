//! Diagnose the event skeleton on real shots: what bounces the tracks claim,
//! what the fitted reconstruction actually does, and where they disagree.
//!
//!   cargo run -p billiards-solver --example skeleton_diag --release -- data/masa4_match/shot_07.shot …

use std::{env, fs};

use billiards_core::{BallSpec, ContactKind, PhysicsParams, Simulation, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, ObservedEvents, event_penalty, fit_action};
use billiards_solver::shotfile;

fn load_calibration(dir: &str) -> PhysicsParams {
    let path = format!("{dir}/calibration.json");
    let Ok(text) = fs::read_to_string(path) else { return PhysicsParams::carom_calibrated() };
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars().skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>().parse().ok()
    };
    match (get("cushion_restitution"), get("cushion_friction"), get("mu_slide"), get("mu_roll")) {
        (Some(e), Some(f), Some(s), Some(r)) => PhysicsParams {
            cushion_restitution: e, cushion_friction: f, mu_slide: s, mu_roll: r,
            ..PhysicsParams::carom_calibrated()
        },
        _ => PhysicsParams::carom_calibrated(),
    }
}

fn sim_bounces(sim: &Simulation, n: usize, b: [f64; 4]) -> Vec<Vec<(f64, u8)>> {
    let mut per: Vec<Vec<(f64, u8)>> = vec![Vec::new(); n];
    for e in &sim.events {
        let ContactKind::Cushion { ball } = e.kind else { continue };
        let bi = ball.0 as usize;
        if bi >= n { continue; }
        let p = sim.trajectories[bi].state_at(e.time).pos;
        let d = [p.x - b[0], b[1] - p.x, p.y - b[2], b[3] - p.y];
        let rail = (0..4u8).min_by(|&x, &y| d[x as usize].total_cmp(&d[y as usize])).unwrap();
        per[bi].push((e.time, rail));
    }
    per
}

fn fmt(b: &[(f64, u8)]) -> String {
    const R: [&str; 4] = ["-x", "+x", "-y", "+y"];
    b.iter().map(|&(t, r)| format!("{}@{t:.2}", R[r as usize])).collect::<Vec<_>>().join(" ")
}

fn main() {
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let bounds = table.center_bounds(ball.radius);
    for p in env::args().skip(1) {
        let dir = p.rsplit_once('/').map_or(".", |(d, _)| d);
        let phys = load_calibration(dir);
        let Ok(text) = fs::read_to_string(&p) else { continue };
        let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };
        let ev = ObservedEvents::from_tracks(&shot.observed, bounds);
        let new = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
        let old = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys,
            &FitConfig { skeleton: false, ..cfg });
        println!("{p}  (ball order: {:?})", shot.order);
        for (k, fm) in ev.first_move.iter().enumerate() {
            println!("  obs first-move ball {}: {fm:?}", k + 1);
        }
        for (label, f) in [("new", &new), ("old", &old)] {
            let sim = simulate(&shot.scene.ball_states(&f.action), &table, &ball, &phys);
            let sb = sim_bounces(&sim, shot.observed.len(), bounds);
            let (h, v) = f.action.tip_offset(ball.radius);
            println!("  {label}: rms {:.0} mm · penalty {:.2} · aim {:.1}° speed {:.2} english {:+.0}%R follow {:+.0}%R",
                f.rms_m * 1000.0, event_penalty(&sim, &ev),
                f.action.aim.to_degrees().rem_euclid(360.0), f.action.speed, h * 100.0, v * 100.0);
            for k in 0..shot.observed.len() {
                let span = shot.observed[k].last().map_or(0.0, |s| s.0);
                println!("    ball {k} ({:.1}s, {} pts): obs [{}] · sim [{}]",
                    span, shot.observed[k].len(), fmt(&ev.cushions[k]), fmt(&sb[k]));
            }
            let story: Vec<String> = sim.events.iter().map(|e| match e.kind {
                ContactKind::Cushion { ball } => format!("b{}·rail@{:.2}", ball.0, e.time),
                ContactKind::BallBall { a, b } => format!("b{}⇄b{}@{:.2}", a.0, b.0, e.time),
            }).collect();
            println!("    sim story: {}", story.join("  "));
        }
    }
}
