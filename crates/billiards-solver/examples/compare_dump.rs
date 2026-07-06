//! Dump the reconstructed trajectory for the side-by-side viewer: fit the cue
//! action to a shot's tracks, simulate it, and write the *simulated* ball
//! positions at the observed timestamps. Python pairs this with the tracked
//! ("actual") positions to render actual-vs-reconstructed.
//!
//!   cargo run -p billiards-solver --example compare_dump --release -- SHOT.csv RECON.csv

use std::collections::HashMap;
use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};

const COLORS: [&str; 3] = ["white", "yellow", "red"];

/// Parse a color-labeled shot file (`cue COLOR` header + `COLOR,t,x,y` lines).
/// Returns the ball colors in cue-first order and each color's trajectory.
fn parse_shot(text: &str, r: f64) -> (Vec<String>, HashMap<String, Vec<(f64, DVec3)>>) {
    let mut cue = "white".to_string();
    let mut by_color: HashMap<String, Vec<(f64, DVec3)>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(c) = line.strip_prefix("cue ") {
            cue = c.trim().to_ascii_lowercase();
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 4 {
            continue; // header or blank
        }
        let (Ok(t), Ok(x), Ok(y)) =
            (f[1].trim().parse(), f[2].trim().parse(), f[3].trim().parse())
        else {
            continue; // the `color,t,x,y` header line
        };
        by_color.entry(f[0].trim().to_string()).or_default().push((t, DVec3::new(x, y, r)));
    }
    let order: Vec<String> = std::iter::once(cue.clone())
        .chain(COLORS.iter().map(|s| s.to_string()).filter(|c| *c != cue))
        .filter(|c| by_color.contains_key(c))
        .collect();
    (order, by_color)
}

fn main() {
    let inp = env::args().nth(1).expect("usage: compare_dump SHOT.csv RECON.csv");
    let out = env::args().nth(2).expect("need output path");
    let text = fs::read_to_string(&inp).expect("read csv");
    let (ball, phys, table) = (BallSpec::carom(), PhysicsParams::default(), TableSpec::carom_match());
    let r = ball.radius;

    // Color-labeled format: a `cue COLOR` header then `COLOR,t,x,y` data lines.
    // Order the balls cue-first (matching the fitter's convention) but keep each
    // ball's color so the reconstruction can be written back color-labeled.
    let (order, by_color) = parse_shot(&text, r);
    let tracks: Vec<Vec<(f64, DVec3)>> = order.iter().map(|c| by_color[c].clone()).collect();

    let cue = tracks[0][0].1;
    let objects: Vec<DVec3> = tracks.iter().skip(1).filter_map(|t| t.first().map(|s| s.1)).collect();
    let scene = Scene::new(cue, objects);
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let fit = fit_action(&scene, &tracks, &table, &ball, &phys, &cfg);
    let sim = simulate(&scene.ball_states(&fit.action), &table, &ball, &phys);

    let mut s = String::from("color,t,x,y\n");
    for (bi, color) in order.iter().enumerate() {
        for (t, _) in &tracks[bi] {
            let p = sim.trajectories[bi].state_at(*t).pos;
            s.push_str(&format!("{color},{t:.4},{:.4},{:.4}\n", p.x, p.y));
        }
    }
    fs::write(&out, s).expect("write recon");
    eprintln!(
        "fit rms {:.0} mm; aim {:.1}°, force {:.2} m/s -> {out}",
        fit.rms_m * 1000.0,
        fit.action.aim.to_degrees(),
        fit.action.speed
    );
}
