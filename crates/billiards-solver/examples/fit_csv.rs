//! End-to-end: reconstruct a hit from a shot exported by `python/track.py`.
//!
//!   python3 python/track.py --frames real_frames --corners "..." --scale 3 \
//!       --fit-out shot.csv
//!   cargo run -p billiards-solver --example fit_csv --release -- shot.csv
//!
//! The shot file is color-labeled: a `cue COLOR` header then `COLOR,t,x,y` data
//! lines in table coordinates (meters). The cue (a white/yellow player ball) is
//! ordered first.

use std::collections::HashMap;
use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec};
use billiards_solver::fit::{FitConfig, fit_action};

const COLORS: [&str; 3] = ["white", "yellow", "red"];

fn main() {
    let path = env::args().nth(1).expect("usage: fit_csv SHOT.csv");
    let text = fs::read_to_string(&path).expect("read csv");
    let (ball, phys, table) = (BallSpec::carom(), PhysicsParams::default(), TableSpec::carom_match());
    let r = ball.radius;

    // Parse the color-labeled shot, then order the balls cue-first.
    let mut cue_color = "white".to_string();
    let mut by_color: HashMap<String, Vec<(f64, DVec3)>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(c) = line.strip_prefix("cue ") {
            cue_color = c.trim().to_ascii_lowercase();
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 4 {
            continue;
        }
        let (Ok(t), Ok(x), Ok(y)) =
            (f[1].trim().parse(), f[2].trim().parse(), f[3].trim().parse())
        else {
            continue; // the `color,t,x,y` header line
        };
        by_color.entry(f[0].trim().to_string()).or_default().push((t, DVec3::new(x, y, r)));
    }
    let order: Vec<String> = std::iter::once(cue_color.clone())
        .chain(COLORS.iter().map(|s| s.to_string()).filter(|c| *c != cue_color))
        .filter(|c| by_color.contains_key(c))
        .collect();
    let tracks: Vec<Vec<(f64, DVec3)>> = order.iter().map(|c| by_color[c].clone()).collect();
    assert!(tracks.first().is_some_and(|t| !t.is_empty()), "no cue track in csv");

    let cue = tracks[0][0].1;
    let objects: Vec<DVec3> = tracks.iter().skip(1).filter_map(|t| t.first().map(|s| s.1)).collect();
    let scene = Scene::new(cue, objects);

    // Optional 2nd arg overrides the aim window (for diagnosis).
    let aim_window = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.03);
    let cfg = FitConfig { aim_window, ..FitConfig::default() };
    let fit = fit_action(&scene, &tracks, &table, &ball, &phys, &cfg);
    let (h, v) = fit.action.tip_offset(r);

    println!("end-to-end hit reconstruction — {path}");
    println!("  cue at ({:+.2},{:+.2}) m, {} object ball(s), {} cue samples",
        cue.x, cue.y, scene.objects.len(), tracks[0].len());
    println!("  fit trajectory rms: {:.0} mm", fit.rms_m * 1000.0);
    println!("  aim      {:+.1}°", fit.action.aim.to_degrees());
    println!("  force    {:.2} m/s", fit.action.speed);
    println!("  english  {:+.0}% R", h * 100.0);
    println!("  follow   {:+.0}% R", v * 100.0);
    println!("\n  (numbers are under the engine's *uncalibrated* physics — Phase 4 makes them physically true)");
}
