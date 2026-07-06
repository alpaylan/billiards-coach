//! Reconstruction snapshot testing: fit every `.shot` in the given game dirs
//! and compare one summary line per shot against the APPROVED snapshot, so any
//! change to the fit/engine/extraction is judged against the whole corpus.
//!
//!   cargo run -p billiards-solver --example snapshot_recon --release -- check data/masa4_day/game_* data/masa4/game_00
//!   cargo run -p billiards-solver --example snapshot_recon --release -- bless data/masa4_day/game_03   # approve current
//!
//! Snapshots live in `snapshots/recon/<dir-with-underscores>.tsv`, one line per
//! shot: action (aim/speed/english/follow), cue & object RMS, event penalty,
//! and the compact simulated event story. The fit is deterministic, so `check`
//! diffs exactly; the report classifies every change by its cue-RMS delta so a
//! human can decide whether to bless.

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, ContactKind, PhysicsParams, Simulation, TableSpec, Trajectory};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, ObservedEvents, event_penalty, fit_action};
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

fn ball_rms(traj: &Trajectory, obs: &[(f64, DVec3)]) -> f64 {
    let sse: f64 = obs.iter().map(|(t, p)| (traj.state_at(*t).pos - *p).length_squared()).sum();
    (sse / obs.len().max(1) as f64).sqrt() * 1000.0
}

/// Compact event story: `x12@0.44` = balls 1⇄2 collide, `r0+y@0.15` = ball 0
/// bounces off the +y rail. Times to centiseconds (the sim is deterministic).
fn story(sim: &Simulation, bounds: [f64; 4]) -> String {
    let mut parts = Vec::new();
    for e in &sim.events {
        match e.kind {
            ContactKind::BallBall { a, b } => parts.push(format!("x{}{}@{:.2}", a.0, b.0, e.time)),
            ContactKind::Cushion { ball } => {
                let p = sim.trajectories[ball.0 as usize].state_at(e.time).pos;
                let d = [p.x - bounds[0], bounds[1] - p.x, p.y - bounds[2], bounds[3] - p.y];
                let rail = ["-x", "+x", "-y", "+y"][(0..4)
                    .min_by(|&i, &j| d[i].total_cmp(&d[j]))
                    .unwrap()];
                parts.push(format!("r{}{}@{:.2}", ball.0, rail, e.time));
            }
        }
    }
    parts.join(" ")
}

fn snapshot_dir(dir: &str) -> String {
    let mut paths: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|p| p.ends_with(".shot"))
        .collect();
    paths.sort();

    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let phys = load_calibration(dir);
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let bounds = table.center_bounds(ball.radius);

    let mut out = String::new();
    for p in &paths {
        let name = p.rsplit('/').next().unwrap();
        let Ok(text) = fs::read_to_string(p) else { continue };
        let Some(shot) = shotfile::parse(&text, &table, ball.radius) else {
            out += &format!("{name}\tUNPARSEABLE\n");
            continue;
        };
        let f = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
        let sim = simulate(&shot.scene.ball_states(&f.action), &table, &ball, &phys);
        let ev = ObservedEvents::from_tracks(&shot.observed, bounds);
        let cue = ball_rms(&sim.trajectories[0], &shot.observed[0]);
        let obj = (1..shot.observed.len())
            .map(|b| ball_rms(&sim.trajectories[b], &shot.observed[b]))
            .fold(0.0f64, f64::max);
        let (h, v) = f.action.tip_offset(ball.radius);
        out += &format!(
            "{name}\tcue={}\taim={:.1}\tspeed={:.2}\teng={:+.0}\tfol={:+.0}\tcue_rms={:.0}\tobj_rms={:.0}\tpen={:.2}\t{}\n",
            shot.order[0],
            f.action.aim.to_degrees().rem_euclid(360.0),
            f.action.speed,
            h * 100.0,
            v * 100.0,
            cue,
            obj,
            event_penalty(&sim, &ev),
            story(&sim, bounds),
        );
    }
    out
}

fn snap_path(dir: &str) -> String {
    format!("snapshots/recon/{}.tsv", dir.trim_end_matches('/').replace(['/', '\\'], "_"))
}

fn cue_rms_of(line: &str) -> Option<f64> {
    line.split('\t').find_map(|f| f.strip_prefix("cue_rms="))?.parse().ok()
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let (Some(mode), dirs) = (args.first().cloned(), &args[1.min(args.len())..]) else {
        eprintln!("usage: snapshot_recon check|bless GAME_DIR…");
        std::process::exit(2);
    };
    if dirs.is_empty() {
        eprintln!("usage: snapshot_recon check|bless GAME_DIR…");
        std::process::exit(2);
    }

    let mut regressions = 0usize;
    for dir in dirs {
        let current = snapshot_dir(dir);
        let path = snap_path(dir);
        match mode.as_str() {
            "bless" => {
                fs::create_dir_all("snapshots/recon").expect("mkdir");
                fs::write(&path, &current).expect("write snapshot");
                println!("blessed {} ({} shots)", path, current.lines().count());
            }
            _ => {
                let Ok(approved) = fs::read_to_string(&path) else {
                    println!("{dir}: NO APPROVED SNAPSHOT ({path}) — run bless first");
                    regressions += 1;
                    continue;
                };
                let a: Vec<&str> = approved.lines().collect();
                let c: Vec<&str> = current.lines().collect();
                let key = |l: &str| l.split('\t').next().unwrap_or("").to_string();
                let amap: std::collections::HashMap<String, &str> =
                    a.iter().map(|l| (key(l), *l)).collect();
                let mut same = 0;
                let mut changed: Vec<(String, Option<f64>, Option<f64>)> = Vec::new();
                for l in &c {
                    let k = key(l);
                    match amap.get(&k) {
                        Some(old) if *old == *l => same += 1,
                        Some(old) => changed.push((k, cue_rms_of(old), cue_rms_of(l))),
                        None => changed.push((k, None, cue_rms_of(l))),
                    }
                }
                let missing = a.len().saturating_sub(c.len().min(a.len()));
                if changed.is_empty() && a.len() == c.len() {
                    println!("{dir}: OK ({same} shots unchanged)");
                } else {
                    regressions += 1;
                    println!("{dir}: {} CHANGED / {same} same / {missing} missing", changed.len());
                    let (mut better, mut worse) = (0, 0);
                    for (k, old, new) in &changed {
                        let delta = match (old, new) {
                            (Some(o), Some(n)) => {
                                if n < o { better += 1 } else { worse += 1 }
                                format!("cue_rms {o:.0} -> {n:.0} mm")
                            }
                            _ => "new/removed".into(),
                        };
                        println!("    {k}: {delta}");
                    }
                    println!("    ({better} improved, {worse} worse — bless to approve)");
                }
            }
        }
    }
    if mode != "bless" && regressions > 0 {
        std::process::exit(1);
    }
}
