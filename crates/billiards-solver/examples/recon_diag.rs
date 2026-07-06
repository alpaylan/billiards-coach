//! Diagnose WHERE a reconstruction goes wrong: per shot, split the fit residual
//! into the cue ball's own path error vs the object balls' error, and report when
//! the cue first diverges. A small cue error near a collision sends an object ball
//! the wrong way, so a large object error with a small cue error = the shot is a
//! collision-sensitivity problem, not a cue-physics problem.
//!
//!   cargo run -p billiards-solver --example recon_diag --release -- data/masa4_1080_match/*.shot

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::shotfile;

/// RMS + endpoint error (mm) of one ball's simulated vs observed track.
fn ball_err(sim_traj: &billiards_core::Trajectory, obs: &[(f64, DVec3)]) -> (f64, f64) {
    let mut sse = 0.0;
    for (t, p) in obs {
        sse += (sim_traj.state_at(*t).pos - *p).length_squared();
    }
    let rms = (sse / obs.len().max(1) as f64).sqrt() * 1000.0;
    let (te, pe) = *obs.last().unwrap();
    let end = (sim_traj.state_at(te).pos - pe).length() * 1000.0;
    (rms, end)
}

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let phys = PhysicsParams::carom_calibrated();
    let r = ball.radius;
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };

    println!("{:<16} {:>4} {:>6} {:>8} {:>9} {:>10} {:>8}",
        "shot", "dur", "cueRMS", "cueEnd", "objRMS", "objEnd", "cueDiv@");
    for p in &paths {
        let Ok(text) = fs::read_to_string(p) else { continue };
        let Some(shot) = shotfile::parse(&text, &table, r) else { continue };
        let (scene, tracks) = (shot.scene, shot.observed);
        let fit = fit_action(&scene, &tracks, &table, &ball, &phys, &cfg);
        let sim = simulate(&scene.ball_states(&fit.action), &table, &ball, &phys);

        let (cue_rms, cue_end) = ball_err(&sim.trajectories[0], &tracks[0]);
        // object balls (mean rms / max endpoint)
        let mut obj_sse = 0.0;
        let mut obj_n = 0usize;
        let mut obj_end = 0.0_f64;
        for bi in 1..tracks.len() {
            let (rms, end) = ball_err(&sim.trajectories[bi], &tracks[bi]);
            obj_sse += rms * rms * tracks[bi].len() as f64;
            obj_n += tracks[bi].len();
            obj_end = obj_end.max(end);
        }
        let obj_rms = (obj_sse / obj_n.max(1) as f64).sqrt();
        // cue divergence onset: first observed time the cue sim is >150 mm off
        let div = tracks[0].iter()
            .find(|(t, p)| (sim.trajectories[0].state_at(*t).pos - *p).length() > 0.15)
            .map(|(t, _)| format!("{t:.1}s"))
            .unwrap_or_else(|| "never".into());
        let dur = tracks[0].last().unwrap().0;
        let name = p.rsplit('/').next().unwrap();
        let aim = fit.action.aim.to_degrees();
        let spd = fit.action.speed;
        println!("{name:<16} {dur:>4.1} {cue_rms:>6.0} {cue_end:>8.0} {obj_rms:>9.0} {obj_end:>10.0} {div:>8}  aim{aim:>6.1} v{spd:>5.2}");
    }
}
