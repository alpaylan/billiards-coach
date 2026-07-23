//! Calibrate one game's physics and store it with the game.
//!
//! Different tables/cloths behave differently, so each game carries its own four
//! fitted parameters instead of a global constant. Reads a match dir's `.shot`
//! files, recovers the cushion/cloth parameters, **validates them held-out** (fit
//! on half the shots, measure the other half — if that generalization error blows
//! up we flag likely overfitting), and writes `<dir>/calibration.json` which the
//! editor loads in place of the built-in defaults.
//!
//!   cargo run -p billiards-solver --example calibrate_match --release -- data/<match>

use std::{env, fs};

use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_solver::calibrate::{CalibConfig, CalibShot, calibrate};
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::shotfile;

/// Load via the shared parser (corrected object rests — same scene the editor uses).
fn load_shot(path: &str, table: &TableSpec, r: f64) -> Option<CalibShot> {
    let text = fs::read_to_string(path).ok()?;
    let s = shotfile::parse(&text, table, r)?;
    Some(CalibShot { scene: s.scene, observed: s.observed })
}

fn rms(shots: &[&CalibShot], table: &TableSpec, ball: &BallSpec, phys: &PhysicsParams, fit: &FitConfig) -> f64 {
    if shots.is_empty() {
        return f64::NAN;
    }
    let sse: f64 = shots.iter().map(|s| {
        let r = fit_action(&s.scene, &s.observed, table, ball, phys, fit);
        r.rms_m * r.rms_m
    }).sum();
    (sse / shots.len() as f64).sqrt() * 1000.0
}

fn main() {
    let dir = env::args().nth(1).expect("usage: calibrate_match <match_dir>");
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    // BB_STEPS switches the ball-ball contact model for the whole calibration
    // (see collision.rs) — used to refit under the integrated contact.
    let mut base = PhysicsParams::default();
    if let Some(v) = env::var("BB_STEPS").ok().and_then(|s| s.parse().ok()) {
        base.ball_contact_steps = v;
    }
    if let Some(v) = env::var("CUSHION_STEPS").ok().and_then(|s| s.parse().ok()) {
        base.cushion_contact_steps = v;
    }
    if let Some(v) = env::var("EC_SLOPE").ok().and_then(|s| s.parse().ok()) {
        base.cushion_restitution_slope = v;
    }
    if let Some(v) = env::var("ECD").ok().and_then(|s| s.parse().ok()) {
        base.cushion_restitution_chain = v;
    }
    let r = ball.radius;

    let mut paths: Vec<String> = fs::read_dir(&dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path().to_string_lossy().into_owned()))
        .filter(|p| p.ends_with(".shot"))
        .collect();
    paths.sort();
    let shots: Vec<CalibShot> = paths.iter().filter_map(|p| load_shot(p, &table, r)).collect();
    if shots.len() < 2 {
        eprintln!("need >=2 shots, found {}", shots.len());
        return;
    }
    let fit = FitConfig { aim_window: 0.03, ..CalibConfig::default().fit };
    let mut cfg = CalibConfig { fit, ..CalibConfig::default() };
    // PIN_EC / PIN_FC: impose an externally measured cushion parameter (e.g.
    // the event-local fit from cushion_events) and let the rest re-optimize.
    if let Some(v) = env::var("PIN_EC").ok().and_then(|s| s.parse().ok()) {
        cfg.pins[0] = Some(v);
    }
    if let Some(v) = env::var("PIN_FC").ok().and_then(|s| s.parse().ok()) {
        cfg.pins[1] = Some(v);
    }
    let all: Vec<&CalibShot> = shots.iter().collect();

    let cal = calibrate(&shots, &table, &ball, &base, &cfg);
    let base_rms = rms(&all, &table, &ball, &base, &fit);
    let full_rms = rms(&all, &table, &ball, &cal, &fit);

    // Held-out generalization: fit on even-indexed shots, measure the odd ones (and
    // vice versa). If the held-out error is far above the in-fit error, the four
    // params are chasing noise in a specific game rather than the table's physics.
    // CALIB_FAST=1 skips it (3× cheaper — for parameter experiments, not for
    // producing production calibrations).
    let mut heldout = f64::NAN;
    if shots.len() >= 6 && env::var("CALIB_FAST").map_or(true, |v| v != "1") {
        let mut acc = 0.0;
        for parity in 0..2 {
            let train: Vec<CalibShot> = shots.iter().enumerate()
                .filter(|(i, _)| i % 2 == parity).map(|(_, s)| clone_shot(s)).collect();
            let test: Vec<&CalibShot> = shots.iter().enumerate()
                .filter(|(i, _)| i % 2 != parity).map(|(_, s)| s).collect();
            let c = calibrate(&train, &table, &ball, &base, &cfg);
            acc += rms(&test, &table, &ball, &c, &fit);
        }
        heldout = acc / 2.0;
    }

    let overfit = heldout.is_finite() && heldout > 1.35 * full_rms;
    let json = format!(
        "{{\n  \"cushion_restitution\": {:.4},\n  \"cushion_friction\": {:.4},\n  \
         \"mu_slide\": {:.4},\n  \"mu_roll\": {:.5},\n  \"n_shots\": {},\n  \
         \"baseline_rms_mm\": {:.0},\n  \"calibrated_rms_mm\": {:.0},\n  \
         \"heldout_rms_mm\": {},\n  \"overfit_flag\": {}\n}}\n",
        cal.cushion_restitution, cal.cushion_friction, cal.mu_slide, cal.mu_roll,
        shots.len(), base_rms, full_rms,
        if heldout.is_finite() { format!("{heldout:.0}") } else { "null".into() },
        overfit,
    );
    // CALIB_OUT: write elsewhere (parameter experiments must not clobber the
    // game's production calibration.json).
    let out = env::var("CALIB_OUT")
        .unwrap_or_else(|_| format!("{}/calibration.json", dir.trim_end_matches('/')));
    fs::write(&out, &json).expect("write calibration.json");
    println!("calibrated {} shots: e_c {:.3} f_c {:.3} mu_s {:.3} mu_r {:.4}",
        shots.len(), cal.cushion_restitution, cal.cushion_friction, cal.mu_slide, cal.mu_roll);
    println!("  in-fit RMS {full_rms:.0} mm (baseline {base_rms:.0}); held-out {}",
        if heldout.is_finite() { format!("{heldout:.0} mm") } else { "n/a (<6 shots)".into() });
    if overfit {
        println!("  ⚠ held-out error >> in-fit — treat params with caution (few/noisy shots)");
    }
    println!("  -> {out}");
}

fn clone_shot(s: &CalibShot) -> CalibShot {
    CalibShot { scene: s.scene.clone(), observed: s.observed.clone() }
}
