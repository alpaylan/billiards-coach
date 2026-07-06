//! Calibrate the table physics against real tracked shots.
//!
//! Feed it a set of color-labeled `.shot` files (from the MASA 4 pipeline). It
//! fits each shot's cue action under the literature-default physics (the
//! baseline), then runs [`calibrate`] to recover the cushion/cloth parameters
//! that best reproduce *all* the observed trajectories at once, and reports the
//! per-shot and overall RMS before vs after.
//!
//!   cargo run -p billiards-solver --example calibrate_shots --release -- data/shots/shot_*.shot

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

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: calibrate_shots SHOT.shot ...");
        return;
    }
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let base = PhysicsParams::default();
    let r = ball.radius;

    let mut names = Vec::new();
    let mut shots = Vec::new();
    for p in &paths {
        match load_shot(p, &table, r) {
            Some(s) => { names.push(p.clone()); shots.push(s); }
            None => eprintln!("skip {p}: no usable cue track"),
        }
    }
    println!("loaded {} shots\n", shots.len());

    // Inner fit config: coarse enough for the nested loop, with the real-shot aim window.
    // Pin the aim to the observed heading so the fit residual is pure physics
    // error (not aim distortion) — that's what calibration should minimize.
    let fit = FitConfig { aim_window: 0.03, ..CalibConfig::default().fit };

    let rms = |phys: &PhysicsParams| {
        let mut sse = 0.0;
        let mut per = Vec::new();
        for s in &shots {
            let f = fit_action(&s.scene, &s.observed, &table, &ball, phys, &fit);
            per.push(f.rms_m * 1000.0);
            sse += f.rms_m * f.rms_m;
        }
        ((sse / shots.len() as f64).sqrt() * 1000.0, per)
    };

    let (base_rms, base_per) = rms(&base);
    println!("BASELINE (literature defaults): overall RMS {base_rms:.0} mm");
    for (n, e) in names.iter().zip(&base_per) {
        println!("  {:<28} {:.0} mm", n.rsplit('/').next().unwrap(), e);
    }

    // What the editor actually renders: reconstruction under the SHIPPED params.
    let (ship_rms, ship_per) = rms(&PhysicsParams::carom_calibrated());
    println!("\nSHIPPED carom_calibrated() — what the editor draws: overall RMS {ship_rms:.0} mm");
    for (n, e) in names.iter().zip(&ship_per) {
        println!("  {:<28} {:.0} mm", n.rsplit('/').next().unwrap(), e);
    }

    println!("\ncalibrating (nested optimization)…");
    let cfg = CalibConfig { fit, ..CalibConfig::default() };
    let cal = calibrate(&shots, &table, &ball, &base, &cfg);

    let (cal_rms, cal_per) = rms(&cal);
    println!("\nCALIBRATED physics:");
    println!("  cushion_restitution e_c : {:.3}  (was {:.3})", cal.cushion_restitution, base.cushion_restitution);
    println!("  cushion_friction    f_c : {:.3}  (was {:.3})", cal.cushion_friction, base.cushion_friction);
    println!("  cloth slide  mu_slide   : {:.3}  (was {:.3})", cal.mu_slide, base.mu_slide);
    println!("  cloth roll   mu_roll    : {:.4} (was {:.4})", cal.mu_roll, base.mu_roll);
    println!("\nCALIBRATED: overall RMS {cal_rms:.0} mm  (baseline {base_rms:.0} mm)");
    for (n, (b, c)) in names.iter().zip(base_per.iter().zip(&cal_per)) {
        let mark = if c < b { "↓" } else { " " };
        println!("  {:<28} {:.0} -> {:.0} mm {mark}", n.rsplit('/').next().unwrap(), b, c);
    }
    let improve = (base_rms - cal_rms) / base_rms * 100.0;
    println!("\noverall improvement: {improve:.0}%");
}
