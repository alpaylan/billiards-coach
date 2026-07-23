//! Verification guard: does each reconstructed shot actually match the original?
//!
//! For every shot we fit the cue action, simulate under the game's OWN stored
//! calibration (so we verify exactly what the editor draws), and compare to the
//! tracked path — splitting the error into the cue's own path vs the object balls
//! it drives. Each shot is graded PASS / WARN / FAIL against fixed thresholds, so
//! bad reconstructions are discovered automatically instead of by eye. A per-game
//! `verify.json` + a printed summary result; run it over several games at once to
//! track model fidelity across tables without overfitting to one.
//!
//!   cargo run -p billiards-solver --example verify_match --release -- data/g0 data/g1 …

use std::{env, fs};

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec, Trajectory};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::shotfile;

const CUE_WARN_MM: f64 = 250.0; // above this the cue path visibly drifts
const CUE_FAIL_MM: f64 = 450.0; // above this the reconstruction is wrong — investigate

struct Shot {
    scene: Scene,
    observed: Vec<Vec<(f64, DVec3)>>,
    name: String,
}

fn load_shot(path: &str, table: &TableSpec, r: f64) -> Option<Shot> {
    let text = fs::read_to_string(path).ok()?;
    let s = shotfile::parse(&text, table, r)?;
    Some(Shot { scene: s.scene, observed: s.observed,
                name: path.rsplit('/').next().unwrap().to_string() })
}

/// Per-game physics from calibration.json, else the built-in default.
fn load_calibration(dir: &str) -> (PhysicsParams, bool) {
    let path = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let Ok(text) = fs::read_to_string(path) else { return (PhysicsParams::carom_calibrated(), false) };
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars().skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>().parse().ok()
    };
    match (get("cushion_restitution"), get("cushion_friction"), get("mu_slide"), get("mu_roll")) {
        (Some(e), Some(f), Some(s), Some(r)) => (PhysicsParams {
            cushion_restitution: e, cushion_friction: f, mu_slide: s, mu_roll: r,
            ..PhysicsParams::carom_calibrated()
        }, true),
        _ => (PhysicsParams::carom_calibrated(), false),
    }
}

fn ball_rms(traj: &Trajectory, obs: &[(f64, DVec3)]) -> f64 {
    let sse: f64 = obs.iter().map(|(t, p)| (traj.state_at(*t).pos - *p).length_squared()).sum();
    (sse / obs.len().max(1) as f64).sqrt() * 1000.0
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    if dirs.is_empty() {
        eprintln!("usage: verify_match <match_dir> [more dirs…]");
        return;
    }
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let fit = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let (mut g_pass, mut g_warn, mut g_fail, mut g_all) = (0, 0, 0, Vec::new());
    let mut g_obj: Vec<f64> = Vec::new();

    // Optional physics sweeps (none of these are among the calibrated 4):
    // BF/BF_B/BF_C override the ball–ball friction curve, MU_SPIN the english
    // decay — for isolating which model change explains a metric shift.
    let envf = |k: &str| env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
    let (bf, bf_b, bf_c, mu_sp) = (envf("BF"), envf("BF_B"), envf("BF_C"), envf("MU_SPIN"));
    let eb = envf("EB");
    let bb_steps: Option<u32> = env::var("BB_STEPS").ok().and_then(|s| s.parse().ok());
    let cushion_steps: Option<u32> = env::var("CUSHION_STEPS").ok().and_then(|s| s.parse().ok());
    let ec_slope = envf("EC_SLOPE");

    for dir in &dirs {
        let (mut phys, had_cal) = load_calibration(dir);
        if let Some(v) = bf {
            phys.ball_friction = v;
        }
        if let Some(v) = bf_b {
            phys.ball_friction_b = v;
        }
        if let Some(v) = bf_c {
            phys.ball_friction_c = v;
        }
        if let Some(v) = mu_sp {
            phys.mu_spin = v;
        }
        if let Some(v) = eb {
            phys.ball_restitution = v;
        }
        if let Some(v) = bb_steps {
            phys.ball_contact_steps = v;
        }
        if let Some(v) = cushion_steps {
            phys.cushion_contact_steps = v;
        }
        if let Some(v) = ec_slope {
            phys.cushion_restitution_slope = v;
        }
        let mut paths: Vec<String> = fs::read_dir(dir).into_iter().flatten().flatten()
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".shot")).collect();
        paths.sort();
        let shots: Vec<Shot> = paths.iter().filter_map(|p| load_shot(p, &table, ball.radius)).collect();
        if shots.is_empty() {
            println!("{dir}: no shots");
            continue;
        }

        let mut rows = Vec::new();
        for s in &shots {
            let f = fit_action(&s.scene, &s.observed, &table, &ball, &phys, &fit);
            let sim = simulate(&s.scene.ball_states(&f.action), &table, &ball, &phys);
            let cue = ball_rms(&sim.trajectories[0], &s.observed[0]);
            let obj = (1..s.observed.len())
                .map(|b| ball_rms(&sim.trajectories[b], &s.observed[b]))
                .fold(0.0_f64, f64::max);
            let grade = if cue > CUE_FAIL_MM { "FAIL" } else if cue > CUE_WARN_MM { "WARN" } else { "PASS" };
            rows.push((s.name.clone(), cue, obj, grade));
            g_all.push(cue);
            g_obj.push(obj);
        }
        let pass = rows.iter().filter(|r| r.3 == "PASS").count();
        let warn = rows.iter().filter(|r| r.3 == "WARN").count();
        let fail = rows.iter().filter(|r| r.3 == "FAIL").count();
        g_pass += pass; g_warn += warn; g_fail += fail;
        let mut cues: Vec<f64> = rows.iter().map(|r| r.1).collect();
        cues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = cues[cues.len() / 2];

        let name = dir.trim_end_matches('/').rsplit('/').next().unwrap();
        println!("{name}: {} shots · median cue {median:.0} mm · {pass} PASS / {warn} WARN / {fail} FAIL{}",
            shots.len(), if had_cal { "" } else { " (default physics — no calibration.json)" });
        for (n, cue, obj, grade) in rows.iter().filter(|r| r.3 != "PASS") {
            println!("    {grade}  {n:<16} cue {cue:.0} mm  obj {obj:.0} mm");
        }

        let worst: Vec<String> = {
            let mut r: Vec<_> = rows.iter().collect();
            r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            r.iter().take(3).map(|(n, c, _, _)| format!("{{\"shot\":\"{n}\",\"cue_mm\":{c:.0}}}")).collect()
        };
        let json = format!(
            "{{\n  \"n_shots\": {}, \"median_cue_mm\": {median:.0},\n  \
             \"pass\": {pass}, \"warn\": {warn}, \"fail\": {fail},\n  \
             \"worst\": [{}]\n}}\n", shots.len(), worst.join(", "));
        let _ = fs::write(format!("{}/verify.json", dir.trim_end_matches('/')), json);
    }

    if dirs.len() > 1 || !g_all.is_empty() {
        g_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        g_obj.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = g_all[g_all.len() / 2];
        let median_obj = g_obj[g_obj.len() / 2];
        let total = g_pass + g_warn + g_fail;
        println!("\n=== overall: {total} shots · median cue {median:.0} mm · median obj {median_obj:.0} mm · \
                  {g_pass} PASS / {g_warn} WARN / {g_fail} FAIL ({:.0}% pass) ==={}",
            100.0 * g_pass as f64 / total.max(1) as f64,
            bf.map(|b| format!("  [BF={b}]")).unwrap_or_default());
    }
}
