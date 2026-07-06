//! Diamond-system validation (corner-5, Dr. Dave TP 7.2).
//!
//! The corner-5 system predicts the third-rail arrival `T = D − F`, where `F` is
//! the first-rail aim diamond and `D` is the cue-ball position number, under
//! *running english at medium speed*. Since the english amount and speed are the
//! system's free calibration, the sharp physical claim we can test is the
//! **slope**: for a fixed cue position, arrival `T` should fall ~1 diamond per
//! diamond of aim `F` (dT/dF ≈ −1), *for shots that actually follow the corner-5
//! pattern* (first rail = far long, second = short, third = near long). This
//! exercises the bank geometry of the Han 2005 cushion model end-to-end.
//!
//! Run: `cargo run -p billiards-engine --example diamond_system`

use billiards_core::math::DVec3;
use billiards_core::{BallId, BallSpec, BallState, ContactKind, PhysicsParams, TableSpec};
use billiards_engine::simulate;

const DIAMOND: f64 = 0.355; // m, = length/8 = width/4

fn number_to_x(n: f64, half_len: f64) -> f64 {
    half_len - n * DIAMOND
}
fn x_to_number(x: f64, half_len: f64) -> f64 {
    (half_len - x) / DIAMOND
}

/// Which rail a cushion-contact position lies on.
fn rail_of(p: DVec3, table: &TableSpec, r: f64) -> char {
    let inset = table.contact_inset(r);
    let (hx, hy) = (table.length / 2.0 - inset, table.width / 2.0 - inset);
    let tol = 0.02;
    if (p.y - hy).abs() < tol {
        'T' // top (far) long rail
    } else if (p.y + hy).abs() < tol {
        'B' // bottom (near) long rail
    } else if (p.x - hx).abs() < tol {
        'R' // right short rail
    } else if (p.x + hx).abs() < tol {
        'L' // left short rail
    } else {
        '?'
    }
}

/// Simulate a corner-5 attempt. Returns the third-rail arrival diamond `T` only
/// if the shot follows the canonical pattern first=Top, second=Right, third=Bottom.
fn corner5_arrival(
    cue_pos: DVec3,
    f: f64,
    english: f64,
    speed: f64,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> Option<f64> {
    let half_len = table.length / 2.0;
    let half_wid = table.width / 2.0;
    let target = DVec3::new(number_to_x(f, half_len), half_wid, ball.radius);
    let dir = (target - cue_pos).normalize();
    let cue = BallState { pos: cue_pos, vel: dir * speed, angular_vel: DVec3::new(0.0, 0.0, english) };

    let sim = simulate(&[cue], table, ball, phys);
    let cushions: Vec<DVec3> = sim
        .events
        .iter()
        .filter(|e| matches!(e.kind, ContactKind::Cushion { ball } if ball == BallId(0)))
        .map(|e| sim.trajectories[0].state_at(e.time).pos)
        .collect();
    if cushions.len() < 3 {
        return None;
    }
    let pattern: String = cushions[..3].iter().map(|&p| rail_of(p, table, ball.radius)).collect();
    if pattern != "TRB" {
        return None;
    }
    Some(x_to_number(cushions[2].x, half_len))
}

fn linfit(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (slope, (sy - slope * sx) / n)
}

fn main() {
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let table = TableSpec::carom_match();

    let cue_pos = DVec3::new(-1.35, -0.60, ball.radius);
    let speed = 4.0;
    let f_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

    println!("Diamond-system corner-5 validation  (valid TRB shots only; T ≈ D − F, slope ≈ −1)");
    println!("cue at ({:+.2},{:+.2}), speed {speed} m/s\n", cue_pos.x, cue_pos.y);
    print!("  english |");
    for f in f_values {
        print!(" F={f:<4}");
    }
    println!(" | slope   D≈");

    for english in [-40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0] {
        let mut pts = Vec::new();
        print!("  {english:+6.0}  |");
        for &f in &f_values {
            match corner5_arrival(cue_pos, f, english, speed, &table, &ball, &phys) {
                Some(t) => {
                    print!(" {t:5.2}");
                    pts.push((f, t));
                }
                None => print!("   -- "),
            }
        }
        if pts.len() >= 3 {
            let (slope, intercept) = linfit(&pts);
            println!(" | {slope:+5.2}   {intercept:5.2}");
        } else {
            println!(" | (too few valid)");
        }
    }
}
