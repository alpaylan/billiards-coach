//! Diamond-system regression guard.
//!
//! Encodes the finding from `examples/diamond_system.rs`: with running english,
//! the Han 2005 cushion banks reproduce the corner-5 relationship `T = D − F`.
//! We assert the physically meaningful invariants — the arrival is ~linear in
//! the aim with slope ≈ −1, and the intercept `D` matches the ~5 the corner-5
//! system assigns near the corner — rather than exact arrivals (the english and
//! speed are the system's free calibration). See `docs/DESIGN.md`.

use billiards_core::math::DVec3;
use billiards_core::{BallId, BallSpec, BallState, ContactKind, PhysicsParams, TableSpec};
use billiards_engine::simulate;

const DIAMOND: f64 = 0.355;

fn corner5_arrival(cue_pos: DVec3, f: f64, english: f64, speed: f64) -> Option<f64> {
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let table = TableSpec::carom_match();
    let (half_len, half_wid) = (table.length / 2.0, table.width / 2.0);
    let inset = table.contact_inset(ball.radius);

    let target = DVec3::new(half_len - f * DIAMOND, half_wid, ball.radius);
    let dir = (target - cue_pos).normalize();
    let cue = BallState { pos: cue_pos, vel: dir * speed, angular_vel: DVec3::new(0.0, 0.0, english) };

    let sim = simulate(&[cue], &table, &ball, &phys);
    let cushions: Vec<DVec3> = sim
        .events
        .iter()
        .filter(|e| matches!(e.kind, ContactKind::Cushion { ball } if ball == BallId(0)))
        .map(|e| sim.trajectories[0].state_at(e.time).pos)
        .collect();
    if cushions.len() < 3 {
        return None;
    }
    // Canonical corner-5 pattern: far long rail, short rail, near long rail.
    let (hx, hy) = (half_len - inset, half_wid - inset);
    let rail = |p: DVec3| -> char {
        if (p.y - hy).abs() < 0.02 { 'T' }
        else if (p.y + hy).abs() < 0.02 { 'B' }
        else if (p.x - hx).abs() < 0.02 { 'R' }
        else if (p.x + hx).abs() < 0.02 { 'L' }
        else { '?' }
    };
    let pattern: String = cushions[..3].iter().map(|&p| rail(p)).collect();
    if pattern != "TRB" {
        return None;
    }
    Some((half_len - cushions[2].x) / DIAMOND)
}

fn linfit(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let (sx, sy): (f64, f64) = (pts.iter().map(|p| p.0).sum(), pts.iter().map(|p| p.1).sum());
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (slope, (sy - slope * sx) / n)
}

/// With running english, corner-5 arrivals track `T = D − F`: slope ≈ −1,
/// intercept `D` ≈ 5 (corner), and the relationship is linear.
#[test]
fn corner5_reproduces_diamond_system() {
    let cue_pos = DVec3::new(-1.35, -0.60, BallSpec::carom().radius);
    let english = 80.0; // calibrated running english (see example)
    let speed = 4.0;

    let pts: Vec<(f64, f64)> = [2.0, 2.5, 3.0, 3.5, 4.0]
        .into_iter()
        .filter_map(|f| corner5_arrival(cue_pos, f, english, speed).map(|t| (f, t)))
        .collect();

    assert!(pts.len() >= 4, "expected valid corner-5 shots, got {}", pts.len());

    let (slope, intercept) = linfit(&pts);
    assert!((slope + 1.0).abs() < 0.2, "slope should be ≈ −1, got {slope:.3}");
    assert!((5.0..=6.5).contains(&intercept), "intercept D should be ≈ 5 (corner), got {intercept:.3}");

    // Linearity: every arrival within ~half a diamond of the T = D − F fit —
    // the diamond system's own accuracy.
    for &(f, t) in &pts {
        let predicted = slope * f + intercept;
        assert!((t - predicted).abs() < 0.5, "F={f}: arrival {t:.2} vs fit {predicted:.2}");
    }
}

/// The bank geometry is spin-dependent in the right direction: reverse/less
/// english banks *steeper* (more negative slope) than running english.
#[test]
fn running_english_opens_the_bank_angle() {
    let cue_pos = DVec3::new(-1.35, -0.60, BallSpec::carom().radius);
    let speed = 4.0;
    let slope_at = |english: f64| {
        let pts: Vec<(f64, f64)> = [2.0, 2.5, 3.0, 3.5, 4.0]
            .into_iter()
            .filter_map(|f| corner5_arrival(cue_pos, f, english, speed).map(|t| (f, t)))
            .collect();
        linfit(&pts).0
    };
    // More running english ⇒ slope closer to −1 (less steep) than no english.
    assert!(slope_at(80.0) > slope_at(0.0), "running english should open the bank");
    assert!(slope_at(0.0) > slope_at(-40.0), "reverse english should steepen the bank");
}
