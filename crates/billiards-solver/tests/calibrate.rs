//! Physics calibration validated against ground truth: simulate diverse shots
//! with *known, non-default* physics, then check the calibrator recovers those
//! parameters starting from the defaults — proving the system-ID works before
//! we trust it on real, noisy trajectories.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, CueAction, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::calibrate::{CalibConfig, CalibShot, calibrate};
use billiards_solver::fit::sample_tracks;

fn make_shot(
    cue: DVec3,
    action: CueAction,
    truth: &PhysicsParams,
    table: &TableSpec,
    ball: &BallSpec,
) -> CalibShot {
    let r = ball.radius;
    let scene = Scene::new(cue, vec![DVec3::new(1.0, 0.5, r), DVec3::new(-0.8, -0.4, r)]);
    let sim = simulate(&scene.ball_states(&action), table, ball, truth);
    CalibShot { scene, observed: sample_tracks(&sim, 30.0) }
}

#[test]
fn recovers_physics_from_synthetic_shots() {
    let ball = BallSpec::carom();
    let table = TableSpec::carom_match();
    let r = ball.radius;

    // The "real table" physics we'll try to recover — deliberately off the defaults.
    let truth = PhysicsParams {
        cushion_restitution: 0.80,
        cushion_friction: 0.15,
        mu_slide: 0.17,
        mu_roll: 0.013,
        ..PhysicsParams::default()
    };

    // Diverse shots so the four parameters are jointly identifiable.
    let mk = |cx, cy, aim, speed, h, v| {
        make_shot(DVec3::new(cx, cy, r), CueAction::from_tip_offset(aim, speed, h, v, r), &truth, &table, &ball)
    };
    let shots = vec![
        mk(-1.1, -0.3, 0.40, 4.0, 0.15, 0.0),
        mk(-1.0, 0.20, -0.50, 3.0, -0.20, 0.10),
        mk(0.50, -0.40, 2.20, 3.5, 0.10, -0.15),
        mk(-0.50, 0.40, 1.00, 2.5, 0.0, 0.20),
    ];

    // Start from the literature defaults (wrong for this table) and calibrate.
    let base = PhysicsParams::default();
    let rec = calibrate(&shots, &table, &ball, &base, &CalibConfig::default());

    println!(
        "recovered: e_c={:.3} f_c={:.3} mu_s={:.3} mu_r={:.4}  (truth 0.80/0.15/0.17/0.0130)",
        rec.cushion_restitution, rec.cushion_friction, rec.mu_slide, rec.mu_roll
    );

    assert!((rec.cushion_restitution - 0.80).abs() < 0.04, "e_c {}", rec.cushion_restitution);
    assert!((rec.mu_roll - 0.013).abs() < 0.003, "mu_roll {}", rec.mu_roll);
    assert!((rec.mu_slide - 0.17).abs() < 0.05, "mu_slide {}", rec.mu_slide);
    // Cushion friction is the weakest-constrained (only bends the rebound with english).
    assert!((rec.cushion_friction - 0.15).abs() < 0.12, "f_c {}", rec.cushion_friction);
}
