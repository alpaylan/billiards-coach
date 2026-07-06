//! Hit reconstruction validated against ground truth: simulate a known cue
//! action, sample the resulting tracks (as the tracker would), then check the
//! fit recovers the action and reproduces the trajectory.

use std::f64::consts::TAU;

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, ContactKind, CueAction, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action, sample_tracks};

fn ang_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

fn setup() -> (BallSpec, PhysicsParams, TableSpec, Scene) {
    let ball = BallSpec::carom();
    let r = ball.radius;
    let scene = Scene::new(
        DVec3::new(-1.0, -0.3, r),
        vec![DVec3::new(0.5, 0.15, r), DVec3::new(0.9, -0.25, r)],
    );
    (ball, PhysicsParams::default(), TableSpec::carom_match(), scene)
}

fn test_cfg() -> FitConfig {
    FitConfig { aim_steps: 41, speed_steps: 12, offset_steps: 7, refine_iters: 50, ..FitConfig::default() }
}

#[test]
fn recovers_known_hit() {
    let (ball, phys, table, scene) = setup();
    let r = ball.radius;
    // A rich shot: strikes an object ball (reveals follow) and banks (reveals english).
    let truth = CueAction::from_tip_offset(0.29, 3.5, 0.20, 0.15, r);

    let sim = simulate(&scene.ball_states(&truth), &table, &ball, &phys);
    assert!(
        sim.events.iter().any(|e| matches!(e.kind, ContactKind::BallBall { .. })),
        "test shot must hit an object ball for follow/draw to be identifiable"
    );
    let observed = sample_tracks(&sim, 30.0);

    let res = fit_action(&scene, &observed, &table, &ball, &phys, &test_cfg());

    assert!(res.rms_m < 0.01, "trajectory not reproduced: rms {} m", res.rms_m);
    assert!(ang_diff(res.action.aim, truth.aim) < 0.06, "aim {} vs {}", res.action.aim, truth.aim);
    assert!((res.action.speed - truth.speed).abs() < 0.3, "speed {} vs {}", res.action.speed, truth.speed);
    let (h, v) = res.action.tip_offset(r);
    let (th, tv) = truth.tip_offset(r);
    assert!((h - th).abs() < 0.1 && (v - tv).abs() < 0.1, "tip ({h:.2},{v:.2}) vs ({th:.2},{tv:.2})");
}

/// Robust to tracking noise: perturb the observed positions by a few mm and the
/// recovered hit is still close.
#[test]
fn robust_to_tracking_noise() {
    let (ball, phys, table, scene) = setup();
    let r = ball.radius;
    let truth = CueAction::from_tip_offset(-0.15, 4.0, -0.18, 0.10, r);

    let sim = simulate(&scene.ball_states(&truth), &table, &ball, &phys);
    let mut observed = sample_tracks(&sim, 30.0);
    // Deterministic ~5 mm jitter so the test is reproducible.
    for (bi, track) in observed.iter_mut().enumerate() {
        for (k, (_, p)) in track.iter_mut().enumerate() {
            let s = ((bi * 7 + k * 13) as f64).sin();
            p.x += 0.005 * s;
            p.y += 0.005 * ((bi * 5 + k * 11) as f64).cos();
        }
    }

    let res = fit_action(&scene, &observed, &table, &ball, &phys, &test_cfg());

    assert!(res.rms_m < 0.03, "rms {} m too high under noise", res.rms_m);
    assert!(ang_diff(res.action.aim, truth.aim) < 0.08, "aim {} vs {}", res.action.aim, truth.aim);
    assert!((res.action.speed - truth.speed).abs() < 0.5, "speed {} vs {}", res.action.speed, truth.speed);
}
