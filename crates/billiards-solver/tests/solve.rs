//! Solver behavior: it finds a genuinely scoring shot and reports sane metrics.

use billiards_core::math::DVec3;
use billiards_core::{BallId, BallSpec, PhysicsParams, Scene, TableSpec, three_cushion_score};
use billiards_engine::simulate;
use billiards_solver::{SolveConfig, solve};

fn setup() -> (BallSpec, PhysicsParams, TableSpec) {
    (BallSpec::carom(), PhysicsParams::default(), TableSpec::carom_match())
}

/// The solver finds a scoring shot for a solvable scene, and the shot it returns
/// actually scores when simulated deterministically.
#[test]
fn returns_a_shot_that_actually_scores() {
    let (ball, phys, table) = setup();
    let r = ball.radius;
    let scene = Scene::new(
        DVec3::new(-1.15, -0.45, r),
        vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)],
    );

    // A slightly coarser config keeps the test quick but still finds shots.
    let cfg = SolveConfig { aim_steps: 120, speed_steps: 8, offset_steps: 5, mc_samples: 24, ..SolveConfig::default() };
    let solution = solve(&scene, &table, &ball, &phys, &cfg).expect("scene is solvable");

    // The nominal recommended action must itself score.
    let states = scene.ball_states(&solution.action);
    let sim = simulate(&states, &table, &ball, &phys);
    assert!(three_cushion_score(&sim, BallId(0)), "recommended action should score");

    // Sane metrics.
    assert!(solution.success_prob > 0.0 && solution.success_prob <= 1.0);
    assert!((solution.difficulty() - (1.0 - solution.success_prob)).abs() < 1e-12);
    assert!(solution.scoring_cells >= 1);
}

/// Determinism: the same seed and config yields the same recommendation.
#[test]
fn is_deterministic() {
    let (ball, phys, table) = setup();
    let r = ball.radius;
    let scene = Scene::new(
        DVec3::new(-1.15, -0.45, r),
        vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)],
    );
    let cfg = SolveConfig { aim_steps: 90, speed_steps: 6, offset_steps: 4, mc_samples: 16, ..SolveConfig::default() };

    let a = solve(&scene, &table, &ball, &phys, &cfg).unwrap();
    let b = solve(&scene, &table, &ball, &phys, &cfg).unwrap();
    assert_eq!(a.action, b.action);
    assert_eq!(a.success_prob, b.success_prob);
}
