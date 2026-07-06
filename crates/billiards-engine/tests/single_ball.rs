//! Physics validation for single-ball free motion.
//!
//! These assert *closed-form* predictions of the cloth-friction model, so they
//! double as a regression guard on the engine and as executable documentation
//! of the physics. They need no data — exactly why the engine is Phase 0.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, MotionPhase, PhysicsParams};
use billiards_engine::simulate_free;

fn setup() -> (BallSpec, PhysicsParams) {
    (BallSpec::carom(), PhysicsParams::default())
}

fn on_table(vel: DVec3, ball: &BallSpec) -> BallState {
    BallState { pos: DVec3::new(0.0, 0.0, ball.radius), vel, angular_vel: DVec3::ZERO }
}

/// A ball struck flat (no spin) slides, then enters natural roll at exactly
/// (5/7) of its launch speed.
#[test]
fn flat_strike_rolls_at_five_sevenths() {
    let (ball, phys) = setup();
    let v0 = 2.0;
    let traj = simulate_free(on_table(DVec3::new(v0, 0.0, 0.0), &ball), &ball, &phys);

    assert_eq!(traj.segments[0].phase, MotionPhase::Sliding);
    assert_eq!(traj.segments[1].phase, MotionPhase::Rolling);

    let at_transition = traj.segments[1].state0;
    let expected = 5.0 / 7.0 * v0;
    assert!(
        (at_transition.vel.length() - expected).abs() < 1e-9,
        "roll speed {} != {expected}",
        at_transition.vel.length()
    );
    // And it is genuinely rolling without slipping at that point.
    assert!(at_transition.is_rolling(ball.radius, 1e-9));
}

/// A ball already in natural roll skips the slide phase and stops after the
/// closed-form rolling distance `v² / (2 μ_roll g)`.
#[test]
fn natural_roll_stopping_distance() {
    let (ball, phys) = setup();
    let v0 = 0.5;
    let mut state = on_table(DVec3::new(v0, 0.0, 0.0), &ball);
    state.angular_vel = state.rolling_angular_vel(ball.radius);

    let traj = simulate_free(state, &ball, &phys);

    // No sliding segment: straight to rolling.
    assert_eq!(traj.segments[0].phase, MotionPhase::Rolling);

    let expected = v0 * v0 / (2.0 * phys.mu_roll * phys.g);
    let travelled = traj.final_state().pos.x;
    assert!(
        (travelled - expected).abs() < 1e-9,
        "distance {travelled} != {expected}"
    );
}

/// Whatever the launch, the ball ends at rest, and time/position advance
/// monotonically through the segments.
#[test]
fn ball_comes_to_rest() {
    let (ball, phys) = setup();
    let traj = simulate_free(on_table(DVec3::new(1.5, 0.7, 0.0), &ball), &ball, &phys);

    assert!(traj.final_state().vel.length() < 1e-9);
    assert_eq!(traj.segments.last().unwrap().phase, MotionPhase::Stationary);

    let mut t = 0.0;
    for seg in &traj.segments {
        assert!(seg.t_start >= t - 1e-12, "segments must be time-ordered");
        t = seg.t_start;
    }
    assert!(traj.time_to_rest() > 0.0);
}

/// Pure topspin with no initial velocity: the slipping contact drives the ball
/// forward (a "follow" effect), so it must actually move.
#[test]
fn pure_topspin_generates_motion() {
    let (ball, phys) = setup();
    // Spin about +y rolls toward +x; with zero launch velocity the contact
    // slips and friction accelerates the center along +x.
    let state = BallState {
        pos: DVec3::new(0.0, 0.0, ball.radius),
        vel: DVec3::ZERO,
        angular_vel: DVec3::new(0.0, 30.0, 0.0),
    };
    let traj = simulate_free(state, &ball, &phys);
    assert!(traj.final_state().pos.x > 1e-4, "topspin should propel the ball");
}
