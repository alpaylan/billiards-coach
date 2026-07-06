//! Validation for the Han 2005 ball–cushion model and on-table simulation.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams, TableSpec};
use billiards_engine::{han2005_rebound, simulate_table};

fn setup() -> (BallSpec, PhysicsParams, TableSpec) {
    (BallSpec::carom(), PhysicsParams::default(), TableSpec::carom_match())
}

/// A rolling ball hitting a cushion square-on reverses its normal velocity and
/// comes back slower (restitution < 1), with no sideways deflection when there
/// is no english.
#[test]
fn head_on_rebound_reverses_and_slows() {
    let (ball, phys, table) = setup();
    let v_in = 2.0;
    // Natural roll along +x toward the right rail: ω_y = v_x / R.
    let state = BallState {
        pos: DVec3::new(1.0, 0.0, ball.radius),
        vel: DVec3::new(v_in, 0.0, 0.0),
        angular_vel: DVec3::new(0.0, v_in / ball.radius, 0.0),
    };

    let out = han2005_rebound(state, DVec3::new(1.0, 0.0, 0.0), &ball, &phys, table.cushion_height);

    assert!(out.vel.x < 0.0, "normal velocity should reverse (got {})", out.vel.x);
    assert!(out.vel.x.abs() < v_in, "rebound must be slower than approach");
    assert!(out.vel.y.abs() < 1e-9, "no sideways deflection without english");
}

/// The signature three-cushion behavior: sidespin (english) bends the rebound.
/// Same approach, different english ⇒ different tangential (post-rebound)
/// velocity. Positive english about +z drives the tangential velocity down.
#[test]
fn english_bends_the_rebound() {
    let (ball, phys, table) = setup();
    let normal = DVec3::new(1.0, 0.0, 0.0);
    let approach = |w_z: f64| BallState {
        pos: DVec3::new(1.0, 0.5, ball.radius),
        vel: DVec3::new(2.0, 2.0, 0.0), // ~45° into the right rail
        angular_vel: DVec3::new(0.0, 0.0, w_z),
    };

    let plain = han2005_rebound(approach(0.0), normal, &ball, &phys, table.cushion_height);
    let english = han2005_rebound(approach(50.0), normal, &ball, &phys, table.cushion_height);

    let delta = (english.vel.y - plain.vel.y).abs();
    assert!(delta > 0.05, "english should meaningfully change the rebound angle (Δ={delta})");
    assert!(english.vel.y < plain.vel.y, "positive english should reduce tangential velocity");
    // Both still leave the cushion.
    assert!(plain.vel.x < 0.0 && english.vel.x < 0.0);
}

/// A struck ball on the full table never escapes the nose rectangle, at any
/// sampled instant.
#[test]
fn ball_stays_on_the_table() {
    let (ball, phys, table) = setup();
    let [min_x, max_x, min_y, max_y] = table.center_bounds(ball.radius);

    let state = BallState {
        pos: DVec3::new(0.0, 0.0, ball.radius),
        vel: DVec3::new(3.0, 1.3, 0.0),
        angular_vel: DVec3::ZERO,
    };
    let traj = simulate_table(state, &table, &ball, &phys);

    let n = 1000;
    for i in 0..=n {
        let t = i as f64 / n as f64 * traj.time_to_rest();
        let p = traj.state_at(t).pos;
        assert!(p.x >= min_x - 1e-6 && p.x <= max_x + 1e-6, "x={} out of bounds at t={t}", p.x);
        assert!(p.y >= min_y - 1e-6 && p.y <= max_y + 1e-6, "y={} out of bounds at t={t}", p.y);
        assert!(p.x.is_finite() && p.y.is_finite());
    }
}

/// A hard shot rebounds off at least one cushion and still comes to rest. Absent
/// cushions this ball would travel several meters off the table; with them, the
/// extra segments prove rebounds happened.
#[test]
fn hard_shot_bounces_and_rests() {
    let (ball, phys, table) = setup();
    let state = BallState {
        pos: DVec3::new(-1.3, -0.6, ball.radius),
        vel: DVec3::new(3.5, 1.6, 0.0),
        angular_vel: DVec3::ZERO,
    };
    let traj = simulate_table(state, &table, &ball, &phys);

    // slide + roll + rest would be 3 segments; more means cushion events.
    assert!(traj.segments.len() > 3, "expected cushion rebounds, got {} segments", traj.segments.len());
    assert!(traj.final_state().vel.length() < 1e-9, "ball must come to rest");

    let [min_x, max_x, min_y, max_y] = table.center_bounds(ball.radius);
    let rest = traj.final_state().pos;
    assert!(rest.x >= min_x - 1e-6 && rest.x <= max_x + 1e-6);
    assert!(rest.y >= min_y - 1e-6 && rest.y <= max_y + 1e-6);
}
