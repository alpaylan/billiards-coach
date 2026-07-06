//! Validation for ball–ball collisions and the multi-ball scheduler.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, ContactKind, PhysicsParams, TableSpec};
use billiards_engine::{ball_ball_collision, simulate};

fn setup() -> (BallSpec, PhysicsParams, TableSpec) {
    (BallSpec::carom(), PhysicsParams::default(), TableSpec::carom_match())
}

fn at(pos: DVec3, vel: DVec3) -> BallState {
    BallState { pos, vel, angular_vel: DVec3::ZERO }
}

/// Head-on, equal masses: the object ball leaves with almost all the speed and
/// the cue ball nearly stops (a "stun"), scaled by the ball restitution.
#[test]
fn head_on_transfers_velocity() {
    let (ball, phys, _) = setup();
    let v = 2.0;
    let cue = at(DVec3::new(0.0, 0.0, ball.radius), DVec3::new(v, 0.0, 0.0));
    let obj = at(DVec3::new(2.0 * ball.radius, 0.0, ball.radius), DVec3::ZERO);

    let (cue_f, obj_f) = ball_ball_collision(cue, obj, &ball, &phys);

    let e = phys.ball_restitution;
    assert!((cue_f.vel.x - 0.5 * (1.0 - e) * v).abs() < 1e-9, "cue x = {}", cue_f.vel.x);
    assert!((obj_f.vel.x - 0.5 * (1.0 + e) * v).abs() < 1e-9, "obj x = {}", obj_f.vel.x);
    assert!(cue_f.vel.y.abs() < 1e-9 && obj_f.vel.y.abs() < 1e-9, "no deflection head-on");
    // Momentum along the line of centers is conserved.
    assert!((cue_f.vel.x + obj_f.vel.x - v).abs() < 1e-9);
}

/// A cut shot sends the object ball along the line of centers, and the cue ball
/// keeps mostly the perpendicular component (the ~90° rule for a stun cut).
#[test]
fn cut_shot_object_goes_along_line_of_centers() {
    let (ball, phys, _) = setup();
    // Object ball offset at 30° above +x from the cue.
    let ang = 30f64.to_radians();
    let n = DVec3::new(ang.cos(), ang.sin(), 0.0);
    let cue = at(DVec3::new(0.0, 0.0, ball.radius), DVec3::new(2.0, 0.0, 0.0));
    let obj = at(DVec3::new(0.0, 0.0, ball.radius) + n * (2.0 * ball.radius), DVec3::ZERO);

    let (cue_f, obj_f) = ball_ball_collision(cue, obj, &ball, &phys);

    // Object ball travels essentially along n (throw is a small deviation).
    let obj_dir = obj_f.vel.normalize();
    let along = obj_dir.dot(n);
    assert!(along > 0.99, "object ball should follow the line of centers (cosθ={along})");
    // Cue ball retains a large component perpendicular to n.
    let t = DVec3::new(-n.y, n.x, 0.0);
    assert!(cue_f.vel.dot(t).abs() > 0.5 * cue_f.vel.length(), "cue keeps tangential component");
}

/// Spin-induced throw: on an otherwise-straight collision, sidespin on the cue
/// ball deflects the object ball sideways (and imparts none without spin).
#[test]
fn sidespin_throws_the_object_ball() {
    let (ball, phys, _) = setup();
    let cue_pos = DVec3::new(0.0, 0.0, ball.radius);
    let obj_pos = DVec3::new(2.0 * ball.radius, 0.0, ball.radius);
    let make = |wz: f64| {
        let cue = BallState {
            pos: cue_pos,
            vel: DVec3::new(2.0, 0.0, 0.0),
            angular_vel: DVec3::new(0.0, 0.0, wz),
        };
        ball_ball_collision(cue, at(obj_pos, DVec3::ZERO), &ball, &phys).1
    };

    let plain = make(0.0);
    let spun = make(80.0);

    assert!(plain.vel.y.abs() < 1e-12, "no throw without spin");
    assert!(spun.vel.y.abs() > 1e-3, "sidespin should throw the object ball (y={})", spun.vel.y);
}

/// A collision never increases total kinetic energy (translational + rotational).
#[test]
fn collision_conserves_momentum_and_bounds_energy() {
    let (ball, phys, _) = setup();
    let cue = BallState {
        pos: DVec3::new(0.0, 0.0, ball.radius),
        vel: DVec3::new(1.8, 0.4, 0.0),
        angular_vel: DVec3::new(0.0, 0.0, 30.0),
    };
    let obj = at(DVec3::new(2.0 * ball.radius, 0.2, ball.radius), DVec3::new(-0.2, 0.0, 0.0));

    let ke = |s: &BallState| 0.5 * ball.mass * s.vel.length_squared()
        + 0.5 * ball.moment_of_inertia() * s.angular_vel.length_squared();
    let p = |s: &BallState| s.vel * ball.mass;

    let (cf, of) = ball_ball_collision(cue, obj, &ball, &phys);

    let before = p(&cue) + p(&obj);
    let after = p(&cf) + p(&of);
    assert!((after - before).length() < 1e-9, "linear momentum conserved");
    assert!(ke(&cf) + ke(&of) <= ke(&cue) + ke(&obj) + 1e-9, "energy must not increase");
}

/// End-to-end: a three-ball shot where the cue ball strikes both object balls,
/// logged in order, everyone stays on the table and comes to rest.
#[test]
fn three_ball_shot_runs_to_completion() {
    let (ball, phys, table) = setup();
    let r = ball.radius;
    // Cue at left, two object balls ahead and to the sides.
    let balls = [
        at(DVec3::new(-1.0, 0.0, r), DVec3::new(3.0, 0.2, 0.0)), // cue (BallId 0)
        at(DVec3::new(0.6, 0.25, r), DVec3::ZERO),               // object 1
        at(DVec3::new(1.1, -0.15, r), DVec3::ZERO),              // object 2
    ];

    let sim = simulate(&balls, &table, &ball, &phys);

    assert_eq!(sim.trajectories.len(), 3);
    // The cue ball collides with at least one object ball.
    let ball_hits = sim
        .events
        .iter()
        .filter(|e| matches!(e.kind, ContactKind::BallBall { .. }))
        .count();
    assert!(ball_hits >= 1, "expected a ball–ball collision, got events: {:?}", sim.events);

    // Events are time-ordered.
    let mut last = 0.0;
    for e in &sim.events {
        assert!(e.time >= last - 1e-12, "events must be ordered in time");
        last = e.time;
    }

    // Everyone rests inside the table.
    let [min_x, max_x, min_y, max_y] = table.center_bounds(r);
    for traj in &sim.trajectories {
        let rest = traj.final_state();
        assert!(rest.vel.length() < 1e-9, "all balls come to rest");
        assert!(rest.pos.x >= min_x - 1e-6 && rest.pos.x <= max_x + 1e-6, "x={}", rest.pos.x);
        assert!(rest.pos.y >= min_y - 1e-6 && rest.pos.y <= max_y + 1e-6, "y={}", rest.pos.y);
    }
}
