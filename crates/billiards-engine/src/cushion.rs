//! Ball–cushion rebound via the **Han (2005)** model, *"Dynamics in Carom and
//! Three Cushion Billiards"* — the model tailored to exactly our game.
//!
//! Ported from the reference implementation in `pooltool`
//! (`physics/resolve/ball_cushion/han_2005`, MIT-licensed). The treatment is an
//! instantaneous impulse at a contact point sitting *above* the ball equator
//! (height `h`), which is what makes sidespin ("english") bend the rebound —
//! the defining feature of three-cushion play. The impulse is solved in a frame
//! whose `+x` axis is the cushion's outward normal, then rotated back.
//!
//! Notation follows the paper: `theta_a = asin(h/R − 1)` is the contact angle,
//! `sx, sy` the contact-point slip, `c` the normal approach speed, and the
//! sliding-vs-sticking branch decides how much tangential impulse is applied.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams, TableSpec};

/// The four straight cushions of a rectangular carom table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rail {
    Left,
    Right,
    Bottom,
    Top,
}

impl Rail {
    pub const ALL: [Rail; 4] = [Rail::Left, Rail::Right, Rail::Bottom, Rail::Top];

    /// Outward normal (pointing from the table interior toward the rail). A ball
    /// contacting this rail is moving in the `+normal` direction.
    pub fn outward_normal(self) -> DVec3 {
        match self {
            Rail::Left => DVec3::new(-1.0, 0.0, 0.0),
            Rail::Right => DVec3::new(1.0, 0.0, 0.0),
            Rail::Bottom => DVec3::new(0.0, -1.0, 0.0),
            Rail::Top => DVec3::new(0.0, 1.0, 0.0),
        }
    }

    /// The center-coordinate bound this rail imposes, from [`TableSpec::center_bounds`]:
    /// `(is_x_axis, bound_value)`.
    pub fn bound(self, table: &TableSpec, radius: f64) -> (bool, f64) {
        let [min_x, max_x, min_y, max_y] = table.center_bounds(radius);
        match self {
            Rail::Left => (true, min_x),
            Rail::Right => (true, max_x),
            Rail::Bottom => (false, min_y),
            Rail::Top => (false, max_y),
        }
    }
}

/// Active rotation of a vector about the `z` axis by `ang` radians.
fn rotate_z(v: DVec3, ang: f64) -> DVec3 {
    let (s, c) = ang.sin_cos();
    DVec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z)
}

/// Apply the Han 2005 cushion impulse to a ball contacting a cushion whose
/// outward normal is `normal` (a unit vector in the table plane). Returns the
/// post-impact state (position unchanged; linear and angular velocity updated).
pub fn han2005_rebound(
    state: BallState,
    normal: DVec3,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cushion_height: f64,
) -> BallState {
    let r = ball.radius;
    let m = ball.mass;
    let e = phys.cushion_restitution;
    let mu = phys.cushion_friction;

    // Into the cushion frame: rotate so the outward normal aligns with +x.
    let psi = normal.y.atan2(normal.x);
    let v = rotate_z(state.vel, -psi);
    let w = rotate_z(state.angular_vel, -psi);

    debug_assert!(v.x > 0.0, "ball must be approaching the cushion");

    let theta = (cushion_height / r - 1.0).asin();
    let (sin_t, cos_t) = theta.sin_cos();

    // Eqs 14: contact-point slip (sx, sy) and normal approach speed c.
    let sx = v.x * sin_t - v.z * cos_t + r * w.y;
    let sy = -v.y - r * w.z * cos_t + r * w.x * sin_t;
    let c = -v.x * cos_t;

    // Eqs 16.
    let ii = 0.4 * m * r * r; // moment of inertia, (2/5) m R²
    let a = 3.5 / m; // 7 / (2m)
    let b = 1.0 / m;

    // Eqs 17 & 20.
    let pz_e = -(1.0 + e) * c / b; // ≥ 0 (normal impulse)
    let abs_s0 = (sx * sx + sy * sy).sqrt();
    let pz_s = abs_s0 / a;

    let (px_e, py_e) = if pz_s <= mu * pz_e {
        // Eqs 18: contact stops slipping during the impact (sticking).
        (sx / a, sy / a)
    } else {
        // Eqs 19: contact keeps sliding through the impact.
        (mu * pz_e * sx / abs_s0, mu * pz_e * sy / abs_s0)
    };

    // Eqs 21 & 22: rotate the impulse from the contact-normal frame to the rail
    // frame.
    let px = -px_e * sin_t - pz_e * cos_t;
    let py = py_e;
    let pz = px_e * cos_t - pz_e * sin_t;

    // Eqs 23: apply impulse. z-velocity stays zero (2D / ball stays on cloth).
    let mut v_out = v;
    v_out.x += px / m;
    v_out.y += py / m;

    let mut w_out = w;
    w_out.x += -r / ii * py * sin_t;
    w_out.y += r / ii * (px * sin_t - pz * cos_t);
    w_out.z += r / ii * py * cos_t;

    // Back to the table frame.
    BallState {
        pos: state.pos,
        vel: rotate_z(v_out, psi),
        angular_vel: rotate_z(w_out, psi),
    }
}
