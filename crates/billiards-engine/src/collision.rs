//! Ball–ball collision resolution with restitution and throw.
//!
//! Equal-mass impulse model in the horizontal plane, following the structure of
//! pooltool's `frictional_inelastic` model (itself based on Dr. Dave Alciatore's
//! throw analysis), and cross-checked against a first-principles derivation.
//!
//! The collision is resolved along the **line of centers** `n` (normal) and the
//! in-plane perpendicular `t` (tangent):
//!
//! - **Normal**: the relative normal velocity reverses, scaled by `e_b`. For
//!   equal masses this is the symmetric exchange
//!   `v₁ₙ' = ½[(1−e)v₁ₙ + (1+e)v₂ₙ]`, and likewise for ball 2.
//! - **Tangent (throw)**: the relative *surface* velocity at the contact point,
//!   `uₜ = (v₁ₜ − v₂ₜ) + R(ω₁𝓏 + ω₂𝓏)`, drives Coulomb friction. If it can be
//!   arrested within the friction cone (`|impulse| ≤ u_b·Pₙ`) the contact sticks
//!   (`uₜ → 0`); otherwise it slides at the cone limit. The tangential impulse
//!   deflects the object ball off the line of centers (collision- and
//!   spin-induced throw) and swaps a little `ω𝓏`.
//!
//! Follow/draw (`ω_x, ω_y`) is deliberately **not** consumed here: a center-ball
//! follow shot stuns at contact and then re-develops roll via the cloth, exactly
//! as the retained horizontal spin will do in the next motion phase.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, BallState, PhysicsParams};

/// Resolve a collision between two equal-mass balls in contact, returning their
/// post-impact states. `a` and `b` may be in any motion state.
pub fn ball_ball_collision(
    a: BallState,
    b: BallState,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> (BallState, BallState) {
    let r = ball.radius;
    let m = ball.mass;
    let ii = 0.4 * m * r * r; // (2/5) m R²
    let e = phys.ball_restitution;
    let u = phys.ball_friction;

    // Line of centers (a → b) and in-plane tangent, both unit vectors.
    let n = (b.pos - a.pos).normalize();
    let t = DVec3::new(-n.y, n.x, 0.0);

    let (a_vn, a_vt) = (a.vel.dot(n), a.vel.dot(t));
    let (b_vn, b_vt) = (b.vel.dot(n), b.vel.dot(t));

    // Normal: symmetric restitution exchange.
    let a_vn_f = 0.5 * ((1.0 - e) * a_vn + (1.0 + e) * b_vn);
    let b_vn_f = 0.5 * ((1.0 + e) * a_vn + (1.0 - e) * b_vn);

    // Normal impulse magnitude on ball a (= m·|Δv_n|).
    let p_n = m * (a_vn_f - a_vn).abs();

    // Tangential relative surface velocity at the contact point. Only ω_z (the
    // in-plane spin) contributes in the tangent direction; see module docs.
    let u_t = (a_vt - b_vt) + r * (a.angular_vel.z + b.angular_vel.z);

    // Coulomb decision: impulse to fully arrest slip (m/7·|u_t|) vs. cone limit.
    let j_stick = (m / 7.0) * u_t.abs();
    let j_mag = j_stick.min(u * p_n);
    let j = -u_t.signum() * j_mag; // signed tangential impulse on ball a, along +t

    let a_vel = a.vel + n * (a_vn_f - a_vn) + t * (j / m);
    let b_vel = b.vel + n * (b_vn_f - b_vn) - t * (j / m);

    // A +t impulse at contact +R·n torques both balls the same way about +z.
    let dwz = r * j / ii;
    let a_w = a.angular_vel + DVec3::new(0.0, 0.0, dwz);
    let b_w = b.angular_vel + DVec3::new(0.0, 0.0, dwz);

    (
        BallState { pos: a.pos, vel: a_vel, angular_vel: a_w },
        BallState { pos: b.pos, vel: b_vel, angular_vel: b_w },
    )
}
