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
///
/// Dispatches on `phys.ball_contact_steps`: `0` uses the closed-form
/// instantaneous model below; `> 0` integrates the contact through impulse
/// space (Mathavan-style), which additionally captures follow/draw friction
/// and ball-to-ball spin transfer.
pub fn ball_ball_collision(
    a: BallState,
    b: BallState,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> (BallState, BallState) {
    if phys.ball_contact_steps > 0 {
        return ball_ball_collision_integrated(a, b, ball, phys);
    }
    let r = ball.radius;
    let m = ball.mass;
    let ii = 0.4 * m * r * r; // (2/5) m R²
    let e = phys.ball_restitution;

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

    // Speed-dependent friction (Alciatore): slow grazing contact grips much
    // harder than a fast one, so throw is strongest on soft shots.
    let u = phys.ball_friction + phys.ball_friction_b * (-phys.ball_friction_c * u_t.abs()).exp();

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

/// Mathavan-style contact integration: march the collision through normal
/// impulse space in `phys.ball_contact_steps` increments. Each step applies a
/// slice of the normal impulse, then Coulomb friction opposite the *current*
/// full 3-D contact-point slip — so the slip direction may rotate mid-contact,
/// follow/draw participates (it slips vertically at the contact face), and
/// both balls exchange spin about all axes ("gearing" transfer), none of which
/// the instantaneous model above can express.
///
/// The normal exchange is identical to the closed form (Poisson's hypothesis
/// on equal masses reproduces Newton's restitution when friction is decoupled
/// from the normal direction, as here). Any vertical velocity the friction
/// imparts is dropped at the end: the engine keeps balls on the cloth, so a
/// follow shot's contact "hop" is treated as absorbed by table and gravity.
pub fn ball_ball_collision_integrated(
    a: BallState,
    b: BallState,
    ball: &BallSpec,
    phys: &PhysicsParams,
) -> (BallState, BallState) {
    let r = ball.radius;
    let m = ball.mass;
    let ii = 0.4 * m * r * r;

    let n = (b.pos - a.pos).normalize();
    let approach = (a.vel - b.vel).dot(n);
    if approach <= 0.0 {
        return (a, b); // separating — nothing to resolve
    }

    let (mut a_v, mut b_v) = (a.vel, b.vel);
    let (mut a_w, mut b_w) = (a.angular_vel, b.angular_vel);

    // Equal masses: compression consumes P_c = m·approach/2, restitution adds
    // e·P_c on top (Poisson).
    let p_total = 0.5 * m * approach * (1.0 + phys.ball_restitution);
    let steps = phys.ball_contact_steps;
    let dp = p_total / steps as f64;

    for _ in 0..steps {
        a_v -= n * (dp / m);
        b_v += n * (dp / m);

        // Relative surface velocity at the contact point (a's face against
        // b's), tangential to the contact — includes the vertical component
        // that follow/draw produces.
        let u = (a_v + a_w.cross(n * r)) - (b_v + b_w.cross(n * -r));
        let u_t = u - n * u.dot(n);
        let s = u_t.length();
        if s < 1e-9 {
            continue; // stuck: impulses through the centers can't re-slip it
        }

        let mu = phys.ball_friction + phys.ball_friction_b * (-phys.ball_friction_c * s).exp();
        // A tangential impulse j changes the relative surface speed by 7j/m
        // (2/m linear + 5/m rotational), so cap the slice at full arrest.
        let j = (mu * dp).min(s * m / 7.0);
        let f = u_t * (-j / s); // on a

        a_v += f / m;
        b_v -= f / m;
        // Friction at ±R·n torques both balls the same way (see module docs).
        let dw = (n * r).cross(f) / ii;
        a_w += dw;
        b_w += dw;
    }

    a_v.z = 0.0;
    b_v.z = 0.0;
    (
        BallState { pos: a.pos, vel: a_v, angular_vel: a_w },
        BallState { pos: b.pos, vel: b_v, angular_vel: b_w },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(vel: DVec3, w: DVec3) -> BallState {
        BallState { pos: DVec3::new(0.0, 0.0, 0.0), vel, angular_vel: w }
    }

    fn setup() -> (BallSpec, PhysicsParams, BallState, BallState) {
        let ball = BallSpec::carom();
        let phys = PhysicsParams { ball_contact_steps: 200, ..PhysicsParams::default() };
        // b one ball-gap along +x from a
        let a = state(DVec3::new(2.0, 0.0, 0.0), DVec3::ZERO);
        let b = BallState { pos: DVec3::new(2.0 * ball.radius, 0.0, 0.0), ..state(DVec3::ZERO, DVec3::ZERO) };
        (ball, phys, a, b)
    }

    /// With planar spin only (english), the integrated contact must converge
    /// to the closed-form instantaneous result: same slip direction all the
    /// way through, same Coulomb budget.
    #[test]
    fn integrated_converges_to_analytic_for_english_only() {
        let (ball, phys, mut a, b) = setup();
        a.vel = DVec3::new(1.5, 0.4, 0.0);
        a.angular_vel = DVec3::new(0.0, 0.0, 25.0);
        let inst = PhysicsParams { ball_contact_steps: 0, ..phys };
        let (ia, ib) = ball_ball_collision(a, b, &ball, &inst);
        let (ga, gb) = ball_ball_collision(a, b, &ball, &phys);
        assert!((ia.vel - ga.vel).length() < 1e-3, "cue vel {:?} vs {:?}", ia.vel, ga.vel);
        assert!((ib.vel - gb.vel).length() < 1e-3, "obj vel {:?} vs {:?}", ib.vel, gb.vel);
        assert!((ia.angular_vel - ga.angular_vel).length() < 0.05);
    }

    /// A rolling (follow) cue ball gears the object ball: the object leaves
    /// with a little backspin and the cue keeps most of its topspin — the
    /// transfer the instantaneous model cannot produce at all.
    #[test]
    fn follow_shot_transfers_gearing_spin() {
        let (ball, phys, mut a, b) = setup();
        let v = 2.0;
        a.vel = DVec3::new(v, 0.0, 0.0);
        a.angular_vel = DVec3::new(0.0, v / ball.radius, 0.0); // natural roll along +x
        let (ga, gb) = ball_ball_collision(a, b, &ball, &phys);
        // object ball: forward along +x with backspin (ω_y < 0 is backspin
        // for +x motion here? natural roll for +x is ω_y = v/R > 0)
        assert!(gb.vel.x > 0.9 * v, "object should take nearly all speed");
        assert!(gb.angular_vel.y < -1e-3, "object should gain reverse (gear) spin, got {}", gb.angular_vel.y);
        assert!(ga.angular_vel.y > 0.5 * v / ball.radius, "cue keeps most of its follow");
        // and the cue must not have gained vertical velocity
        assert_eq!(ga.vel.z, 0.0);
    }
}
