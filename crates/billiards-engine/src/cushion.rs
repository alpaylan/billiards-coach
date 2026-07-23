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

/// Effective cushion restitution at normal approach speed `v_n` and
/// tangential (along-rail) speed `v_t`, both m/s: linear in each — `v_n`
/// around the 1 m/s reference so the calibrated constant keeps its meaning,
/// `v_t` from zero (a perpendicular impact is the reference) — clamped to a
/// physical range.
fn effective_restitution(phys: &PhysicsParams, v_n: f64, v_t: f64) -> f64 {
    (phys.cushion_restitution
        + phys.cushion_restitution_slope * (v_n - 1.0)
        + phys.cushion_restitution_slope_t * v_t.abs())
    .clamp(0.3, 0.99)
}

/// How long (s) a ball takes to shed the disturbed state a cushion leaves it
/// in — the chain-restitution window (see `cushion_restitution_chain`).
pub const CHAIN_RECOVERY_T: f64 = 0.35;

/// Resolve a ball–cushion impact: Han 2005 closed form by default, Mathavan
/// 2010 numerical integration when `phys.cushion_contact_steps > 0`.
///
/// `recovery` ∈ [0, 1] is the chain factor: 0 for a settled (isolated)
/// arrival, rising to 1 as the ball's previous cushion contact approaches
/// this one (`1 − Δt/CHAIN_RECOVERY_T`). It boosts the effective restitution
/// by `phys.cushion_restitution_chain · recovery` — a measured state
/// dependence the flat single-e model cannot express.
pub fn cushion_rebound(
    state: BallState,
    normal: DVec3,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cushion_height: f64,
    recovery: f64,
) -> BallState {
    let boosted;
    let phys = if phys.cushion_restitution_chain != 0.0 && recovery > 0.0 {
        boosted = PhysicsParams {
            cushion_restitution: phys.cushion_restitution
                + phys.cushion_restitution_chain * recovery.clamp(0.0, 1.0),
            ..phys.clone()
        };
        &boosted
    } else {
        phys
    };
    if phys.cushion_contact_steps > 0 {
        mathavan2010_rebound(state, normal, ball, phys, cushion_height)
    } else {
        han2005_rebound(state, normal, ball, phys, cushion_height)
    }
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
    let mu = phys.cushion_friction;

    // Into the cushion frame: rotate so the outward normal aligns with +x.
    let psi = normal.y.atan2(normal.x);
    let v = rotate_z(state.vel, -psi);
    let w = rotate_z(state.angular_vel, -psi);

    debug_assert!(v.x > 0.0, "ball must be approaching the cushion");

    let e = effective_restitution(phys, v.x, v.y);

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

/// **Mathavan 2010** ball–cushion impact, *"A theoretical analysis of billiard
/// ball dynamics under cushion impacts"* (Proc. IMechE Part C), following the
/// reference port in pooltool (`ball_cushion/mathavan_2010`, MIT-licensed).
///
/// Unlike Han's single instantaneous impulse, the impact is integrated through
/// normal-impulse space with **two simultaneous contacts** — the cushion nose
/// at height `h` (friction `f_c`) and the table under the ball (friction
/// `mu_slide`, pressed by the impact) — whose slip directions rotate as the
/// ball's state evolves. Compression runs until the normal velocity reverses;
/// restitution then returns `e²` of the compression work (energetic
/// restitution), which is what makes rebound speed and angle vary with how
/// the spin evolves *during* the impact.
///
/// Frame: the paper works with the ball approaching along `+y`, so we rotate
/// by `π/2 − ψ` (Han's frame uses `+x`). Where pooltool takes friction
/// direction from `atan2` even at zero slip, we zero a contact's friction when
/// its slip vanishes (the paper's stick condition) to avoid chatter.
pub fn mathavan2010_rebound(
    state: BallState,
    normal: DVec3,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cushion_height: f64,
) -> BallState {
    let r = ball.radius;
    let m = ball.mass;
    let steps = phys.cushion_contact_steps.max(1);

    // Into the paper's frame: outward normal along +y.
    let psi = normal.y.atan2(normal.x);
    let rot = std::f64::consts::FRAC_PI_2 - psi;
    let v = rotate_z(state.vel, rot);
    let w = rotate_z(state.angular_vel, rot);

    debug_assert!(v.y > 0.0, "ball must be approaching the cushion");

    let sin_t = (cushion_height - r) / r;
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let mu_w = phys.cushion_friction;
    let mu_s = phys.mu_slide;
    let ee = effective_restitution(phys, v.y, v.x);

    let (mut vx, mut vy) = (v.x, v.y);
    let (mut wx, mut wy, mut wz) = (w.x, w.y, w.z);

    // One impulse slice `dp`: slip at both contacts (eqs 12–13), then the
    // coupled velocity / angular-velocity increments (eqs 15–17). Returns the
    // work done at the cushion's normal for this slice.
    let step = |vx: &mut f64,
                vy: &mut f64,
                wx: &mut f64,
                wy: &mut f64,
                wz: &mut f64,
                dp: f64|
     -> f64 {
        // Cushion contact (I) and table contact (C) slip.
        let (sxi, syi) = (*vx + *wy * r * sin_t - *wz * r * cos_t, -*vy * sin_t + *wx * r);
        let (sxc, syc) = (*vx - *wy * r, *vy + *wx * r);
        let si = (sxi * sxi + syi * syi).sqrt();
        let sc = (sxc * sxc + syc * syc).sqrt();
        // Friction direction cosines; a vanished slip sticks (no friction).
        let (ci, si_) = if si > 1e-9 { (sxi / si, syi / si) } else { (0.0, 0.0) };
        let (cc, sc_) = if sc > 1e-9 { (sxc / sc, syc / sc) } else { (0.0, 0.0) };

        // Table normal impulse per unit cushion impulse (the cushion presses
        // the ball into the slate; its friction's vertical part adds to that).
        let table_n = sin_t + mu_w * si_ * cos_t;

        *vx -= (mu_w * ci + mu_s * cc * table_n) * dp / m;
        *vy -= (cos_t - mu_w * sin_t * si_ + mu_s * sc_ * table_n) * dp / m;

        let f = 5.0 / (2.0 * m * r);
        *wx -= f * (mu_w * si_ + mu_s * sc_ * table_n) * dp;
        *wy -= f * (mu_w * ci * sin_t - mu_s * cc * table_n) * dp;
        *wz += f * (mu_w * ci * cos_t) * dp;

        dp * vy.abs() * cos_t
    };

    // Compression: integrate until the normal velocity reverses. The last
    // slice is linearly shrunk to land on vy = 0.
    let dp = m * vy / steps as f64;
    let mut work = 0.0;
    let mut guard = 0;
    while vy > 0.0 {
        let (pvx, pvy, pwx, pwy, pwz) = (vx, vy, wx, wy, wz);
        let dw = step(&mut vx, &mut vy, &mut wx, &mut wy, &mut wz, dp);
        if vy <= 0.0 {
            // Redo the slice at the interpolated size that ends at vy = 0.
            let vy_full = vy;
            (vx, vy, wx, wy, wz) = (pvx, pvy, pwx, pwy, pwz);
            let dpf = dp * (pvy / (pvy - vy_full)).clamp(0.0, 1.0);
            work += step(&mut vx, &mut vy, &mut wx, &mut wy, &mut wz, dpf);
            break;
        }
        work += dw;
        guard += 1;
        if guard > 20 * steps {
            break;
        }
    }

    // Restitution: return e² of the compression work through the same
    // dynamics; the final slice is sized to hit the target exactly.
    let target = ee * ee * work;
    let mut ret = 0.0;
    let mut guard = 0;
    while ret < target {
        // Predicted work of a full slice (|vy| grows from ~0 through
        // restitution, so the first slices contribute almost nothing).
        let dw_pred = dp * vy.abs() * cos_t;
        if ret + dw_pred > target && vy.abs() * cos_t > 1e-12 {
            let dpf = (target - ret) / (vy.abs() * cos_t);
            step(&mut vx, &mut vy, &mut wx, &mut wy, &mut wz, dpf);
            break;
        }
        ret += step(&mut vx, &mut vy, &mut wx, &mut wy, &mut wz, dp);
        guard += 1;
        if guard > 20 * steps {
            break;
        }
    }

    // Back to the table frame; the ball stays on the cloth (no z velocity).
    BallState {
        pos: state.pos,
        vel: rotate_z(DVec3::new(vx, vy, 0.0), -rot),
        angular_vel: rotate_z(DVec3::new(wx, wy, wz), -rot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (BallSpec, PhysicsParams, DVec3) {
        let ball = BallSpec::carom();
        let phys = PhysicsParams { cushion_contact_steps: 2000, ..PhysicsParams::default() };
        (ball, phys, DVec3::new(0.0, 1.0, 0.0)) // top rail
    }

    fn rolling_at(v: DVec3, r: f64) -> BallState {
        let w = DVec3::new(-v.y / r, v.x / r, 0.0);
        BallState { pos: DVec3::ZERO, vel: v, angular_vel: w }
    }

    /// A dead-perpendicular rolling impact rebounds straight back at roughly
    /// e·v — no sideways drift by symmetry, some extra loss to friction.
    #[test]
    fn mathavan_perpendicular_rebound() {
        let (ball, phys, n) = setup();
        let v_in = 2.0;
        let s = rolling_at(DVec3::new(0.0, v_in, 0.0), ball.radius);
        let out = mathavan2010_rebound(s, n, &ball, &phys, 7.0 * ball.radius / 5.0);
        assert!(out.vel.y < 0.0, "must rebound away, got {:?}", out.vel);
        assert!(out.vel.x.abs() < 1e-6, "symmetric impact must stay straight, got vx {}", out.vel.x);
        let e = phys.cushion_restitution;
        let ratio = -out.vel.y / v_in;
        assert!(
            ratio > 0.45 * e && ratio < 1.05 * e,
            "rebound ratio {ratio:.3} out of range for e {e}"
        );
    }

    /// English bends the rebound the same direction as Han 2005 predicts.
    #[test]
    fn mathavan_english_matches_han_direction() {
        let (ball, phys, n) = setup();
        let mut s = rolling_at(DVec3::new(0.3, 2.0, 0.0), ball.radius);
        s.angular_vel.z = 30.0;
        let h = 7.0 * ball.radius / 5.0;
        let han = han2005_rebound(s, n, &ball, &PhysicsParams { cushion_contact_steps: 0, ..phys }, h);
        let mat = mathavan2010_rebound(s, n, &ball, &phys, h);
        let d_han = han.vel.x - s.vel.x;
        let d_mat = mat.vel.x - s.vel.x;
        assert!(
            d_han.signum() == d_mat.signum(),
            "english must bend the same way: han Δvx {d_han:.3}, mathavan Δvx {d_mat:.3}"
        );
        assert!(mat.vel.y < 0.0);
    }

    /// A negative restitution slope makes fast impacts rebound relatively
    /// slower than slow ones (both models share the same effective e).
    #[test]
    fn restitution_slope_scales_with_speed() {
        let (ball, mut phys, n) = setup();
        phys.cushion_restitution_slope = -0.05;
        let h = 7.0 * ball.radius / 5.0;
        let slow = mathavan2010_rebound(rolling_at(DVec3::new(0.0, 1.0, 0.0), ball.radius), n, &ball, &phys, h);
        let fast = mathavan2010_rebound(rolling_at(DVec3::new(0.0, 4.0, 0.0), ball.radius), n, &ball, &phys, h);
        let (r_slow, r_fast) = (-slow.vel.y / 1.0, -fast.vel.y / 4.0);
        assert!(
            r_fast < r_slow - 0.02,
            "fast impact should rebound relatively slower: slow {r_slow:.3}, fast {r_fast:.3}"
        );
    }
}
