//! Ball identity and instantaneous physical state.

use crate::math::DVec3;

/// Stable identifier for a ball within a single game/scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BallId(pub u8);

/// The three balls of a three-cushion game. Each player owns a cue ball
/// (white or yellow); the red is the shared object ball.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BallColor {
    White,
    Yellow,
    Red,
}

/// Instantaneous rigid-body state of a single ball.
///
/// Positions are the ball *center* in table coordinates (meters); `z = radius`
/// when resting on the cloth. `angular_vel` is in rad/s about the world axes —
/// its `z` component is the "english" (sidespin) that dominates three-cushion
/// play, while the horizontal components encode follow/draw.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BallState {
    pub pos: DVec3,
    pub vel: DVec3,
    pub angular_vel: DVec3,
}

impl BallState {
    /// Velocity of the material point at the ball's contact with the cloth,
    /// projected into the table plane. This "slip velocity" is what kinetic
    /// friction opposes; when it reaches zero the ball is rolling without
    /// slipping.
    ///
    /// Derivation: the contact point sits at `r_c = (0, 0, -radius)` relative
    /// to the center, so its velocity is `v + ω × r_c`, which expands to
    /// `(v_x - R·ω_y, v_y + R·ω_x, 0)`.
    pub fn contact_slip(&self, radius: f64) -> DVec3 {
        let r_c = DVec3::new(0.0, 0.0, -radius);
        let slip = self.vel + self.angular_vel.cross(r_c);
        DVec3::new(slip.x, slip.y, 0.0)
    }

    /// Whether the ball is rolling without slipping (slip speed within `eps`).
    pub fn is_rolling(&self, radius: f64, eps: f64) -> bool {
        self.contact_slip(radius).length() <= eps
    }

    /// The angular velocity a rolling ball must have for its current linear
    /// velocity, preserving the vertical english component. Used to snap state
    /// exactly onto the rolling manifold at a slide→roll transition, avoiding
    /// numerical drift.
    pub fn rolling_angular_vel(&self, radius: f64) -> DVec3 {
        DVec3::new(-self.vel.y / radius, self.vel.x / radius, self.angular_vel.z)
    }
}
