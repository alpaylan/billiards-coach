//! A shot setup: ball positions plus the cue action applied to the cue ball.
//!
//! This is the input the solver searches over and the editor manipulates. The
//! cue ball is always `BallId(0)`; the remaining balls are the object balls.

use crate::ball::BallState;
use crate::math::DVec3;

/// A planar cue strike, parameterised the way a player thinks about one.
///
/// `aim` is the direction of travel (radians, table frame). `sidespin` is the
/// english (`ω_z`, rad/s). `follow` is topspin/draw: spin (rad/s) about the
/// horizontal axis perpendicular to the aim — positive follows, negative draws.
/// Cue elevation (massé) is a future extension; for now the cue is level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CueAction {
    pub aim: f64,
    pub speed: f64,
    pub sidespin: f64,
    pub follow: f64,
}

impl CueAction {
    /// The cue ball's initial state when struck from `pos`. Sidespin is about
    /// the vertical axis; follow/draw is about the horizontal axis to the left
    /// of the aim (`ẑ × âim`), so positive `follow` rolls the ball forward.
    pub fn initial(&self, pos: DVec3) -> BallState {
        let (s, c) = self.aim.sin_cos();
        BallState {
            pos,
            vel: DVec3::new(self.speed * c, self.speed * s, 0.0),
            angular_vel: DVec3::new(0.0, 0.0, self.sidespin) + DVec3::new(-s, c, 0.0) * self.follow,
        }
    }

    /// Where the cue tip contacts the ball face, as fractions of the ball radius
    /// (`horizontal` = english side, `vertical` = above/below center). Follows
    /// from `ω = 5·(offset·R)·v / (2R²)`, i.e. `offset/R = 2R·ω / (5v)`. Note the
    /// dependence on speed: the same spin needs a larger offset at lower speed.
    pub fn tip_offset(&self, radius: f64) -> (f64, f64) {
        if self.speed.abs() < 1e-9 {
            return (0.0, 0.0);
        }
        let k = 2.0 * radius / (5.0 * self.speed);
        (k * self.sidespin, k * self.follow)
    }

    /// Inverse of [`tip_offset`]: the spins produced by striking at a tip offset
    /// (fractions of the radius) at the given aim and speed.
    pub fn from_tip_offset(aim: f64, speed: f64, horizontal: f64, vertical: f64, radius: f64) -> Self {
        let k = 5.0 * speed / (2.0 * radius);
        Self { aim, speed, sidespin: k * horizontal, follow: k * vertical }
    }
}

/// A configuration of balls on the table (positions only), with the cue ball
/// distinguished. Applying a [`CueAction`] yields the initial states to simulate.
#[derive(Clone, Debug)]
pub struct Scene {
    pub cue: DVec3,
    pub objects: Vec<DVec3>,
}

impl Scene {
    pub fn new(cue: DVec3, objects: Vec<DVec3>) -> Self {
        Self { cue, objects }
    }

    /// Initial ball states: cue (index 0) struck by `action`, objects at rest.
    pub fn ball_states(&self, action: &CueAction) -> Vec<BallState> {
        let mut states = Vec::with_capacity(1 + self.objects.len());
        states.push(action.initial(self.cue));
        for &o in &self.objects {
            states.push(BallState { pos: o, vel: DVec3::ZERO, angular_vel: DVec3::ZERO });
        }
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BallSpec;

    #[test]
    fn tip_offset_roundtrips() {
        let r = BallSpec::carom().radius;
        let a = CueAction { aim: 0.7, speed: 3.5, sidespin: 40.0, follow: -15.0 };
        let (h, v) = a.tip_offset(r);
        let b = CueAction::from_tip_offset(a.aim, a.speed, h, v, r);
        assert!((a.sidespin - b.sidespin).abs() < 1e-9 && (a.follow - b.follow).abs() < 1e-9);
    }

    #[test]
    fn natural_roll_is_forty_percent_above_center() {
        // Striking for natural roll (ω = v/R about the follow axis) corresponds
        // to a tip contact 0.4·R above center — the textbook value.
        let r = BallSpec::carom().radius;
        let speed = 3.0;
        let a = CueAction { aim: 0.0, speed, sidespin: 0.0, follow: speed / r };
        let (h, v) = a.tip_offset(r);
        assert!(h.abs() < 1e-12);
        assert!((v - 0.4).abs() < 1e-9, "natural roll offset {v} != 0.4");
    }
}
