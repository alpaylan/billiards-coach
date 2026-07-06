//! The [`Trajectory`] contract: the shared representation of a ball's motion
//! over time that every downstream consumer (editor playback, solver scoring,
//! analytics, reconstruction fitting) reads.
//!
//! A trajectory is a sequence of [`MotionSegment`]s. Each segment covers one
//! motion phase during which the acceleration is **constant**, so the state at
//! any instant is a closed-form quadratic — no time-stepping, exactly
//! samplable, and cheap. The engine's job is to compute *where* the phase
//! boundaries fall; sampling within a segment lives here.

use crate::ball::BallState;

/// Which regime of motion a segment describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionPhase {
    /// The contact point is slipping on the cloth; kinetic friction dominates.
    Sliding,
    /// Rolling without slipping; only rolling resistance decelerates the ball.
    Rolling,
    /// At rest (the final segment; open-ended in time).
    Stationary,
}

/// One constant-acceleration phase of a single ball's motion.
#[derive(Clone, Copy, Debug)]
pub struct MotionSegment {
    pub phase: MotionPhase,
    pub t_start: f64,
    /// End time; `f64::INFINITY` for the final stationary segment.
    pub t_end: f64,
    pub state0: BallState,
    /// Constant linear acceleration during the segment (m/s²).
    pub lin_acc: crate::math::DVec3,
    /// Constant angular acceleration during the segment (rad/s²).
    pub ang_acc: crate::math::DVec3,
}

impl MotionSegment {
    pub fn duration(&self) -> f64 {
        self.t_end - self.t_start
    }

    /// Whether time `t` falls within this segment's half-open interval.
    pub fn contains(&self, t: f64) -> bool {
        t >= self.t_start && (t < self.t_end || self.t_end.is_infinite())
    }

    /// Closed-form state at absolute time `t`, clamped to the segment's span.
    pub fn state_at(&self, t: f64) -> BallState {
        let raw = t - self.t_start;
        let dt = if self.t_end.is_infinite() {
            raw.max(0.0)
        } else {
            raw.clamp(0.0, self.duration())
        };
        BallState {
            pos: self.state0.pos + self.state0.vel * dt + self.lin_acc * (0.5 * dt * dt),
            vel: self.state0.vel + self.lin_acc * dt,
            angular_vel: self.state0.angular_vel + self.ang_acc * dt,
        }
    }
}

/// A single ball's full motion, start to rest.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub segments: Vec<MotionSegment>,
}

impl Trajectory {
    pub fn new(segments: Vec<MotionSegment>) -> Self {
        Self { segments }
    }

    /// Time at which the ball comes to rest (start of the final stationary
    /// segment), or 0 for an empty trajectory.
    pub fn time_to_rest(&self) -> f64 {
        self.segments.last().map_or(0.0, |s| s.t_start)
    }

    /// State at absolute time `t`. Before the start returns the initial state;
    /// after rest returns the resting state.
    pub fn state_at(&self, t: f64) -> BallState {
        for seg in &self.segments {
            if seg.contains(t) {
                return seg.state_at(t);
            }
        }
        self.final_state()
    }

    /// The resting state of the ball.
    pub fn final_state(&self) -> BallState {
        self.segments
            .last()
            .map_or_else(BallState::default, |s| s.state_at(s.t_start))
    }
}
