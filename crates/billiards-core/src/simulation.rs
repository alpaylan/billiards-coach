//! The result of simulating a full multi-ball shot: one [`Trajectory`] per ball
//! plus a time-ordered log of contact events.
//!
//! The event log is what three-cushion *scoring* will read — a point requires
//! the cue ball to contact both object balls with at least three cushion
//! contacts in between — so we record the ordered sequence now even though the
//! scorer itself is a later phase.

use crate::ball::BallId;
use crate::trajectory::Trajectory;

/// A contact that occurred during a shot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactKind {
    /// A ball touched a cushion.
    Cushion { ball: BallId },
    /// Two balls collided.
    BallBall { a: BallId, b: BallId },
}

/// A contact event, tagged with the time it occurred.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent {
    pub time: f64,
    pub kind: ContactKind,
}

/// The full outcome of a shot: a trajectory per ball (indexed to match the input
/// order, i.e. trajectory `i` belongs to `BallId(i)`) and the ordered contacts.
#[derive(Clone, Debug, Default)]
pub struct Simulation {
    pub trajectories: Vec<Trajectory>,
    pub events: Vec<ContactEvent>,
}

impl Simulation {
    /// Time at which the last ball comes to rest.
    pub fn settled_time(&self) -> f64 {
        self.trajectories
            .iter()
            .map(Trajectory::time_to_rest)
            .fold(0.0, f64::max)
    }

    /// Number of cushion contacts made by a given ball.
    pub fn cushion_count(&self, ball: BallId) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e.kind, ContactKind::Cushion { ball: b } if b == ball))
            .count()
    }
}
