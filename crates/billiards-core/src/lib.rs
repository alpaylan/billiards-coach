//! Shared domain model for the billiards coach.
//!
//! This crate is the *spine* of the whole system: the reconstruction pipeline,
//! the interactive editor, the solver, and the analytics layer all speak these
//! same types. It intentionally contains **no physics policy** (when collisions
//! happen, how cushions rebound) — only the vocabulary of the domain and the
//! representation of a computed [`Trajectory`]. The `billiards-engine` crate
//! decides *how* trajectories are produced; everything else only reads them.

pub mod ball;
pub mod math;
pub mod scene;
pub mod scoring;
pub mod simulation;
pub mod table;
pub mod trajectory;

pub use ball::{BallColor, BallId, BallState};
pub use math::DVec3;
pub use scene::{CueAction, Scene};
pub use scoring::three_cushion_score;
pub use simulation::{ContactEvent, ContactKind, Simulation};
pub use table::{BallSpec, PhysicsParams, TableSpec};
pub use trajectory::{MotionPhase, MotionSegment, Trajectory};
