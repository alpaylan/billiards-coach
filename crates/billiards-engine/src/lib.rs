//! Event-based physics engine for three-cushion billiards.
//!
//! The engine advances the world from one *event* to the next (phase
//! transitions today; ball–ball and ball–cushion collisions next), solving
//! each phase in closed form rather than time-stepping. This is both more
//! accurate and dramatically faster than fixed-step integration — which
//! matters because the solver will invoke the engine millions of times.
//!
//! ## Status
//! Implemented: single-ball motion (slide → roll → stop) with the standard
//! cloth-friction model, ball–cushion rebound (Han 2005 carom model), ball–ball
//! collision with throw, and a multi-ball event scheduler ([`simulate`]) — a
//! complete shot can now be simulated. Not yet implemented: english decay via
//! spinning friction, and validation against the diamond system. See
//! `docs/DESIGN.md`.

pub mod collision;
pub mod cushion;
pub mod sim;
pub mod world;

pub use collision::ball_ball_collision;
pub use cushion::{Rail, han2005_rebound};
pub use sim::{simulate_free, simulate_table};
pub use world::simulate;
