//! Vector math for the domain model.
//!
//! We use double precision throughout (`glam::DVec3`). Billiards trajectories
//! accumulate many analytic phase transitions, and the solver will run the
//! engine millions of times; `f32` rounding is not worth the risk here.
//!
//! Coordinate convention: the table surface is the `z = 0` plane, `+z` points
//! up, and a ball resting on the cloth has its center at `z = radius`.

pub use glam::DVec3;

/// A small nonzero speed (m/s) below which linear motion is treated as stopped.
pub const V_EPSILON: f64 = 1e-6;

/// A small nonzero contact-slip speed (m/s) below which a ball is considered to
/// be rolling without slipping rather than sliding.
pub const SLIP_EPSILON: f64 = 1e-6;
