//! Perception: reconstruct a billiards [`Scene`](billiards_core::Scene) from a
//! 2D image.
//!
//! First vertical slice — the *geometric backbone*, end-to-end and testable
//! without captured video or trained models:
//!
//! 1. [`Homography`] / [`Calibration`] — relate image pixels to the table plane
//!    from four corner correspondences (the table is a known-dimension plane, so
//!    one view suffices; no full 3D reconstruction).
//! 2. [`BallDetector`] — the seam where a learned ONNX model will drop in;
//!    [`ColorBlobDetector`] is a working classical baseline for clean footage.
//! 3. [`reconstruct_scene`] — lift detections to table coordinates and assemble
//!    a domain-model `Scene` that flows straight into the editor and solver.
//! 4. [`render_scene`] — a synthetic camera for ground-truth pipeline tests.
//!
//! ## Not yet (later slices)
//! Learned detectors on real video (Python-trained, ONNX/`ort` inference),
//! automatic table-corner detection, multi-frame tracking, shot segmentation,
//! and radius-height parallax correction.

pub mod detect;
pub mod homography;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod reconstruct;
pub mod track;
pub mod render;

pub use detect::{BallDetector, ColorBlobDetector, Detection, Image};
pub use homography::Homography;
pub use reconstruct::{Calibration, reconstruct_scene};
pub use render::render_scene;
