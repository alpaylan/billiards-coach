//! Turn detections into a domain-model [`Scene`], via a calibrated table.

use billiards_core::math::DVec3;
use billiards_core::{BallColor, Scene, TableSpec};

use crate::detect::Detection;
use crate::homography::Homography;

/// A calibrated table view: the homographies between image pixels and table
/// coordinates, in both directions.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub image_to_table: Homography,
    pub table_to_image: Homography,
}

impl Calibration {
    /// Calibrate from the four cushion-nose corners located in the image, given
    /// in the order **far-left, far-right, near-right, near-left** (matching the
    /// table corners `(∓L/2, ±W/2)`).
    pub fn from_table_corners(image_corners: [(f64, f64); 4], table: &TableSpec) -> Option<Self> {
        let (hl, hw) = (table.length / 2.0, table.width / 2.0);
        let table_corners = [(-hl, hw), (hl, hw), (hl, -hw), (-hl, -hw)];
        Some(Self {
            image_to_table: Homography::from_correspondences(image_corners, table_corners)?,
            table_to_image: Homography::from_correspondences(table_corners, image_corners)?,
        })
    }

    /// Lift an image point onto the table plane (meters).
    pub fn to_table(&self, u: f64, v: f64) -> (f64, f64) {
        self.image_to_table.apply(u, v)
    }

    /// Project a table point into the image (pixels).
    pub fn to_image(&self, x: f64, y: f64) -> (f64, f64) {
        self.table_to_image.apply(x, y)
    }
}

/// Reconstruct a [`Scene`] from detections: the white ball becomes the cue, the
/// others the object balls. Returns `None` if the cue ball wasn't detected.
///
/// Note: detections are lifted through the *plane* homography, so this assumes
/// the ball's imaged center sits over its table position — a good approximation
/// that ignores the few-mm parallax from the ball's radius-height. Correcting
/// that requires the camera pose and is a later refinement.
pub fn reconstruct_scene(detections: &[Detection], calib: &Calibration, radius: f64) -> Option<Scene> {
    let mut cue = None;
    let mut objects = Vec::new();
    for d in detections {
        let (x, y) = calib.to_table(d.u, d.v);
        let pos = DVec3::new(x, y, radius);
        if d.color == BallColor::White {
            cue = Some(pos);
        } else {
            objects.push(pos);
        }
    }
    Some(Scene::new(cue?, objects))
}
