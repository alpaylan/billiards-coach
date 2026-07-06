//! End-to-end perception: render a known scene to a perspective image, then
//! recover it. Validates calibration + detection + reconstruction against ground
//! truth without any captured video or trained model.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, Scene, TableSpec};
use billiards_vision::{BallDetector, Calibration, ColorBlobDetector, reconstruct_scene, render_scene};

fn image_corners() -> [(f64, f64); 4] {
    [(250.0, 80.0), (550.0, 80.0), (720.0, 400.0), (80.0, 400.0)]
}

#[test]
fn recovers_scene_from_synthetic_image() {
    let table = TableSpec::carom_match();
    let ball = BallSpec::carom();
    let r = ball.radius;
    let truth = Scene::new(
        DVec3::new(-1.15, -0.45, r),
        vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)],
    );

    let calib = Calibration::from_table_corners(image_corners(), &table).unwrap();
    let image = render_scene(&truth, &table, r, &calib, 800, 480);
    let detections = ColorBlobDetector::default().detect(&image);
    assert_eq!(detections.len(), 3, "all three balls detected");

    let recon = reconstruct_scene(&detections, &calib, r).unwrap();

    // White → cue; objects recovered in render order (yellow, red).
    assert!((recon.cue - truth.cue).length() < 5e-3, "cue off by {} m", (recon.cue - truth.cue).length());
    assert_eq!(recon.objects.len(), 2);
    for (got, want) in recon.objects.iter().zip(truth.objects.iter()) {
        assert!((*got - *want).length() < 5e-3, "object off by {} m", (*got - *want).length());
    }
}

#[test]
fn detections_land_inside_the_table() {
    let table = TableSpec::carom_match();
    let ball = BallSpec::carom();
    let r = ball.radius;
    let truth = Scene::new(DVec3::new(-0.5, 0.2, r), vec![DVec3::new(0.9, -0.3, r), DVec3::new(0.2, 0.5, r)]);

    let calib = Calibration::from_table_corners(image_corners(), &table).unwrap();
    let image = render_scene(&truth, &table, r, &calib, 800, 480);
    let recon = reconstruct_scene(&ColorBlobDetector::default().detect(&image), &calib, r).unwrap();

    let (hl, hw) = (table.length / 2.0, table.width / 2.0);
    for p in std::iter::once(recon.cue).chain(recon.objects.iter().copied()) {
        assert!(p.x.abs() <= hl + 1e-3 && p.y.abs() <= hw + 1e-3, "recovered pos off table: {p:?}");
    }
}
