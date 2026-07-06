//! End-to-end perception demo: render a known scene to a perspective image,
//! detect the balls, reconstruct the scene, and solve it — image → Scene → shot.
//! Writes the synthetic frame to a PNG you can open.
//!
//! Run: `cargo run -p billiards-vision --example reconstruct_demo`

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec};
use billiards_solver::{SolveConfig, solve};
use billiards_vision::{BallDetector, Calibration, ColorBlobDetector, Image, reconstruct_scene, render_scene};

fn main() {
    let table = TableSpec::carom_match();
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let r = ball.radius;

    // Ground-truth scene we'll try to recover.
    let truth = Scene::new(
        DVec3::new(-1.15, -0.45, r),
        vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)],
    );

    // A synthetic perspective camera, calibrated from where the table's four
    // corners land in the image.
    let image_corners = [(250.0, 80.0), (550.0, 80.0), (720.0, 400.0), (80.0, 400.0)];
    let calib = Calibration::from_table_corners(image_corners, &table).unwrap();

    let (w, h) = (800, 480);
    let image = render_scene(&truth, &table, r, &calib, w, h);
    let png = std::env::temp_dir().join("reconstruction.png");
    save_png(&png, &image);

    let detections = ColorBlobDetector::default().detect(&image);
    let recon = reconstruct_scene(&detections, &calib, r).expect("cue detected");

    println!("perception demo — image ({w}×{h}) → scene");
    println!("  synthetic frame written to {}", png.display());
    println!("  detected {} balls\n", detections.len());

    let err_mm = |a: DVec3, b: DVec3| ((a - b).length() * 1000.0).round();
    println!("  ball     truth (m)        recovered (m)     error");
    report("cue", truth.cue, recon.cue, err_mm);
    for (i, (t, rr)) in truth.objects.iter().zip(recon.objects.iter()).enumerate() {
        report(&format!("obj {}", i + 1), *t, *rr, err_mm);
    }

    // The reconstructed scene flows straight into the solver.
    match solve(&recon, &table, &ball, &phys, &SolveConfig::default()) {
        Some(s) => println!(
            "\n  solved reconstructed scene: aim {:+.1}°, speed {:.2} m/s, success {:.0}% — {}",
            s.action.aim.to_degrees(),
            s.action.speed,
            s.success_prob * 100.0,
            s.category()
        ),
        None => println!("\n  no scoring shot found for reconstructed scene"),
    }
}

fn report(name: &str, truth: DVec3, recon: DVec3, err_mm: impl Fn(DVec3, DVec3) -> f64) {
    println!(
        "  {name:<8} ({:+.3},{:+.3})   ({:+.3},{:+.3})   {:.0} mm",
        truth.x, truth.y, recon.x, recon.y, err_mm(truth, recon)
    );
}

fn save_png(path: &std::path::Path, img: &Image) {
    let mut buf = Vec::with_capacity(img.width * img.height * 3);
    for p in &img.pixels {
        buf.extend_from_slice(p);
    }
    image::save_buffer(path, &buf, img.width as u32, img.height as u32, image::ExtendedColorType::Rgb8)
        .expect("write png");
}
