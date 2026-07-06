//! Synthetic table renderer — generates perspective test images from a known
//! scene, so the whole perception pipeline can be validated end-to-end against
//! ground truth without any captured video or trained model.
//!
//! Balls are drawn at the projection of their table position (treated as flat on
//! the cloth), which keeps the geometry exact for testing; real footage carries
//! a small radius-height parallax (see [`crate::reconstruct`]).

use billiards_core::{BallColor, Scene, TableSpec};

use crate::detect::Image;
use crate::reconstruct::Calibration;

const CLOTH: [u8; 3] = [27, 104, 66];
const SURROUND: [u8; 3] = [22, 22, 28];

fn ball_rgb(color: BallColor) -> [u8; 3] {
    match color {
        BallColor::White => [244, 244, 236],
        BallColor::Yellow => [240, 200, 48],
        BallColor::Red => [200, 54, 44],
    }
}

/// Render a scene to a perspective image using `calib` as the (synthetic)
/// camera. The object balls are drawn yellow then red, matching how
/// [`reconstruct_scene`](crate::reconstruct::reconstruct_scene) assigns identity.
pub fn render_scene(
    scene: &Scene,
    table: &TableSpec,
    radius: f64,
    calib: &Calibration,
    width: usize,
    height: usize,
) -> Image {
    let mut img = Image::filled(width, height, SURROUND);
    let (hl, hw) = (table.length / 2.0, table.width / 2.0);

    // Cloth: any pixel whose table-plane pre-image lies inside the nose rect.
    for y in 0..height {
        for x in 0..width {
            let (tx, ty) = calib.to_table(x as f64, y as f64);
            if tx.abs() <= hl && ty.abs() <= hw {
                img.set(x, y, CLOTH);
            }
        }
    }

    // Balls: cue (white), then objects (yellow, red).
    draw_ball(&mut img, calib, radius, scene.cue.x, scene.cue.y, BallColor::White);
    let object_colors = [BallColor::Yellow, BallColor::Red];
    for (o, &color) in scene.objects.iter().zip(object_colors.iter()) {
        draw_ball(&mut img, calib, radius, o.x, o.y, color);
    }
    img
}

fn draw_ball(img: &mut Image, calib: &Calibration, radius: f64, x: f64, y: f64, color: BallColor) {
    let (cu, cv) = calib.to_image(x, y);
    // Local projected radius: how far a ball-radius step in table-x lands.
    let (eu, ev) = calib.to_image(x + radius, y);
    let px_r = ((eu - cu).hypot(ev - cv)).max(3.0);
    let rgb = ball_rgb(color);

    let (w, h) = (img.width as f64, img.height as f64);
    let r2 = px_r * px_r;
    let x0 = (cu - px_r).floor().max(0.0) as usize;
    let x1 = (cu + px_r).ceil().min(w - 1.0) as usize;
    let y0 = (cv - px_r).floor().max(0.0) as usize;
    let y1 = (cv + px_r).ceil().min(h - 1.0) as usize;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let (dx, dy) = (px as f64 - cu, py as f64 - cv);
            if dx * dx + dy * dy <= r2 {
                img.set(px, py, rgb);
            }
        }
    }
}
