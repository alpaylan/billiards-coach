//! Ball detection: from an image to a set of (image point, color) detections.
//!
//! [`BallDetector`] is the seam the polyglot plan hinges on: the classical
//! [`ColorBlobDetector`] here is a real baseline for clean, well-lit footage,
//! and a model trained in Python and exported to ONNX (run in Rust via `ort`)
//! will implement the *same trait* for messy real-world video. Everything
//! downstream (calibration, reconstruction) only sees `Vec<Detection>`.

use billiards_core::BallColor;

/// A simple RGB image (row-major, `pixels[y*width + x]`).
#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 3]>,
}

impl Image {
    pub fn filled(width: usize, height: usize, color: [u8; 3]) -> Self {
        Self { width, height, pixels: vec![color; width * height] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [u8; 3] {
        self.pixels[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, color: [u8; 3]) {
        self.pixels[y * self.width + x] = color;
    }
}

/// A detected ball: its center in image pixels and its color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub u: f64,
    pub v: f64,
    pub color: BallColor,
}

/// Anything that can find balls in an image. Implemented by the classical
/// detector today and by a learned ONNX model later.
pub trait BallDetector {
    fn detect(&self, image: &Image) -> Vec<Detection>;
}

/// Classical detector: for each ball color, threshold matching pixels and take
/// the centroid of the largest connected blob. Robust on clean footage where
/// the three balls are distinctly colored against the cloth.
pub struct ColorBlobDetector {
    /// Minimum blob size (pixels) to count as a ball, rejecting speckle.
    pub min_pixels: usize,
}

impl Default for ColorBlobDetector {
    fn default() -> Self {
        Self { min_pixels: 20 }
    }
}

impl ColorBlobDetector {
    fn matches(color: BallColor, p: [u8; 3]) -> bool {
        let [r, g, b] = p;
        match color {
            BallColor::White => r > 200 && g > 200 && b > 200,
            BallColor::Yellow => r > 190 && g > 150 && b < 130,
            BallColor::Red => r > 150 && g < 120 && b < 120,
        }
    }

    /// Centroid of the largest connected component of pixels matching `color`.
    fn largest_blob(&self, image: &Image, color: BallColor) -> Option<(f64, f64)> {
        let (w, h) = (image.width, image.height);
        let mut visited = vec![false; w * h];
        let mut best: Option<(usize, f64, f64)> = None; // (size, sum_x, sum_y)
        let mut stack: Vec<(usize, usize)> = Vec::new();

        for sy in 0..h {
            for sx in 0..w {
                let idx = sy * w + sx;
                if visited[idx] || !Self::matches(color, image.get(sx, sy)) {
                    continue;
                }
                // Flood-fill this component (4-connectivity).
                let (mut size, mut sum_x, mut sum_y) = (0usize, 0.0, 0.0);
                stack.push((sx, sy));
                visited[idx] = true;
                while let Some((x, y)) = stack.pop() {
                    size += 1;
                    sum_x += x as f64;
                    sum_y += y as f64;
                    let neighbors = [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ];
                    for &(nx, ny) in &neighbors {
                        if nx < w && ny < h {
                            let nidx = ny * w + nx;
                            if !visited[nidx] && Self::matches(color, image.get(nx, ny)) {
                                visited[nidx] = true;
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
                if size >= self.min_pixels && best.is_none_or(|(bs, _, _)| size > bs) {
                    best = Some((size, sum_x, sum_y));
                }
            }
        }
        best.map(|(size, sx, sy)| (sx / size as f64, sy / size as f64))
    }
}

impl BallDetector for ColorBlobDetector {
    fn detect(&self, image: &Image) -> Vec<Detection> {
        [BallColor::White, BallColor::Yellow, BallColor::Red]
            .into_iter()
            .filter_map(|color| {
                self.largest_blob(image, color).map(|(u, v)| Detection { u, v, color })
            })
            .collect()
    }
}
