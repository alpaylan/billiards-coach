//! The learned ball detector, running the Python-trained model via ONNX Runtime.
//!
//! `detector.onnx` (exported by `python/export_onnx.py`) is the WHOLE
//! torchvision pipeline — internal resize/normalize, RPN, ROI heads, NMS — as
//! one graph with a FIXED input of `3×515×290` raw RGB floats in `0..=1`.
//! Fixed, because traced detection graphs bake input-size-dependent constants:
//! at any other resolution they compute different (wrong) activations. So this
//! side letterboxes every frame into that canvas — `letterbox()` here must stay
//! the mirror of `letterbox()` in `export_onnx.py`, which is parity-gated
//! against the PyTorch model on real frames.
//!
//! Parity (measured, `examples/onnx_detect.rs` vs the .pt model): native-size
//! frames are BIT-EXACT det-for-det; letterboxed odd-size frames keep the same
//! detections/labels/boxes with scores within ~0.05 (our bilinear vs cv2's
//! fixed-point resampler — an input difference, not a graph one).
//!
//! Linking: pyke's prebuilt STATIC onnxruntime for macOS references libc++
//! `to_chars` overloads (incl. long double) Apple's SDK never shipped, so this
//! crate uses ort's `load-dynamic`: set `ORT_DYLIB_PATH` to a runtime dylib —
//! the pip wheel's works:
//! `ORT_DYLIB_PATH=…/site-packages/onnxruntime/capi/libonnxruntime.<ver>.dylib`.
//! (`ort` is pinned `=2.0.0-rc.10`; rc.12 fails to compile under
//! `default-features = false`.)

use std::path::Path;
use std::sync::Mutex;

use billiards_core::BallColor;
use ort::session::Session;
use ort::value::Tensor;

use crate::detect::{BallDetector, Detection, Image};

/// Canonical model input (rows, cols) — the native overhead-inset frame.
pub const CANON_H: usize = 515;
pub const CANON_W: usize = 290;

/// A raw scored detection, before any thresholding or per-color assignment.
/// The tracker's joint color assignment wants all candidates with scores;
/// the plain [`BallDetector`] impl reduces these to one ball per color.
#[derive(Clone, Copy, Debug)]
pub struct ScoredDetection {
    /// Box center in SOURCE image pixels.
    pub u: f64,
    pub v: f64,
    /// Box size in source pixels (for the tracker's "a ball is small" gate).
    pub w: f64,
    pub h: f64,
    pub color: BallColor,
    pub score: f32,
}

pub struct OnnxDetector {
    // `Session::run` takes `&mut self`; the seam trait takes `&self`.
    session: Mutex<Session>,
    /// Detections below this score are dropped in [`BallDetector::detect`]
    /// (the raw [`Self::detect_scored`] keeps everything the graph emits).
    pub threshold: f32,
}

impl OnnxDetector {
    pub fn from_file(model: impl AsRef<Path>) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(model)?;
        Ok(Self { session: Mutex::new(session), threshold: 0.22 })
    }

    /// Run the model on a frame of any size; boxes come back in the frame's
    /// own pixel coordinates.
    pub fn detect_scored(&self, image: &Image) -> ort::Result<Vec<ScoredDetection>> {
        let (tensor, scale) = letterbox(image);
        let input = Tensor::from_array(([3usize, CANON_H, CANON_W], tensor))?;
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs!["image" => input])?;
        let (_, boxes) = outputs["boxes"].try_extract_tensor::<f32>()?;
        let (_, labels) = outputs["labels"].try_extract_tensor::<i64>()?;
        let (_, scores) = outputs["scores"].try_extract_tensor::<f32>()?;

        let mut out = Vec::new();
        for (i, (&label, &score)) in labels.iter().zip(scores).enumerate() {
            let color = match label {
                1 => BallColor::White,
                2 => BallColor::Yellow,
                3 => BallColor::Red,
                _ => continue,
            };
            let b = &boxes[i * 4..i * 4 + 4];
            out.push(ScoredDetection {
                u: (0.5 * (b[0] + b[2]) as f64) / scale,
                v: (0.5 * (b[1] + b[3]) as f64) / scale,
                w: ((b[2] - b[0]) as f64) / scale,
                h: ((b[3] - b[1]) as f64) / scale,
                color,
                score,
            });
        }
        Ok(out)
    }
}

impl BallDetector for OnnxDetector {
    /// Best detection per color above the threshold — the simple interface
    /// downstream reconstruction uses. (Scores arrive sorted by the graph's
    /// NMS, so the first hit per color is its best.)
    fn detect(&self, image: &Image) -> Vec<Detection> {
        let mut out: Vec<Detection> = Vec::new();
        let Ok(scored) = self.detect_scored(image) else { return out };
        for d in scored {
            if d.score < self.threshold || out.iter().any(|o| o.color == d.color) {
                continue;
            }
            out.push(Detection { u: d.u, v: d.v, color: d.color });
        }
        out
    }
}

/// Fit `image` into the canonical canvas: scale to fit (bilinear), pad
/// bottom/right with black. Returns the CHW float tensor and the scale;
/// canvas coordinates map back to source pixels as `p / scale`.
/// Mirror of `letterbox()` in `python/export_onnx.py`.
fn letterbox(image: &Image) -> (Vec<f32>, f64) {
    let (h, w) = (image.height, image.width);
    let s = (CANON_H as f64 / h as f64).min(CANON_W as f64 / w as f64);
    let (nh, nw) = ((h as f64 * s).round() as usize, (w as f64 * s).round() as usize);

    let mut tensor = vec![0.0f32; 3 * CANON_H * CANON_W];
    // cv2.INTER_LINEAR sampling: src = (dst + 0.5) / s - 0.5, clamped edges.
    for y in 0..nh {
        let sy = ((y as f64 + 0.5) / s - 0.5).clamp(0.0, (h - 1) as f64);
        let (y0, fy) = (sy.floor() as usize, sy.fract() as f32);
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..nw {
            let sx = ((x as f64 + 0.5) / s - 0.5).clamp(0.0, (w - 1) as f64);
            let (x0, fx) = (sx.floor() as usize, sx.fract() as f32);
            let x1 = (x0 + 1).min(w - 1);
            let (p00, p01) = (image.get(x0, y0), image.get(x1, y0));
            let (p10, p11) = (image.get(x0, y1), image.get(x1, y1));
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p01[c] as f32 * fx;
                let bot = p10[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                let v = top * (1.0 - fy) + bot * fy;
                tensor[c * CANON_H * CANON_W + y * CANON_W + x] = v / 255.0;
            }
        }
    }
    (tensor, s)
}
