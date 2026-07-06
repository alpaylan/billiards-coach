//! Run the ONNX ball detector on real frames — the Rust half of the
//! cross-language parity check (compare against the PyTorch model's output
//! for the same frames; see python/export_onnx.py).
//!
//!   cargo run -p billiards-vision --features onnx --example onnx_detect --release -- \
//!       python/detector.onnx FRAME.png [more frames…]

use billiards_vision::detect::Image;
use billiards_vision::onnx::OnnxDetector;

fn load_png(path: &str) -> Image {
    let img = image::open(path).expect("read frame").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    Image { width: w, height: h, pixels }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: onnx_detect MODEL.onnx FRAME.png…");
    let det = OnnxDetector::from_file(&model).expect("load model");
    for path in args {
        let img = load_png(&path);
        let t0 = std::time::Instant::now();
        let mut scored = det.detect_scored(&img).expect("inference");
        let ms = t0.elapsed().as_millis();
        scored.retain(|d| d.score >= 0.04);
        println!("{path}  ({ms} ms)");
        for d in &scored {
            println!("  {:?} {:.3} at ({:.1},{:.1}) box {:.0}x{:.0}", d.color, d.score, d.u, d.v, d.w, d.h);
        }
    }
}
