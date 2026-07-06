//! Track a frames directory with the ONNX detector — the Rust `track.py`.
//! Prints `.shot`-style rows (`color,t,x,y`) plus the detected shot segments,
//! for diffing against the Python tracker's output on the same frames.
//!
//!   ORT_DYLIB_PATH=… cargo run -p billiards-vision --features onnx \
//!       --example track_frames --release -- \
//!       python/detector.onnx FRAMES_DIR "19,19 264,15 270,492 23,492" vertical > rust.rows

use billiards_vision::detect::Image;
use billiards_vision::onnx::OnnxDetector;
use billiards_vision::track::{self, Orient, RawDet};

fn load_png(path: &std::path::Path) -> Image {
    let img = image::open(path).expect("read frame").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image { width: w, height: h, pixels: img.pixels().map(|p| [p[0], p[1], p[2]]).collect() }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, dir, corners, orient] = &args[..] else {
        eprintln!("usage: track_frames MODEL.onnx FRAMES_DIR \"u,v u,v u,v u,v\" horizontal|vertical");
        std::process::exit(2);
    };
    let corners: Vec<(f64, f64)> = corners
        .split_whitespace()
        .map(|p| {
            let (u, v) = p.split_once(',').expect("corner u,v");
            (u.parse().unwrap(), v.parse().unwrap())
        })
        .collect();
    let corners: [(f64, f64); 4] = corners.try_into().expect("need 4 corners");
    let orient = match orient.as_str() {
        "horizontal" => Orient::Horizontal,
        _ => Orient::Vertical,
    };

    let det = OnnxDetector::from_file(model).expect("load model");
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("frames dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();

    let t0 = std::time::Instant::now();
    let frames: Vec<(Image, Vec<RawDet>)> = paths
        .iter()
        .map(|p| {
            let img = load_png(p);
            let dets = det
                .detect_scored(&img)
                .expect("inference")
                .into_iter()
                .map(|d| RawDet { u: d.u, v: d.v, w: d.w, h: d.h, color: d.color, score: d.score })
                .collect();
            (img, dets)
        })
        .collect();
    eprintln!("{} frames detected in {:.1}s", paths.len(), t0.elapsed().as_secs_f64());

    // mask pad: track.py's learned path dilates with a 12×12 kernel ≈ 6 px
    let tracks = track::track_clip(frames, corners, orient, 6.0).expect("calibration");
    let shots = track::segment_shots(&tracks, track::FPS);
    eprintln!("segments: {shots:?}");

    let names = ["white", "yellow", "red"];
    for (ci, name) in names.iter().enumerate() {
        for (i, p) in tracks[ci].iter().enumerate() {
            if let Some((x, y)) = p {
                println!("{name},{:.4},{x:.4},{y:.4}", i as f64 / track::FPS);
            }
        }
    }
}
