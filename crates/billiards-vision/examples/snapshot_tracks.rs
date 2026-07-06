//! Tracker snapshot testing: run the ONNX detector + tracker over a curated
//! sample of frame directories and compare the full row output against the
//! APPROVED snapshot. Inference is ~45 s per shot, so the sample is a manifest
//! (`snapshots/tracks/manifest.txt`), not the whole corpus:
//!
//!   # comment lines allowed; one entry per line:
//!   # NAME  FRAMES_DIR  "u,v u,v u,v u,v"  vertical|horizontal
//!
//!   ORT_DYLIB_PATH=… cargo run -p billiards-vision --features onnx \
//!       --example snapshot_tracks --release -- check|bless
//!
//! The detector graph and tracker are deterministic, so `check` diffs exactly;
//! a changed shot is reported with its row-count and max-position deltas.

use std::fs;

use billiards_vision::detect::Image;
use billiards_vision::onnx::OnnxDetector;
use billiards_vision::track::{self, Orient, RawDet};

const MANIFEST: &str = "snapshots/tracks/manifest.txt";
const MODEL: &str = "python/detector.onnx";

fn load_png(path: &std::path::Path) -> Image {
    let img = image::open(path).expect("read frame").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image { width: w, height: h, pixels: img.pixels().map(|p| [p[0], p[1], p[2]]).collect() }
}

fn track_rows(det: &OnnxDetector, dir: &str, corners: [(f64, f64); 4], orient: Orient) -> String {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .expect("frames dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
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
    let tracks = track::track_clip(frames, corners, orient, 6.0).expect("degenerate corners");
    let shots = track::segment_shots(&tracks, track::FPS);

    let mut out = format!("# segments {shots:?}\n");
    let names = ["white", "yellow", "red"];
    for (ci, name) in names.iter().enumerate() {
        for (i, p) in tracks[ci].iter().enumerate() {
            if let Some((x, y)) = p {
                out += &format!("{name},{:.4},{x:.4},{y:.4}\n", i as f64 / track::FPS);
            }
        }
    }
    out
}

fn max_delta(a: &str, b: &str) -> (usize, usize, f64) {
    let parse = |s: &str| -> std::collections::HashMap<(String, String), (f64, f64)> {
        s.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                (f.len() == 4).then(|| {
                    ((f[0].to_string(), f[1].to_string()), (f[2].parse().unwrap_or(0.0), f[3].parse().unwrap_or(0.0)))
                })
            })
            .collect()
    };
    let (ma, mb) = (parse(a), parse(b));
    let common: Vec<_> = ma.keys().filter(|k| mb.contains_key(*k)).collect();
    let dmax = common
        .iter()
        .map(|k| {
            let (p, q) = (ma[*k], mb[*k]);
            ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
        })
        .fold(0.0f64, f64::max);
    (ma.len(), mb.len(), dmax)
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let manifest = fs::read_to_string(MANIFEST).expect("snapshots/tracks/manifest.txt");
    let det = OnnxDetector::from_file(MODEL).expect("load model");

    let mut regressions = 0usize;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // NAME DIR "u,v u,v u,v u,v" orient
        let Some((head, tail)) = line.split_once('"') else {
            eprintln!("bad manifest line (no quoted corners): {line}");
            continue;
        };
        let Some((corners_str, orient_str)) = tail.split_once('"') else {
            eprintln!("bad manifest line (unterminated corners): {line}");
            continue;
        };
        let mut hw = head.split_whitespace();
        let (Some(name), Some(dir)) = (hw.next(), hw.next()) else {
            eprintln!("bad manifest line (need NAME DIR): {line}");
            continue;
        };
        let corners: Vec<(f64, f64)> = corners_str
            .split_whitespace()
            .filter_map(|p| {
                let (u, v) = p.split_once(',')?;
                Some((u.parse().ok()?, v.parse().ok()?))
            })
            .collect();
        let corners: [(f64, f64); 4] = corners.try_into().expect("4 corners");
        let orient = if orient_str.trim() == "horizontal" { Orient::Horizontal } else { Orient::Vertical };

        let t0 = std::time::Instant::now();
        let current = track_rows(&det, dir, corners, orient);
        let secs = t0.elapsed().as_secs_f64();
        let path = format!("snapshots/tracks/{name}.rows");
        match mode.as_str() {
            "bless" => {
                fs::create_dir_all("snapshots/tracks").expect("mkdir");
                fs::write(&path, &current).expect("write");
                println!("blessed {name} ({} rows, {secs:.0}s)", current.lines().count() - 1);
            }
            _ => match fs::read_to_string(&path) {
                Err(_) => {
                    println!("{name}: NO APPROVED SNAPSHOT — bless first");
                    regressions += 1;
                }
                Ok(approved) if approved == current => {
                    println!("{name}: OK ({} rows, {secs:.0}s)", current.lines().count() - 1);
                }
                Ok(approved) => {
                    let (na, nc, dmax) = max_delta(&approved, &current);
                    println!("{name}: CHANGED — rows {na} -> {nc}, max common-row delta {:.1} mm", dmax * 1000.0);
                    regressions += 1;
                }
            },
        }
    }
    if mode != "bless" && regressions > 0 {
        std::process::exit(1);
    }
}
