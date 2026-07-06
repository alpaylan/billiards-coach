//! `billiards track` — video/frames → tracked `.shot` files, no Python.
//!
//! The single-binary tracker: ONNX detector (`billiards-vision::onnx`) +
//! tracking core (`billiards-vision::track`), fed either by a directory of
//! PNG frames or by any video file via ffmpeg piping raw RGB frames (ffmpeg
//! and yt-dlp stay external tools, exactly as in the Python pipeline).
//!
//!   billiards track --model detector.onnx --corners "19,19 264,15 270,492 23,492" \
//!       --orient vertical --out-dir shots/ (--frames DIR | --video CLIP.mp4)
//!
//! Writes one `shot_NN.shot` per detected shot segment (skipping segments
//! without an at-rest start), in the same format as track.py.

use std::io::Read;
use std::process::{Command, Stdio};

use billiards_vision::detect::Image;
use billiards_vision::onnx::OnnxDetector;
use billiards_vision::track::{self, Orient, RawDet};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

pub fn run(args: &[String]) {
    let usage = "usage: billiards track --model M.onnx --corners \"u,v u,v u,v u,v\" \
                 [--orient vertical|horizontal] --out-dir DIR (--frames DIR | --video FILE) [--fps N]";
    let (Some(model), Some(corners_str), Some(out_dir)) =
        (arg(args, "--model"), arg(args, "--corners"), arg(args, "--out-dir"))
    else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let orient = match arg(args, "--orient").as_deref() {
        Some("horizontal") => Orient::Horizontal,
        _ => Orient::Vertical,
    };
    let fps: f64 = arg(args, "--fps").and_then(|s| s.parse().ok()).unwrap_or(track::FPS);
    let corners: Vec<(f64, f64)> = corners_str
        .split_whitespace()
        .filter_map(|p| {
            let (u, v) = p.split_once(',')?;
            Some((u.parse().ok()?, v.parse().ok()?))
        })
        .collect();
    let corners: [(f64, f64); 4] = corners.try_into().unwrap_or_else(|_| {
        eprintln!("--corners needs 4 comma pairs");
        std::process::exit(2);
    });

    let det = OnnxDetector::from_file(&model).expect("load ONNX model");
    let t0 = std::time::Instant::now();
    let mut n_frames = 0usize;
    let mut detect = |img: Image| -> (Image, Vec<RawDet>) {
        n_frames += 1;
        let dets = det
            .detect_scored(&img)
            .expect("inference")
            .into_iter()
            .map(|d| RawDet { u: d.u, v: d.v, w: d.w, h: d.h, color: d.color, score: d.score })
            .collect();
        (img, dets)
    };

    let (frames, frames_dir): (Vec<(Image, Vec<RawDet>)>, Option<String>) =
        if let Some(dir) = arg(args, "--frames") {
            let mut paths: Vec<_> = std::fs::read_dir(&dir)
                .expect("frames dir")
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "png"))
                .collect();
            paths.sort();
            let abs = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone().into());
            (paths.iter().map(|p| detect(load_png(p))).collect(), Some(abs.display().to_string()))
        } else if let Some(video) = arg(args, "--video") {
            (ffmpeg_frames(&video).map(&mut detect).collect(), None)
        } else {
            eprintln!("{usage}");
            std::process::exit(2);
        };
    eprintln!("{n_frames} frames detected in {:.1}s", t0.elapsed().as_secs_f64());

    let tracks = track::track_clip(frames, corners, orient, 6.0).expect("degenerate corners");
    let shots = track::segment_shots(&tracks, fps);
    eprintln!("{} motion segments", shots.len());

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let mut written = 0;
    for (k, &(a, b)) in shots.iter().enumerate() {
        let (o, s, clean) = track::rest_bounds(&tracks, a, b, fps);
        if !clean {
            eprintln!("  skip segment {k}: no at-rest start (began mid-shot)");
            continue;
        }
        let hdr = track::ShotHeader {
            frames_dir: frames_dir.as_deref(),
            corners: &corners_str,
            orient,
            video_t0: None,
        };
        let text = track::write_shot(&tracks, o, s, fps, &hdr);
        let path = format!("{}/shot_{k:02}.shot", out_dir.trim_end_matches('/'));
        std::fs::write(&path, text).expect("write shot");
        eprintln!("  wrote {path} ({:.2}-{:.2}s)", o as f64 / fps, s as f64 / fps);
        written += 1;
    }
    eprintln!("{written} shots -> {out_dir}");
}

fn load_png(path: &std::path::Path) -> Image {
    let img = image::open(path).expect("read frame").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image { width: w, height: h, pixels: img.pixels().map(|p| [p[0], p[1], p[2]]).collect() }
}

/// Track one extracted clip directory and serialize its BUSIEST shot (the
/// match pipeline's per-clip path — one stroke per clock window). Returns
/// `None` when there is no clean at-rest start to fit from.
pub fn track_dir(
    det: &OnnxDetector,
    dir: &str,
    corners_str: &str,
    orient: Orient,
    fps: f64,
    video_t0: Option<f64>,
) -> Option<String> {
    let corners: Vec<(f64, f64)> = corners_str
        .split_whitespace()
        .filter_map(|p| {
            let (u, v) = p.split_once(',')?;
            Some((u.parse().ok()?, v.parse().ok()?))
        })
        .collect();
    let corners: [(f64, f64); 4] = corners.try_into().ok()?;

    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .ok()?
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
    let tracks = track::track_clip(frames, corners, orient, 6.0)?;
    let shots = track::segment_shots(&tracks, fps);
    let k = track::busiest_shot(&tracks, &shots)?;
    let (a, b) = shots[k];
    let (o, s, clean) = track::rest_bounds(&tracks, a, b, fps);
    if !clean {
        return None;
    }
    let abs = std::fs::canonicalize(dir).map(|p| p.display().to_string()).unwrap_or_else(|_| dir.into());
    let hdr = track::ShotHeader {
        frames_dir: Some(&abs),
        corners: corners_str,
        orient,
        video_t0,
    };
    Some(track::write_shot(&tracks, o, s, fps, &hdr))
}

/// Stream a video's frames as RGB images via ffmpeg (`-f rawvideo`).
fn ffmpeg_frames(video: &str) -> impl Iterator<Item = Image> + use<> {
    // Probe dimensions first — the raw stream is headerless.
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
               "-of", "csv=p=0", video])
        .output()
        .expect("ffprobe (is ffmpeg installed?)");
    let dims = String::from_utf8_lossy(&probe.stdout);
    let (w, h) = dims
        .trim()
        .split_once(',')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?)))
        .expect("ffprobe dimensions");

    let child = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i", video, "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    let mut out = child.stdout.expect("ffmpeg stdout");
    let frame_bytes = w * h * 3;

    std::iter::from_fn(move || {
        let mut buf = vec![0u8; frame_bytes];
        let mut filled = 0;
        while filled < frame_bytes {
            match out.read(&mut buf[filled..]) {
                Ok(0) => return None, // stream ended
                Ok(n) => filled += n,
                Err(_) => return None,
            }
        }
        let pixels = buf.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        Some(Image { width: w, height: h, pixels })
    })
}
