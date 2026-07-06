//! `billiards track --match VIDEO` — whole-game segmentation + tracking.
//!
//! The Rust port of `detect_shots.py` + the tracking loop of `build_match.py`:
//! one pass over the full-frame video reads the broadcast's green shot-clock
//! ring (resets once per shot — no OCR) and the overhead-inset motion; each
//! clock window is bounded rest-to-rest around its biggest motion burst; then
//! every shot's inset frames are extracted (ffmpeg crop) and tracked into a
//! `shot_NN.shot` with the frames back-link and `video_t0`, ready for the
//! editor / fit / bundle.
//!
//!   billiards track --match GAME.mp4 --model detector.onnx \
//!       --out-root data --name mygame [--preset 1080p] [--t0 S] [--t1 S]
//!
//! Not ported (still Python): montage sheet, scoreboard make/miss annotation.

use std::io::Read;
use std::process::{Command, Stdio};

use billiards_vision::onnx::OnnxDetector;
use billiards_vision::track::{rgb_to_hsv, Orient};

use crate::track_cmd;

/// Broadcast geometry per stream resolution (build_match.py's PRESETS).
/// `inset` = extraction crop (x,y,w,h); `interior` = playing surface in
/// full-frame coords (x0,x1,y0,y1) for motion; `corners` in inset-crop coords.
struct Preset {
    inset: (usize, usize, usize, usize),
    interior: (usize, usize, usize, usize),
    corners: &'static str,
}

fn preset(name: &str) -> Preset {
    match name {
        "720p" => Preset {
            inset: (15, 270, 185, 333),
            interior: (25, 190, 285, 595),
            corners: "7,12 172,10 174,327 9,327",
        },
        _ => Preset {
            inset: (15, 405, 290, 515),
            interior: (50, 270, 440, 880),
            corners: "19,19 264,15 270,492 23,492",
        },
    }
}

// Shot-clock overlay geometry at the 720p base (scoreboard.py); scales with
// frame height.
const CLOCK_RING: (f64, f64, f64, f64) = (606.0, 535.0, 74.0, 62.0);
const GREEN: ((u8, u8, u8), (u8, u8, u8)) = ((38, 70, 70), (88, 255, 255));
const FULL_RING_PX: f64 = 1500.0;
const BASE_H: f64 = 720.0;

// Rest-to-rest hysteresis on per-frame inset motion (detect_shots.py).
const MOTION_HI: u32 = 55;
const MOTION_LO: u32 = 10;
const REST_S: f64 = 0.5;
const PAD_PRE: f64 = 0.5;
const PAD_POST: f64 = 0.6;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ffprobe(video: &str) -> (usize, usize, f64) {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", video])
        .output()
        .expect("ffprobe (is ffmpeg installed?)");
    let s = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = s.trim().split(',').collect();
    let (w, h) = (f[0].parse().expect("width"), f[1].parse().expect("height"));
    let fps = f[2]
        .split_once('/')
        .map(|(n, d)| n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap().max(1.0))
        .unwrap_or(30.0);
    (w, h, fps)
}

struct Shot {
    onset: f64,  // seconds, absolute video time
    settle: f64, // seconds, absolute video time
}

/// Pass 1: stream the full frames once; per-frame interior motion + the clock
/// fraction at ~5 Hz. Returns (fps, per-frame motion, clock series (t, frac)).
fn analyze(video: &str, t0: f64, t1: Option<f64>) -> (f64, Vec<u32>, Vec<(f64, f64)>) {
    let (w, h, fps) = ffprobe(video);
    // The preset's inset/interior boxes are ALREADY in this resolution's absolute
    // pixels (build_match.py's PRESETS); only the scoreboard overlay geometry is
    // specified at the 720p base and scales with frame height (scoreboard.py).
    let s = h as f64 / BASE_H;
    let scale = |v: f64| (v * s).round() as usize;
    let p = preset(if h >= 1000 { "1080p" } else { "720p" });
    let (ix0, ix1, iy0, iy1) = p.interior;
    let (cx, cy, cw, ch) = CLOCK_RING;
    let (cx, cy, cw, ch) = (scale(cx), scale(cy), scale(cw), scale(ch));
    let step = (fps / 5.0).round().max(1.0) as usize;

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error"]);
    if t0 > 0.0 {
        cmd.args(["-ss", &format!("{t0}")]);
    }
    cmd.args(["-i", video]);
    if let Some(t1) = t1 {
        cmd.args(["-t", &format!("{}", t1 - t0)]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout(Stdio::piped());
    let child = cmd.spawn().expect("spawn ffmpeg");
    let mut out = child.stdout.expect("ffmpeg stdout");

    let frame_bytes = w * h * 3;
    let mut buf = vec![0u8; frame_bytes];
    let (gw, gh) = (ix1 - ix0, iy1 - iy0);
    let mut prev: Vec<i16> = Vec::new();
    let mut fine: Vec<u32> = Vec::new();
    let mut clock: Vec<(f64, f64)> = Vec::new();
    let mut i = 0usize;
    loop {
        let mut filled = 0;
        while filled < frame_bytes {
            match out.read(&mut buf[filled..]) {
                Ok(0) if filled == 0 => return (fps, fine, clock),
                Ok(0) => return (fps, fine, clock), // truncated tail frame
                Ok(n) => filled += n,
                Err(_) => return (fps, fine, clock),
            }
        }
        // interior gray + diff count (cv2 gray weights, |Δ| > 28)
        let mut gray = Vec::with_capacity(gw * gh);
        for y in iy0..iy1 {
            let row = (y * w + ix0) * 3;
            for x in 0..gw {
                let p = &buf[row + x * 3..row + x * 3 + 3];
                let g = 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64;
                gray.push(g.round() as i16);
            }
        }
        let m = if prev.is_empty() {
            0
        } else {
            gray.iter().zip(&prev).filter(|(a, b)| (**a - **b).abs() > 28).count() as u32
        };
        fine.push(m);
        prev = gray;

        if i % step == 0 {
            // green pixel count inside the clock-ring box
            let mut px = 0usize;
            for y in cy..(cy + ch).min(h) {
                let row = (y * w + cx) * 3;
                for x in 0..cw.min(w - cx) {
                    let pnt = &buf[row + x * 3..row + x * 3 + 3];
                    let (hh, ss, vv) = rgb_to_hsv([pnt[0], pnt[1], pnt[2]]);
                    let ((h0, s0, v0), (h1, s1, v1)) = GREEN;
                    if hh >= h0 && hh <= h1 && ss >= s0 && ss <= s1 && vv >= v0 && vv <= v1 {
                        px += 1;
                    }
                }
            }
            clock.push((i as f64 / fps, (px as f64 / (FULL_RING_PX * s * s)).min(1.0)));
        }
        i += 1;
    }
}

/// Rest-to-rest span of the main stroke inside frames [a, b) — `_stroke_span`.
fn stroke_span(fine: &[u32], fps: f64, a: usize, b: usize) -> Option<(usize, usize)> {
    let seg = &fine[a..b.min(fine.len())];
    if seg.is_empty() {
        return None;
    }
    let rest = (REST_S * fps) as usize;
    let mut eps: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < seg.len() {
        if seg[i] > MOTION_HI {
            let (mut j, mut gap) = (i, 0);
            while j < seg.len() && gap < rest {
                gap = if seg[j] <= MOTION_LO { gap + 1 } else { 0 };
                j += 1;
            }
            eps.push((i, j - gap));
            i = j;
        } else {
            i += 1;
        }
    }
    let &(s, e) = eps.iter().max_by_key(|&&(s, e)| seg[s..e.max(s + 1)].iter().sum::<u32>())?;
    let mut o = s;
    while o > 0 && seg[o] > MOTION_LO {
        o -= 1;
    }
    let (mut st, mut run) = (e, 0);
    while st < seg.len() && run < rest {
        run = if seg[st] <= MOTION_LO { run + 1 } else { 0 };
        st += 1;
    }
    Some((a + o, a + st))
}

/// Shot windows from clock resets, bounded rest-to-rest — `find_shots`.
fn find_shots(fps: f64, clock: &[(f64, f64)], fine: &[u32]) -> Vec<Shot> {
    let resets: Vec<usize> = (1..clock.len())
        .filter(|&k| clock[k].1 - clock[k - 1].1 > 0.25 && clock[k].1 > 0.5)
        .map(|k| (clock[k].0 * fps) as usize)
        .collect();
    let spill = (1.5 * fps) as usize;
    let mut shots = Vec::new();
    for (i, &a) in resets.iter().enumerate() {
        let b = resets.get(i + 1).copied().unwrap_or(fine.len());
        if let Some((onset, settle)) = stroke_span(fine, fps, a, (b + spill).min(fine.len())) {
            shots.push(Shot { onset: onset as f64 / fps, settle: settle as f64 / fps });
        }
    }
    shots
}

pub fn run(args: &[String]) {
    let usage = "usage: billiards track --match VIDEO --model M.onnx --out-root DIR --name NAME \
                 [--preset 720p|1080p] [--orient vertical|horizontal] [--t0 S] [--t1 S]";
    let (Some(video), Some(model), Some(out_root), Some(name)) =
        (arg(args, "--match"), arg(args, "--model"), arg(args, "--out-root"), arg(args, "--name"))
    else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let t0: f64 = arg(args, "--t0").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let t1: Option<f64> = arg(args, "--t1").and_then(|s| s.parse().ok());
    let orient = match arg(args, "--orient").as_deref() {
        Some("horizontal") => Orient::Horizontal,
        _ => Orient::Vertical,
    };
    let (_, h, _) = ffprobe(&video);
    let pname = arg(args, "--preset").unwrap_or(if h >= 1000 { "1080p".into() } else { "720p".into() });
    let p = preset(&pname);
    let s = h as f64 / BASE_H;

    eprintln!("[1/3] scanning {video} for shot-clock resets + strokes ({pname}, scale {s:.2})…");
    let (fps, fine, clock) = analyze(&video, t0, t1);
    let mut shots = find_shots(fps, &clock, &fine);
    for sh in &mut shots {
        sh.onset += t0;
        sh.settle += t0;
    }
    eprintln!("      {} shots over {:.0}s of footage", shots.len(), fine.len() as f64 / fps);

    let det = OnnxDetector::from_file(&model).expect("load ONNX model");
    let frames_root = format!("{}/{}_frames", out_root.trim_end_matches('/'), name);
    let shots_dir = format!("{}/{}", out_root.trim_end_matches('/'), name);
    std::fs::create_dir_all(&shots_dir).expect("create out dir");

    eprintln!("[2/3] extracting + tracking {} shots…", shots.len());
    // inset crop is already in this resolution's absolute pixels
    let (ins_x, ins_y, ins_w, ins_h) = p.inset;
    let mut ok = 0usize;
    for (k, sh) in shots.iter().enumerate() {
        let dir = format!("{frames_root}/shot_{k:02}");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("frames dir");
        let clip_t0 = (sh.onset - PAD_PRE).max(0.0);
        let clip_end = (sh.settle + PAD_POST).max(sh.onset + 11.5);
        let status = Command::new("ffmpeg")
            .args(["-nostdin", "-loglevel", "error", "-ss", &format!("{clip_t0}"),
                   "-i", &video, "-t", &format!("{}", clip_end - clip_t0),
                   "-vf", &format!("crop={ins_w}:{ins_h}:{ins_x}:{ins_y}"),
                   &format!("{dir}/f_%04d.png"), "-y"])
            .status()
            .expect("ffmpeg extract");
        if !status.success() {
            eprintln!("      shot_{k:02}: ffmpeg extract failed");
            continue;
        }
        match track_cmd::track_dir(&det, &dir, p.corners, orient, fps, Some(clip_t0)) {
            Some(text) => {
                let path = format!("{shots_dir}/shot_{k:02}.shot");
                std::fs::write(&path, text).expect("write shot");
                eprintln!("      shot_{k:02}: ok ({:.0}s)", sh.onset);
                ok += 1;
            }
            None => eprintln!("      shot_{k:02}: no clean stroke"),
        }
    }
    eprintln!("[3/3] done — {ok}/{} shots tracked -> {shots_dir}/", shots.len());
    eprintln!("      (make/miss annotation + montage remain in Python: annotate_results.py)");
}
