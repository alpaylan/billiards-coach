//! Ball tracking: per-frame scored detections → per-color table-coordinate
//! tracks → motion-based shot segments. The Rust port of `python/track.py`'s
//! tracking core (the learned-detector path), consuming any detector that
//! yields scored candidates (see `onnx::OnnxDetector::detect_scored`).
//!
//! Port notes (deliberate divergences from track.py):
//! - The white/yellow continuity-swap check uses the REPAIRED thresholds
//!   (0.055 / 0.35) from `fix_swaps.py`, not the original 0.16 / 0.4 — the old
//!   floor verifiably let through exchanges that happen while the balls are
//!   close (combined jump ≈ their separation).
//! - `mask_pad` here is a signed DISTANCE in pixels; track.py passes a cv2
//!   morphology kernel SIZE (12 → ~6 px of dilation). Callers convert.

use billiards_core::BallColor;

use crate::detect::Image;
use crate::homography::Homography;

pub const TABLE_L: f64 = 2.84;
pub const TABLE_W: f64 = 1.42;
pub const FPS: f64 = 30.0;
pub const COLORS: [BallColor; 3] = [BallColor::White, BallColor::Yellow, BallColor::Red];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orient {
    /// The table's long (2.84 m) axis runs left–right in the image.
    Horizontal,
    /// The top image edge is a short rail (overhead inset).
    Vertical,
}

/// One scored candidate from the detector, in source-image pixels.
#[derive(Clone, Copy, Debug)]
pub struct RawDet {
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub h: f64,
    pub color: BallColor,
    pub score: f32,
}

/// Table corner targets (meters) for image corners given clockwise from
/// top-left — mirror of track.py's `table_targets`.
fn table_targets(orient: Orient) -> [(f64, f64); 4] {
    let (hl, hw) = (TABLE_L / 2.0, TABLE_W / 2.0);
    match orient {
        Orient::Horizontal => [(-hl, hw), (hl, hw), (hl, -hw), (-hl, -hw)],
        Orient::Vertical => [(-hl, hw), (-hl, -hw), (hl, -hw), (hl, hw)],
    }
}

/// Image→table homography from the 4 clicked corners.
pub fn calibrate(image_corners: [(f64, f64); 4], orient: Orient) -> Option<Homography> {
    Homography::from_correspondences(image_corners, table_targets(orient))
}

/// The table interior as a convex quad, padded outward by `pad` pixels
/// (positive = dilate, to keep balls half over the nose; negative = erode).
pub struct TableMask {
    corners: [(f64, f64); 4],
    pad: f64,
    /// +1 or -1: which cross-product sign means "inside" for this winding.
    inward: f64,
}

impl TableMask {
    pub fn new(corners: [(f64, f64); 4], pad: f64) -> Self {
        // Signed area decides the winding (image y grows downward).
        let mut area2 = 0.0;
        for i in 0..4 {
            let (x0, y0) = corners[i];
            let (x1, y1) = corners[(i + 1) % 4];
            area2 += x0 * y1 - x1 * y0;
        }
        Self { corners, pad, inward: area2.signum() }
    }

    pub fn contains(&self, u: f64, v: f64) -> bool {
        for i in 0..4 {
            let (x0, y0) = self.corners[i];
            let (x1, y1) = self.corners[(i + 1) % 4];
            let (ex, ey) = (x1 - x0, y1 - y0);
            let len = (ex * ex + ey * ey).sqrt().max(1e-9);
            // signed distance of (u,v) left/right of this edge, made "inside-positive"
            let d = self.inward * (ex * (v - y0) - ey * (u - x0)) / len;
            if d < -self.pad {
                return false;
            }
        }
        true
    }
}

/// Reduce one frame's raw detections to at most one image point per color —
/// track.py's `make_learned_detector` inner logic: gate, merge into blobs,
/// assign colors to blobs JOINTLY (exclusivity recovers a ball whose color the
/// CNN misjudged, because the other blobs are strongly claimed), and fall back
/// to a ball-shaped classical HSV pick for colors the CNN missed entirely
/// (motion-blurred fast balls and corner-tucked ones are outside its training
/// set — and the fast ball is precisely the stroke being tracked).
pub fn assign_frame(image: &Image, dets: &[RawDet], mask: &TableMask) -> [Option<(f64, f64)>; 3] {
    let frame_w = image.width as f64;
    // blobs: (position, per-color best score)
    let mut blobs: Vec<((f64, f64), [f32; 3])> = Vec::new();
    for d in dets {
        if d.score < 0.04 {
            continue;
        }
        if d.w.max(d.h) > 0.09 * frame_w {
            continue; // a ball is small; a player's arm/shirt is not
        }
        if !mask.contains(d.u, d.v) {
            continue; // off the table (rail/background false positive)
        }
        let ci = COLORS.iter().position(|&c| c == d.color).unwrap();
        let mut merged = false;
        for (pos, scores) in blobs.iter_mut() {
            if (pos.0 - d.u).hypot(pos.1 - d.v) < 9.0 {
                scores[ci] = scores[ci].max(d.score);
                merged = true;
                break;
            }
        }
        if !merged {
            blobs.push(((d.u, d.v), {
                let mut s = [0.0; 3];
                s[ci] = d.score;
                s
            }));
        }
    }
    blobs.truncate(8); // plenty; keeps the assignment tiny

    // Joint assignment: each color takes a distinct blob or none; maximize
    // total score. Blob counts are tiny, so brute force is exact and cheap.
    let n = blobs.len();
    let mut best: Option<[Option<usize>; 3]> = None;
    let mut best_total = -1.0f32;
    for pw in 0..=n {
        for py in 0..=n {
            for pr in 0..=n {
                let pick = [pw, py, pr].map(|p| if p == n { None } else { Some(p) });
                let mut used = [false; 8];
                let mut total = 0.0;
                let mut ok = true;
                for (ci, p) in pick.iter().enumerate() {
                    let Some(b) = *p else { continue };
                    if used[b] || blobs[b].1[ci] <= 0.0 {
                        ok = false;
                        break;
                    }
                    used[b] = true;
                    total += blobs[b].1[ci];
                }
                if ok && total > best_total {
                    best_total = total;
                    best = Some(pick);
                }
            }
        }
    }

    let mut out = [None; 3];
    if let Some(pick) = best {
        for (ci, p) in pick.iter().enumerate() {
            if let Some(b) = *p {
                // exclusivity may hand a color a weakly-scored blob, but never
                // a junk one: 0.12 floor
                if blobs[b].1[ci] >= 0.12 {
                    out[ci] = Some(blobs[b].0);
                }
            }
        }
    }
    // Hybrid fallback: colors the CNN missed go to a BALL-SHAPED classical
    // pick, provided it isn't one of the already-claimed blobs.
    if out.iter().any(Option::is_none) {
        let mut taken: Vec<(f64, f64)> = out.iter().flatten().copied().collect();
        for ci in 0..3 {
            if out[ci].is_none() {
                if let Some(p) = classical_ball(image, mask, COLORS[ci]) {
                    if taken.iter().all(|q| (p.0 - q.0).hypot(p.1 - q.1) > 9.0) {
                        out[ci] = Some(p);
                        taken.push(p);
                    }
                }
            }
        }
    }
    out
}

/// HSV ranges per ball color, in OpenCV convention (H 0..180, S/V 0..255) —
/// mirror of track.py's `HSV_RANGES`.
fn hsv_ranges(color: BallColor) -> &'static [((u8, u8, u8), (u8, u8, u8))] {
    match color {
        BallColor::Red => &[((0, 120, 90), (10, 255, 255)), ((170, 120, 90), (179, 255, 255))],
        BallColor::Yellow => &[((18, 70, 130), (35, 255, 255))],
        BallColor::White => &[((0, 0, 180), (179, 60, 255))],
    }
}

/// RGB → OpenCV-convention HSV (H 0..180, S/V 0..255).
pub fn rgb_to_hsv(p: [u8; 3]) -> (u8, u8, u8) {
    let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
    let v = r.max(g).max(b);
    let min = r.min(g).min(b);
    let s = if v > 0.0 { 255.0 * (v - min) / v } else { 0.0 };
    let h = if v == min {
        0.0
    } else if v == r {
        60.0 * (g - b) / (v - min)
    } else if v == g {
        120.0 + 60.0 * (b - r) / (v - min)
    } else {
        240.0 + 60.0 * (r - g) / (v - min)
    };
    let h = if h < 0.0 { h + 360.0 } else { h } / 2.0;
    (h.round() as u8, s.round().min(255.0) as u8, v.round() as u8)
}

/// The most BALL-LIKE blob of `color`: compact, round, and nearest the
/// expected ball area for this frame scale — mirror of track.py's
/// `_classical_ball` (HSV threshold → 3×3 morphological open → connected
/// components → shape gates → closest-to-expected-area pick).
pub fn classical_ball(image: &Image, mask: &TableMask, color: BallColor) -> Option<(f64, f64)> {
    let (w, h) = (image.width, image.height);
    let ranges = hsv_ranges(color);
    let mut bin = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !mask.contains(x as f64, y as f64) {
                continue;
            }
            let (hh, ss, vv) = rgb_to_hsv(image.get(x, y));
            if ranges.iter().any(|&((h0, s0, v0), (h1, s1, v1))| {
                hh >= h0 && hh <= h1 && ss >= s0 && ss <= s1 && vv >= v0 && vv <= v1
            }) {
                bin[y * w + x] = true;
            }
        }
    }
    morph_open3(&mut bin, w, h);

    // Connected components (4-connectivity, like cv2's default).
    let exp_area = std::f64::consts::PI * (0.0185 * w as f64).powi(2);
    let mut label = vec![0u32; w * h];
    let mut next = 0u32;
    let mut best: Option<(f64, f64)> = None;
    let mut best_ratio = 4.0;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !bin[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        stack.push(start);
        label[start] = next;
        let (mut area, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
        let (mut x0, mut x1, mut y0, mut y1) = (w, 0usize, h, 0usize);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            area += 1;
            sx += x as f64;
            sy += y as f64;
            (x0, x1) = (x0.min(x), x1.max(x));
            (y0, y1) = (y0.min(y), y1.max(y));
            for n in [i.wrapping_sub(1), i + 1, i.wrapping_sub(w), i + w] {
                let valid = n < w * h
                    && bin[n]
                    && label[n] == 0
                    && (n % w).abs_diff(x) + (n / w).abs_diff(y) == 1;
                if valid {
                    label[n] = next;
                    stack.push(n);
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f64, (y1 - y0 + 1) as f64);
        let a = area as f64;
        if a < 0.25 * exp_area || a > 4.0 * exp_area {
            continue;
        }
        if bw.max(bh) / bw.min(bh).max(1.0) > 1.8 || a < 0.5 * bw * bh {
            continue; // elongated or sparse — an arm or cue, not a ball
        }
        let ratio = a.max(exp_area) / a.min(exp_area).max(1.0);
        if ratio < best_ratio {
            best_ratio = ratio;
            best = Some((sx / a, sy / a));
        }
    }
    best
}

/// In-place 3×3 binary morphological open (erode then dilate).
fn morph_open3(bin: &mut [bool], w: usize, h: usize) {
    let src = bin.to_vec();
    let full3 = |buf: &[bool], x: usize, y: usize| -> bool {
        if x == 0 || y == 0 || x + 1 >= w || y + 1 >= h {
            return false; // cv2 border: erosion clears the frame edge
        }
        (y - 1..=y + 1).all(|yy| (x - 1..=x + 1).all(|xx| buf[yy * w + xx]))
    };
    let mut eroded = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            eroded[y * w + x] = full3(&src, x, y);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let any = (y.saturating_sub(1)..=(y + 1).min(h - 1))
                .any(|yy| (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| eroded[yy * w + xx]));
            bin[y * w + x] = any;
        }
    }
}

/// Track a clip: per-frame detections (already reduced or raw) → per-color
/// table-coordinate tracks with short gaps interpolated. `frames` yields each
/// frame's raw detections in source pixels.
pub fn track_clip(
    frames: impl IntoIterator<Item = (Image, Vec<RawDet>)>,
    image_corners: [(f64, f64); 4],
    orient: Orient,
    mask_pad: f64,
) -> Option<[Vec<Option<(f64, f64)>>; 3]> {
    let h = calibrate(image_corners, orient)?;
    let mask = TableMask::new(image_corners, mask_pad);
    let d = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);

    // A ball can't teleport, but it keeps moving while undetected: the jump
    // allowance grows with the gap since the last good fix, capped so a false
    // blob across the table is still rejected.
    const MAX_JUMP: f64 = 0.30; // m/frame (~9 m/s at 30 fps)
    const GAP_CAP: usize = 5;

    let mut raw: [Vec<Option<(f64, f64)>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut prev: [Option<(f64, f64)>; 3] = [None; 3];
    let mut gap: [usize; 3] = [0; 3];

    for (image, dets) in frames {
        let img_pts = assign_frame(&image, &dets, &mask);
        let mut cand: [Option<(f64, f64)>; 3] = [None; 3];
        for ci in 0..3 {
            cand[ci] = img_pts[ci].map(|(u, v)| h.apply(u, v));
        }

        // White/yellow are the confusable pair: if exchanging the labels turns
        // two large jumps into two small ones, they were swapped (checked
        // before gating so the teleport gates judge corrected identities).
        if let (Some(a), Some(b), Some(pa), Some(pb)) = (cand[0], cand[1], prev[0], prev[1]) {
            let keep = d(a, pa) + d(b, pb);
            let swap = d(b, pa) + d(a, pb);
            if keep > 0.055 && swap < 0.35 * keep {
                cand.swap(0, 1);
            }
        }

        for ci in 0..3 {
            if let (Some(p), Some(pv)) = (cand[ci], prev[ci]) {
                if d(p, pv) > MAX_JUMP * ((gap[ci] + 1).min(GAP_CAP)) as f64 {
                    cand[ci] = None; // implausible jump even accounting for the gap
                }
            }
        }

        // Two colors can't be the same blob: if two candidates coincide
        // (closer than a ball diameter), keep the one consistent with its own
        // motion and drop the impostor.
        for i in 0..3 {
            for j in i + 1..3 {
                if let (Some(a), Some(b)) = (cand[i], cand[j]) {
                    if d(a, b) < 0.055 {
                        let ja = prev[i].map_or(1e9, |p| d(a, p));
                        let jb = prev[j].map_or(1e9, |p| d(b, p));
                        cand[if ja > jb { i } else { j }] = None;
                    }
                }
            }
        }

        for ci in 0..3 {
            raw[ci].push(cand[ci]);
            match cand[ci] {
                Some(p) => {
                    prev[ci] = Some(p);
                    gap[ci] = 0;
                }
                None => gap[ci] += 1,
            }
        }
    }
    Some(raw.map(|t| fill_gaps(&t, 5)))
}

/// Linear-interpolate only SHORT missing runs (≤ `max_gap` frames). Longer
/// dropouts stay `None` — the ball went somewhere unseen, and inventing a
/// straight path there would be a lie. Leading/trailing gaps also stay `None`.
pub fn fill_gaps(track: &[Option<(f64, f64)>], max_gap: usize) -> Vec<Option<(f64, f64)>> {
    let mut out = track.to_vec();
    let known: Vec<usize> = track.iter().enumerate().filter_map(|(i, p)| p.map(|_| i)).collect();
    if known.is_empty() {
        return out;
    }
    for i in 0..track.len() {
        if out[i].is_some() {
            continue;
        }
        let lo = known.iter().rev().find(|&&k| k < i).copied();
        let hi = known.iter().find(|&&k| k > i).copied();
        let (Some(lo), Some(hi)) = (lo, hi) else { continue };
        if hi - lo - 1 > max_gap {
            continue;
        }
        let t = (i - lo) as f64 / (hi - lo) as f64;
        let (a, b) = (track[lo].unwrap(), track[hi].unwrap());
        out[i] = Some((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
    }
    out
}

/// Frame intervals `(start, end)` where any ball is moving — track.py's
/// `segment_shots` (speed threshold, stillness bridging, minimum length).
pub fn segment_shots(tracks: &[Vec<Option<(f64, f64)>>; 3], fps: f64) -> Vec<(usize, usize)> {
    const V_MOVE: f64 = 0.12; // m/s
    const MIN_LEN: f64 = 0.25; // s
    const BRIDGE: f64 = 0.3; // s of stillness bridged inside one shot

    let n = tracks[0].len();
    let mut speed = vec![0.0f64; n];
    for tr in tracks {
        for i in 1..n {
            if let (Some(a), Some(b)) = (tr[i - 1], tr[i]) {
                speed[i] = speed[i].max((a.0 - b.0).hypot(a.1 - b.1) * fps);
            }
        }
    }
    let moving: Vec<bool> = speed.iter().map(|&s| s > V_MOVE).collect();

    let mut shots = Vec::new();
    let mut i = 0;
    while i < n {
        if !moving[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut gap = 0.0;
        while j + 1 < n && (moving[j + 1] || gap < BRIDGE * fps) {
            j += 1;
            gap = if moving[j] { 0.0 } else { gap + 1.0 };
        }
        while j > i && !moving[j] {
            j -= 1; // trim trailing bridged stillness
        }
        if (j - i) as f64 / fps >= MIN_LEN {
            shots.push((i, j));
        }
        i = j + 1;
    }
    shots
}

/// Max per-frame ball speed (m/s) at each frame, over gap-FILLED tracks.
fn speeds(tracks: &[Vec<Option<(f64, f64)>>; 3], fps: f64) -> Vec<f64> {
    let n = tracks[0].len();
    let mut spd = vec![0.0f64; n];
    for tr in tracks {
        for i in 1..n {
            if let (Some(a), Some(b)) = (tr[i - 1], tr[i]) {
                spd[i] = spd[i].max((a.0 - b.0).hypot(a.1 - b.1) * fps);
            }
        }
    }
    spd
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// Extend a motion interval outward to the surrounding ball-rest frames —
/// track.py's `_rest_bounds`: back to the last still frame before the stroke
/// (t=0 must be the true starting layout) and forward to a median-still settle
/// (+0.15 s tail). `clean=false` means the recording began mid-shot: skip it.
pub fn rest_bounds(
    tracks: &[Vec<Option<(f64, f64)>>; 3],
    a: usize,
    b: usize,
    fps: f64,
) -> (usize, usize, bool) {
    const V_REST: f64 = 0.04;
    const HOLD: f64 = 0.3;
    let spd = speeds(tracks, fps);
    let n = spd.len();
    let w = ((HOLD * fps) as usize).max(1);

    let mut o = a;
    while o > 0 && spd[o] > V_REST {
        o -= 1;
    }
    let clean = o > 0 && median(&spd[o.saturating_sub(w)..=o]) <= V_REST;
    let mut s = b;
    while s < n - w && median(&spd[s..s + w]) >= V_REST {
        s += 1;
    }
    (o, (s + (0.15 * fps) as usize).min(n - 1), clean)
}

/// Total travel of one ball over [a, b].
fn travel(tr: &[Option<(f64, f64)>], a: usize, b: usize) -> f64 {
    (a..b)
        .filter_map(|i| match (tr[i], tr[i + 1]) {
            (Some(p), Some(q)) => Some((p.0 - q.0).hypot(p.1 - q.1)),
            _ => None,
        })
        .sum()
}

/// The segment with the most ball travel — track.py's default export pick.
pub fn busiest_shot(tracks: &[Vec<Option<(f64, f64)>>; 3], shots: &[(usize, usize)]) -> Option<usize> {
    (0..shots.len()).max_by(|&i, &j| {
        let t = |k: usize| {
            let (a, b) = shots[k];
            tracks.iter().map(|tr| travel(tr, a, b)).fold(0.0, f64::max)
        };
        t(i).total_cmp(&t(j))
    })
}

/// The cue is the player ball (white or yellow) STRUCK first: object balls
/// only move once the cue reaches them, so first motion identifies it.
pub fn shot_cue(tracks: &[Vec<Option<(f64, f64)>>; 3], a: usize, b: usize, fps: f64) -> BallColor {
    let first_move = |ci: usize| -> usize {
        let tr = &tracks[ci];
        for i in a..b {
            if let (Some(p), Some(q)) = (tr[i], tr[i + 1]) {
                if (p.0 - q.0).hypot(p.1 - q.1) * fps > 0.15 {
                    return i;
                }
            }
        }
        b + 1
    };
    if first_move(0) <= first_move(1) { BallColor::White } else { BallColor::Yellow }
}

/// Footage back-link headers for the editor's synced video panel.
pub struct ShotHeader<'a> {
    pub frames_dir: Option<&'a str>,
    pub corners: &'a str,
    pub orient: Orient,
    pub video_t0: Option<f64>,
}

/// Serialize one shot in the `.shot` format (track.py's `export_for_fit`):
/// `cue COLOR` + back-link headers + `color,t,x,y` rows, cue's color first,
/// t=0 at the rest onset.
pub fn write_shot(
    tracks: &[Vec<Option<(f64, f64)>>; 3],
    a: usize,
    b: usize,
    fps: f64,
    hdr: &ShotHeader,
) -> String {
    let cue = shot_cue(tracks, a, b, fps);
    let names = ["white", "yellow", "red"];
    let cue_i = COLORS.iter().position(|&c| c == cue).unwrap();
    let mut order = vec![cue_i];
    order.extend((0..3).filter(|&i| i != cue_i));

    let mut out = format!("cue {}\n", names[cue_i]);
    if let Some(dir) = hdr.frames_dir {
        out += &format!("frames {dir}\n");
        out += &format!("fps {fps}\n");
        out += &format!("start {a}\n");
        out += &format!("corners {}\n", hdr.corners);
        out += &format!(
            "orient {}\n",
            if hdr.orient == Orient::Horizontal { "horizontal" } else { "vertical" }
        );
    }
    if let Some(t0) = hdr.video_t0 {
        out += &format!("video_t0 {t0:.3}\n");
    }
    out += "color,t,x,y\n";
    for &ci in &order {
        for i in a..=b.min(tracks[ci].len() - 1) {
            if let Some((x, y)) = tracks[ci][i] {
                out += &format!("{},{:.4},{x:.4},{y:.4}\n", names[ci], (i - a) as f64 / fps);
            }
        }
    }
    out
}
