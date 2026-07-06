//! Billiards Coach — interactive table (Phase 1).
//!
//! A top-down 2D editor: drag the three balls, dial the cue action (aim / speed /
//! english), and watch the simulated trajectories. "Solve" runs the solver and
//! loads its recommended shot, with the success probability and difficulty. This
//! is the "as-if" scenario tool — every trajectory here comes from the same
//! engine the solver and (future) reconstruction use.

mod view;

use billiards_core::math::DVec3;
use billiards_core::{
    BallId, BallSpec, PhysicsParams, Scene, TableSpec, Trajectory, three_cushion_score,
};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::{Repair, RepairConfig, SolveConfig, Solution, repair, solve, success_probability};
use billiards_vision::Homography;
use eframe::egui;
use view::View;

/// Which central view is shown: the synthetic reconstruction, or the actual video
/// frame with the reconstructed ball positions overlaid (for verification).
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Reconstructed,
    Actual,
}

/// A scene imported from the video pipeline, with the source frame + calibration.
struct ImportedScene {
    scene: Scene,
    image_path: Option<String>,
    /// Table → image-pixel homography, to overlay reconstructed balls on the frame.
    tab2img: Option<Homography>,
}

const DIAMOND: f64 = 0.355;
const BALL_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(244, 244, 236), // cue (white)
    egui::Color32::from_rgb(240, 200, 48),  // object (yellow)
    egui::Color32::from_rgb(200, 54, 44),   // object (red)
];
fn main() -> eframe::Result {
    // Optional argument: a `.scene`/`.shot` file, OR a *directory* of `.shot`
    // files — a whole tracked match to browse (from the MASA 4 pipeline).
    let arg = std::env::args().nth(1);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 680.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Billiards Coach",
        options,
        Box::new(move |_cc| {
            let mut app = App::new();
            if let Some(p) = &arg {
                if std::path::Path::new(p).is_dir() {
                    app.open_dir(p);
                } else {
                    app.try_load_scene(p);
                }
            }
            Ok(Box::new(app))
        }),
    )
}

/// One shot in a loaded match: its file, the cue color, and — when the scoreboard
/// annotator has run — which player took it and whether it scored (for the browser).
struct MatchShot {
    path: std::path::PathBuf,
    cue_color: egui::Color32,
    player: Option<String>, // "left" / "right"
    result: Option<String>, // "make" / "miss"
}

/// One game in a multi-game match manifest (`match.json` from `pipeline.py`).
struct GameEntry {
    dir: String,
    left: String,
    right: String,
    n_shots: i64,
    n_made: Option<i64>,
}

/// Extract a `"key":"string"` value from a flat JSON fragment.
fn json_str(obj: &str, key: &str) -> Option<String> {
    let i = obj.find(&format!("\"{key}\""))?;
    let after = &obj[i + key.len() + 2..];
    let after = &after[after.find(':')? + 1..];
    let q = after.find('"')?;
    let rest = &after[q + 1..];
    Some(rest[..rest.find('"')?].to_string())
}

/// Extract a `"key": number` value from a flat JSON fragment.
fn json_int(obj: &str, key: &str) -> Option<i64> {
    let i = obj.find(&format!("\"{key}\""))?;
    let after = &obj[i + key.len() + 2..];
    let after = &after[after.find(':')? + 1..];
    after.chars().skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>().parse().ok()
}

/// Parse the game list out of a `match.json` string. Hand-rolled (the manifest is
/// our own flat structure) — each game object is split on braces, safe as they
/// don't nest.
fn parse_match_games(text: &str) -> Vec<GameEntry> {
    let Some(gi) = text.find("\"games\"") else { return Vec::new() };
    text[gi..].split('{').skip(1).filter_map(|obj| {
        Some(GameEntry {
            dir: json_str(obj, "dir")?,
            left: json_str(obj, "left").unwrap_or_default(),
            right: json_str(obj, "right").unwrap_or_default(),
            n_shots: json_int(obj, "n_shots").unwrap_or(0),
            n_made: json_int(obj, "n_made"),
        })
    }).collect()
}

/// Parse `<dir>/match.json`'s game list (empty if the manifest is absent).
fn parse_match_json(dir: &str) -> Vec<GameEntry> {
    let path = format!("{}/match.json", dir.trim_end_matches('/'));
    std::fs::read_to_string(path).map(|t| parse_match_games(&t)).unwrap_or_default()
}

/// Pull a `key value` header field out of a `.shot` file's text.
fn header_field(text: &str, key: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.trim().strip_prefix(key).filter(|r| r.starts_with(' ')))
        .map(|r| r.trim().to_string())
}

/// Per-game physics from `<dir>/calibration.json` (written by `calibrate_match`),
/// so each table/cloth uses its own fitted parameters. Falls back to the built-in
/// calibration when the file is absent. Hand-rolled parse — the file is our own
/// flat object of `"key": number` pairs, so no JSON dependency is needed.
fn load_calibration(dir: &str) -> Option<PhysicsParams> {
    let path = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let text = std::fs::read_to_string(path).ok()?;
    let get = |key: &str| -> Option<f64> {
        let i = text.find(&format!("\"{key}\""))?;
        let after = &text[i + key.len() + 2..];
        let after = &after[after.find(':')? + 1..];
        let num: String = after
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect();
        num.parse().ok()
    };
    Some(PhysicsParams {
        cushion_restitution: get("cushion_restitution")?,
        cushion_friction: get("cushion_friction")?,
        mu_slide: get("mu_slide")?,
        mu_roll: get("mu_roll")?,
        ..PhysicsParams::carom_calibrated()
    })
}

/// The `.shot` files in a directory, sorted, with each cue color read for the
/// browser list (full tracks load only when a shot is selected).
fn scan_match(dir: &str, radius: f64) -> Vec<MatchShot> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("shot"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let shot = parse_shot(&text, radius)?;
            Some(MatchShot {
                path: p,
                cue_color: shot.colors.first().copied().unwrap_or(egui::Color32::WHITE),
                player: header_field(&text, "player"),
                result: header_field(&text, "result"),
            })
        })
        .collect()
}

/// Parse a `.scene` file. Ball lines are `color x y` (table meters); optional
/// `image PATH`, `corners x,y x,y x,y x,y`, and `orient horizontal|vertical`
/// carry the source frame and calibration. Comments start `#`. White is the cue
/// ball; yellow and red are the objects (matching the editor colors).
fn parse_scene(text: &str, radius: f64) -> Option<ImportedScene> {
    let (mut white, mut yellow, mut red) = (None, None, None);
    let mut image_path = None;
    let mut image_corners: Option<[(f64, f64); 4]> = None;
    let mut orient = "horizontal";

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        let rest: Vec<&str> = it.collect();
        match key.to_ascii_lowercase().as_str() {
            "image" => image_path = rest.first().map(|s| s.to_string()),
            "orient" if rest.first() == Some(&"vertical") => orient = "vertical",
            "corners" => {
                let pts: Vec<(f64, f64)> = rest
                    .iter()
                    .filter_map(|p| {
                        let mut c = p.split(',');
                        Some((c.next()?.parse().ok()?, c.next()?.parse().ok()?))
                    })
                    .collect();
                if pts.len() == 4 {
                    image_corners = Some([pts[0], pts[1], pts[2], pts[3]]);
                }
            }
            c @ ("white" | "yellow" | "red") if rest.len() >= 2 => {
                if let (Ok(x), Ok(y)) = (rest[0].parse::<f64>(), rest[1].parse::<f64>()) {
                    let p = DVec3::new(x, y, radius);
                    match c {
                        "white" => white = Some(p),
                        "yellow" => yellow = Some(p),
                        _ => red = Some(p),
                    }
                }
            }
            _ => {}
        }
    }

    let cue = white.or(yellow)?;
    let objects = [yellow, red, white].into_iter().flatten().filter(|&p| p != cue).collect();

    // Build the table→image homography (matching how the balls were lifted).
    let tab2img = image_corners.and_then(|ic| table_to_image(ic, orient));

    Some(ImportedScene { scene: Scene::new(cue, objects), image_path, tab2img })
}

/// Table (meters) → image-pixel homography from the four detected table corners.
/// The corner order must match the tracker's (far-left, far-right, near-right,
/// near-left for a landscape frame; rotated for a portrait inset).
fn table_to_image(corners: [(f64, f64); 4], orient: &str) -> Option<Homography> {
    let (hl, hw) = (1.42, 0.71);
    let tc = if orient == "vertical" {
        [(-hl, hw), (-hl, -hw), (hl, -hw), (hl, hw)]
    } else {
        [(-hl, hw), (hl, hw), (hl, -hw), (-hl, -hw)]
    };
    Homography::from_correspondences(tc, corners)
}

/// A back-reference from a shot to the video clip it was tracked from, so the
/// editor can play the actual footage beside the reconstruction.
struct VideoRef {
    frames_dir: String,
    fps: f64,
    /// Index (into the sorted frames) of the frame at shot time t = 0.
    start: usize,
    /// Table → image-pixel homography, to ring reconstructed positions on the frame.
    tab2img: Option<Homography>,
}

/// A tracked shot: per-ball trajectories ordered cue-first, with each ball's
/// true color (so the editor draws them correctly regardless of which is the cue),
/// plus an optional link back to the source video clip.
struct ImportedShot {
    tracks: Vec<Vec<(f64, DVec3)>>,
    colors: Vec<egui::Color32>,
    video: Option<VideoRef>,
}

fn color_for(name: &str) -> Option<egui::Color32> {
    match name {
        "white" => Some(BALL_COLORS[0]),
        "yellow" => Some(BALL_COLORS[1]),
        "red" => Some(BALL_COLORS[2]),
        _ => None,
    }
}

/// Parse a shot file: a header (`cue COLOR`, and — when tracked from real footage —
/// `frames DIR`, `fps N`, `start N`, `corners …`, `orient …`) then `COLOR,t,x,y`
/// data lines (table meters). This is the tracker's `--fit-out` output. Returns
/// the trajectories ordered cue-first with their true colors, and the video link.
fn parse_shot(text: &str, radius: f64) -> Option<ImportedShot> {
    let (mut white, mut yellow, mut red) = (Vec::new(), Vec::new(), Vec::new());
    let mut cue = "white".to_string();
    let mut frames_dir: Option<String> = None;
    let mut fps = 30.0;
    let mut start = 0usize;
    let mut corners: Option<[(f64, f64); 4]> = None;
    let mut orient = "horizontal";

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Header lines are space-separated `key value…` (data lines are comma-separated).
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let rest: Vec<&str> = it.collect();
        match key {
            "cue" => {
                if let Some(c) = rest.first() { cue = c.to_ascii_lowercase(); }
                continue;
            }
            "frames" => { frames_dir = rest.first().map(|s| s.to_string()); continue; }
            "fps" => { if let Some(f) = rest.first().and_then(|s| s.parse().ok()) { fps = f; } continue; }
            "start" => { if let Some(s) = rest.first().and_then(|s| s.parse().ok()) { start = s; } continue; }
            "orient" if rest.first() == Some(&"vertical") => { orient = "vertical"; continue; }
            "corners" => {
                let pts: Vec<(f64, f64)> = rest
                    .iter()
                    .filter_map(|p| {
                        let mut c = p.split(',');
                        Some((c.next()?.parse().ok()?, c.next()?.parse().ok()?))
                    })
                    .collect();
                if pts.len() == 4 {
                    corners = Some([pts[0], pts[1], pts[2], pts[3]]);
                }
                continue;
            }
            _ => {}
        }

        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 4 {
            continue; // not a data line (and a `.scene` uses spaces, so it won't match)
        }
        let (Ok(t), Ok(x), Ok(y)) = (f[1].trim().parse::<f64>(), f[2].trim().parse(), f[3].trim().parse()) else {
            continue;
        };
        let p = (t, DVec3::new(x, y, radius));
        match f[0].trim() {
            "white" => white.push(p),
            "yellow" => yellow.push(p),
            "red" => red.push(p),
            _ => {}
        }
    }

    // Order cue-first, then the remaining colors; keep each ball's true color.
    let named = [("white", white), ("yellow", yellow), ("red", red)];
    let mut tracks = Vec::new();
    let mut colors = Vec::new();
    for want_cue in [true, false] {
        for (name, tr) in &named {
            if (*name == cue) == want_cue && !tr.is_empty() {
                tracks.push(tr.clone());
                colors.push(color_for(name)?);
            }
        }
    }
    // Needs a cue ball with an actual trajectory.
    if !tracks.first().is_some_and(|t| t.len() > 1) {
        return None;
    }
    let video = frames_dir.map(|dir| VideoRef {
        frames_dir: dir,
        fps,
        start,
        tab2img: corners.and_then(|c| table_to_image(c, orient)),
    });
    Some(ImportedShot { tracks, colors, video })
}

/// The loaded source video clip: the frame image paths (sorted), the timing that
/// maps a playhead time to a frame, and a one-frame texture cache.
struct VideoClip {
    frames: Vec<std::path::PathBuf>,
    fps: f64,
    start: usize,
    tab2img: Option<Homography>,
    /// Frame transform aligning the video's table with the reconstruction beside
    /// it: an optional vertical flip, then 0-3 clockwise quarter-turns. The flip
    /// is needed when the inset's calibration is mirrored relative to the
    /// editor's drawing convention — pure rotations can't fix chirality, and a
    /// mirrored overhead table crop is visually harmless (no text in it).
    xform: (bool, u8),
    /// The currently uploaded frame: (frame index, texture).
    current: Option<(usize, egui::TextureHandle)>,
}

/// Pick the (flip, quarter-turns) that best aligns the video with the
/// reconstruction: table +x pointing right and +y up, as the editor draws them.
fn video_xform(h: &Homography) -> (bool, u8) {
    let o = h.apply(0.0, 0.0);
    let dx = h.apply(0.4, 0.0);
    let dy = h.apply(0.0, 0.4);
    let (dxu, dxv) = (dx.0 - o.0, dx.1 - o.1);
    let (dyu, dyv) = (dy.0 - o.0, dy.1 - o.1);
    let nx = (dxu * dxu + dxv * dxv).sqrt().max(1e-9);
    let ny = (dyu * dyu + dyv * dyv).sqrt().max(1e-9);
    // Vector effect: vertical flip negates dv; each clockwise quarter-turn maps
    // (du,dv) -> (−dv,du).
    let apply = |mut du: f64, mut dv: f64, (flip, r): (bool, u8)| {
        if flip {
            dv = -dv;
        }
        for _ in 0..r {
            (du, dv) = (-dv, du);
        }
        (du, dv)
    };
    let mut best = (false, 0u8);
    let mut best_score = f64::NEG_INFINITY;
    for flip in [false, true] {
        for r in 0u8..4 {
            let x = apply(dxu, dxv, (flip, r));
            let y = apply(dyu, dyv, (flip, r));
            let score = x.0 / nx - y.1 / ny; // +x right, +y up (v grows downward)
            if score > best_score {
                best_score = score;
                best = (flip, r);
            }
        }
    }
    best
}

/// Map an original-frame pixel through the display transform (dims `ow`×`oh`).
fn xform_px(mut u: f64, mut v: f64, ow: f64, oh: f64, (flip, rot): (bool, u8)) -> (f64, f64) {
    if flip {
        v = oh - v;
    }
    let (mut w, mut h) = (ow, oh);
    for _ in 0..rot {
        (u, v) = (h - v, u);
        (w, h) = (h, w);
    }
    let _ = (w, h);
    (u, v)
}

impl VideoClip {
    /// Frame index for shot time `t` (clamped to the available frames).
    fn frame_at(&self, t: f64) -> usize {
        let idx = self.start as i64 + (t * self.fps).round() as i64;
        idx.clamp(0, self.frames.len() as i64 - 1) as usize
    }
}

/// The `.png` frames in a directory, sorted the same way the tracker enumerated
/// them (lexicographic — filenames are zero-padded), so frame indices line up.
fn list_frames(dir: &str) -> Vec<std::path::PathBuf> {
    let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()).is_some_and(|s| s.eq_ignore_ascii_case("png")))
        .collect();
    v.sort();
    v
}

/// First rail contact of a ray from `cue` along `aim` within the ball-center
/// bounds. Returns the hit point and which rail: 0=top(+y),1=bottom,2=left,3=right.
fn aim_rail_hit(cue: DVec3, aim: f64, bounds: [f64; 4]) -> Option<(DVec3, u8)> {
    let [min_x, max_x, min_y, max_y] = bounds;
    let (dx, dy) = (aim.cos(), aim.sin());
    let mut best: Option<(f64, DVec3, u8)> = None;
    let mut consider = |t: f64, rail: u8| {
        if t > 1e-6 {
            let p = DVec3::new(cue.x + dx * t, cue.y + dy * t, cue.z);
            let ok = match rail {
                0 | 1 => p.x >= min_x - 1e-6 && p.x <= max_x + 1e-6,
                _ => p.y >= min_y - 1e-6 && p.y <= max_y + 1e-6,
            };
            if ok && best.map_or(true, |(bt, _, _)| t < bt) {
                best = Some((t, p, rail));
            }
        }
    };
    if dy.abs() > 1e-9 {
        consider((max_y - cue.y) / dy, 0);
        consider((min_y - cue.y) / dy, 1);
    }
    if dx.abs() > 1e-9 {
        consider((min_x - cue.x) / dx, 2);
        consider((max_x - cue.x) / dx, 3);
    }
    best.map(|(_, p, r)| (p, r))
}

/// Turn a [`Repair`]'s numeric deltas into plain coaching lines (only the
/// adjustments that actually matter — sub-threshold tweaks are dropped).
fn repair_advice(rep: &Repair) -> Vec<String> {
    let mut out = Vec::new();
    if rep.d_speed.abs() > 0.1 {
        let word = if rep.d_speed > 0.0 { "harder" } else { "softer" };
        out.push(format!("hit {word} by {:.1} m/s", rep.d_speed.abs()));
    }
    let d_aim = rep.d_aim.to_degrees();
    if d_aim.abs() > 0.6 {
        out.push(format!("turn your aim {d_aim:+.1}°"));
    }
    if rep.d_english.abs() > 0.03 {
        let word = if rep.d_english > 0.0 { "more" } else { "less" };
        out.push(format!("{word} side spin ({:+.0}% R)", rep.d_english * 100.0));
    }
    if rep.d_follow.abs() > 0.03 {
        let word = if rep.d_follow > 0.0 { "more follow" } else { "more draw" };
        out.push(format!("{word} ({:+.0}% R)", rep.d_follow * 100.0));
    }
    if out.is_empty() {
        out.push("a hair different — you were almost there".to_string());
    }
    out
}

/// Largest time step across which a tracked path is still one continuous run; a
/// bigger jump is a detection gap the tracker left unfilled (the ball went
/// somewhere it wasn't seen), and must not be drawn or interpolated across.
const TRACK_GAP_S: f64 = 0.1;

/// Position at time `t` along a tracked trajectory, or `None` if `t` falls in a
/// gap (the ball's location there is genuinely unknown — don't invent it).
fn interp(track: &[(f64, DVec3)], t: f64) -> Option<DVec3> {
    if track.is_empty() {
        return None;
    }
    if t <= track[0].0 {
        return Some(track[0].1);
    }
    for w in track.windows(2) {
        let (t0, p0) = w[0];
        let (t1, p1) = w[1];
        if t <= t1 {
            if t1 - t0 > TRACK_GAP_S {
                return None; // inside an unfilled gap — unknown
            }
            let a = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return Some(p0 + (p1 - p0) * a);
        }
    }
    Some(track.last().unwrap().1)
}

/// Split a tracked path into continuous runs, breaking at detection gaps, and
/// call `emit` with each run's screen points so gaps are drawn as gaps.
fn tracked_runs(track: &[(f64, DVec3)], mut point: impl FnMut(DVec3) -> egui::Pos2, mut emit: impl FnMut(Vec<egui::Pos2>)) {
    let mut run: Vec<egui::Pos2> = Vec::new();
    let mut last_t: Option<f64> = None;
    for &(t, p) in track {
        if let Some(pt) = last_t {
            if t - pt > TRACK_GAP_S && run.len() >= 2 {
                emit(std::mem::take(&mut run));
            } else if t - pt > TRACK_GAP_S {
                run.clear();
            }
        }
        run.push(point(p));
        last_t = Some(t);
    }
    if run.len() >= 2 {
        emit(run);
    }
}

/// Invoke `f(prev, s, e, frame_dt, gap_dt)` for each detection gap in a track:
/// `s` is the last point before the gap (arriving from `prev` one frame earlier)
/// and `e` the first point after it. A gap at the very start is skipped — there's
/// no preceding point to read the ball's heading from.
fn track_gaps(track: &[(f64, DVec3)], mut f: impl FnMut(DVec3, DVec3, DVec3, f64, f64)) {
    for i in 2..track.len() {
        let (tp, pp) = track[i - 2];
        let (t0, p0) = track[i - 1];
        let (t1, p1) = track[i];
        if t1 - t0 > TRACK_GAP_S && t0 - tp <= TRACK_GAP_S {
            f(pp, p0, p1, t0 - tp, t1 - t0);
        }
    }
}

/// A physics-plausible path bridging a tracking gap. Starting from the last seen
/// point `s` with the velocity implied by `prev`→`s`, the ball rolls straight and
/// reflects off the cushions; we march until closest approach to the recovery
/// point `e`. Returns the bridge polyline when a simple roll-and-bounce reconnects
/// within tolerance — the gap is explained by the ball rolling into a rail/corner
/// and back — else `None` (a collision or a bad heading lives in the gap, so we
/// don't invent a path). This is a *prediction*: drawn dashed, never fed to the
/// physics fit, which only ever sees the observed points.
fn physics_bridge(
    prev: DVec3, s: DVec3, e: DVec3, frame_dt: f64, gap_dt: f64, bounds: [f64; 4],
) -> Option<Vec<DVec3>> {
    let [min_x, max_x, min_y, max_y] = bounds;
    let (dx, dy) = (s.x - prev.x, s.y - prev.y);
    let seg = (dx * dx + dy * dy).sqrt();
    let entry_speed = seg / frame_dt.max(1e-6);
    if entry_speed < 0.05 || seg < 1e-6 {
        return None; // essentially stationary at the gap — nothing to dead-reckon
    }
    let (mut ux, mut uy) = (dx / seg, dy / seg);
    let step = 0.005;
    let max_len = entry_speed * gap_dt * 1.4; // constant-speed reach + slack for decel / estimate error
    let n = ((max_len / step).ceil() as usize).min(4000);
    let (mut px, mut py) = (s.x, s.y);
    let mut pts = vec![s];
    let mut best_d = ((px - e.x).powi(2) + (py - e.y).powi(2)).sqrt();
    let mut best_i = 0usize;
    for k in 1..=n {
        px += ux * step;
        py += uy * step;
        // Reflect the center off the cushion bounds (specular; energy loss is
        // second-order over a short gap and irrelevant to the drawn shape).
        if px < min_x { px = 2.0 * min_x - px; ux = -ux; }
        else if px > max_x { px = 2.0 * max_x - px; ux = -ux; }
        if py < min_y { py = 2.0 * min_y - py; uy = -uy; }
        else if py > max_y { py = 2.0 * max_y - py; uy = -uy; }
        pts.push(DVec3::new(px, py, s.z));
        let d = ((px - e.x).powi(2) + (py - e.y).powi(2)).sqrt();
        if d < best_d { best_d = d; best_i = k; }
    }
    if best_i < 1 || best_d > 0.09 {
        return None; // never came back near where the ball was re-acquired
    }
    pts.truncate(best_i + 1);
    pts.push(e); // close the final short hop onto the observed recovery point
    Some(pts)
}

/// Which action the cue controls are showing: the reconstructed shot, its
/// repaired version, or the solver's best shot for this scene.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActionView {
    Shot,
    Repaired,
    Solved,
}

struct App {
    table: TableSpec,
    ball: BallSpec,
    phys: PhysicsParams,
    scene: Scene,
    aim_deg: f64,
    speed: f64,
    sidespin: f64,
    follow: f64,
    playhead: f64,
    playing: bool,
    dragging: Option<usize>,
    solution: Option<Solution>,
    // Actual-frame overlay (from an imported video scene).
    view_mode: ViewMode,
    image_path: Option<String>,
    texture: Option<egui::TextureHandle>,
    tab2img: Option<Homography>,
    // Imported shot: the actual tracked trajectories + the reconstruction fit.
    tracked: Option<Vec<Vec<(f64, DVec3)>>>,
    fit_rms: Option<f64>,
    show_tracked: bool,
    /// Bridge tracking gaps with a dashed physics prediction (dead-reckon + cushion
    /// reflection) — makes "ball went into the corner and back" legible without
    /// fabricating tracked data (bridges are display-only, kept out of the fit).
    show_gap_fill: bool,
    /// Color to draw each ball (scene order), so the cue isn't forced to white.
    ball_colors: Vec<egui::Color32>,
    /// Source video clip (shown beside the reconstruction, synced to the playhead).
    video: Option<VideoClip>,
    /// Overlay the *reconstruction* (model) on the video frame.
    overlay_on_video: bool,
    /// Overlay the *tracking* (raw detected path + position) on the video frame —
    /// so tracker mistakes (a frozen or jumped ball) are visible against reality.
    overlay_tracked_on_video: bool,
    /// Playback speed multiplier (slow-mo to inspect a shot frame by frame).
    play_speed: f64,
    /// Loaded match: every tracked shot, for browsing.
    match_shots: Vec<MatchShot>,
    /// Where the active physics params came from: "per-game" (calibration.json) or "default".
    calib_source: String,
    /// Multi-game match: the manifest games and which is loaded (empty for a single game).
    match_root: Option<String>,
    games: Vec<GameEntry>,
    current_game: Option<usize>,
    current_shot: Option<usize>,
    /// Result of the last shot-repair search (nearest scoring action, or None if
    /// nothing nearby scores). `Some(None)` = searched, found nothing.
    repair: Option<Option<Repair>>,
    /// The reconstructed (fitted) action of the loaded shot, kept so switching
    /// between shot / repaired / solved never loses the original.
    shot_action: Option<billiards_core::CueAction>,
    /// Whether the loaded shot's reconstruction scores (a made shot offers no
    /// "repaired" view — there is nothing to repair).
    shot_scored: bool,
    /// Make-probability of the player's line under execution noise.
    shot_prob: Option<f64>,
    /// Which of the three actions the cue controls currently show.
    active_view: ActionView,
    /// Bug-report window state: free text, an optional bad time segment, an
    /// optional ball + its correct position (clicked on the table). Saved
    /// reports land in `reports/` where the assistant picks them up.
    report_open: bool,
    report_text: String,
    report_t0: Option<f64>,
    report_t1: Option<f64>,
    report_ball: usize,
    report_pos: Option<DVec3>,
    /// Armed: the next click on the table records the ball's correct position.
    report_marking: bool,
    /// Path of the currently loaded shot/scene file (for the report header).
    loaded_path: Option<String>,
}

impl App {
    fn new() -> Self {
        let ball = BallSpec::carom();
        let r = ball.radius;
        let scene = Scene::new(
            DVec3::new(-1.15, -0.45, r),
            vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)],
        );
        Self {
            table: TableSpec::carom_match(),
            ball,
            phys: PhysicsParams::carom_calibrated(),
            scene,
            aim_deg: 20.0,
            speed: 4.5,
            sidespin: 0.0,
            follow: 0.0,
            playhead: 0.0,
            playing: false,
            dragging: None,
            solution: None,
            view_mode: ViewMode::Reconstructed,
            image_path: None,
            texture: None,
            tab2img: None,
            tracked: None,
            fit_rms: None,
            show_tracked: false,
            show_gap_fill: true,
            ball_colors: BALL_COLORS.to_vec(),
            video: None,
            overlay_on_video: false,
            overlay_tracked_on_video: true,
            play_speed: 1.0,
            match_shots: Vec::new(),
            calib_source: "default".into(),
            match_root: None,
            games: Vec::new(),
            current_game: None,
            current_shot: None,
            repair: None,
            shot_action: None,
            shot_scored: false,
            shot_prob: None,
            active_view: ActionView::Shot,
            report_open: false,
            report_text: String::new(),
            report_t0: None,
            report_t1: None,
            report_ball: 0,
            report_pos: None,
            report_marking: false,
            loaded_path: None,
        }
    }

    /// Load a whole tracked match (a directory of `.shot` files) for browsing.
    /// Open a directory: a multi-game match (has `match.json`) shows a game picker;
    /// a plain directory of `.shot` files loads directly.
    fn open_dir(&mut self, dir: &str) {
        let games = parse_match_json(dir);
        if games.is_empty() {
            self.match_root = None;
            self.games.clear();
            self.current_game = None;
            self.load_match(dir);
        } else {
            self.match_root = Some(dir.trim_end_matches('/').to_string());
            self.games = games;
            self.load_game(0);
        }
    }

    /// Load game `g` of a multi-game match (its own shot dir + calibration).
    fn load_game(&mut self, g: usize) {
        if let (Some(root), Some(entry)) = (self.match_root.clone(), self.games.get(g)) {
            let dir = format!("{root}/{}", entry.dir);
            self.current_game = Some(g);
            self.load_match(&dir);
        }
    }

    fn load_match(&mut self, dir: &str) {
        self.match_shots = scan_match(dir, self.ball.radius);
        // Each game carries its own calibrated physics; use it if present.
        match load_calibration(dir) {
            Some(p) => { self.phys = p; self.calib_source = "per-game".into(); }
            None => { self.phys = PhysicsParams::carom_calibrated(); self.calib_source = "default".into(); }
        }
        self.repair = None;
        if self.match_shots.is_empty() {
            eprintln!("no .shot files in {dir}");
        } else {
            self.load_shot_index(0);
        }
    }

    /// Load shot `i` of the current match (full tracks + video) into the editor.
    fn load_shot_index(&mut self, i: usize) {
        if let Some(ms) = self.match_shots.get(i) {
            let path = ms.path.clone();
            self.current_shot = Some(i);
            self.repair = None;
            self.try_load_scene(&path.to_string_lossy());
        }
    }

    /// Shot repair (coaching): search near the current cue action for the smallest
    /// change that turns this shot into a score, and report the adjustment.
    fn repair_shot(&mut self) {
        self.playing = false;
        self.playhead = 0.0;
        let found = repair(
            &self.scene,
            &self.action(),
            &self.table,
            &self.ball,
            &self.phys,
            &RepairConfig::default(),
        );
        self.repair = Some(found);
    }

    /// Human name of ball `i` (scene order), from its display color.
    fn ball_name(&self, i: usize) -> &'static str {
        match self.ball_colors.get(i) {
            Some(c) if *c == BALL_COLORS[0] => "white",
            Some(c) if *c == BALL_COLORS[1] => "yellow",
            Some(c) if *c == BALL_COLORS[2] => "red",
            _ => "ball",
        }
    }

    /// The floating bug-report window: free text + optional structured marks
    /// (bad time segment, ball, its correct table position). Saved to
    /// `reports/report_<unix>.md` for the assistant to pick up.
    fn report_window(&mut self, ctx: &egui::Context) {
        if !self.report_open {
            return;
        }
        let mut open = true;
        let mut save = false;
        egui::Window::new("🚩 Report a problem").open(&mut open).show(ctx, |ui| {
            if let Some(p) = &self.loaded_path {
                ui.label(egui::RichText::new(p.as_str()).weak().size(10.0));
            }
            ui.label("What's wrong?");
            ui.add(egui::TextEdit::multiline(&mut self.report_text).desired_rows(4).desired_width(300.0));
            ui.separator();
            ui.label(egui::RichText::new("Optional marks (scrub the playhead first):").weak());
            ui.horizontal(|ui| {
                if ui.button("bad from ⏵").clicked() {
                    self.report_t0 = Some(self.playhead);
                }
                if ui.button("bad until ⏵").clicked() {
                    self.report_t1 = Some(self.playhead);
                }
                match (self.report_t0, self.report_t1) {
                    (Some(a), Some(b)) => { ui.label(format!("{a:.2}–{b:.2}s")); }
                    (Some(a), None) => { ui.label(format!("{a:.2}s–…")); }
                    _ => { ui.weak("no segment marked"); }
                }
            });
            ui.horizontal(|ui| {
                ui.label("ball:");
                let n = 1 + self.scene.objects.len();
                for i in 0..n {
                    let name = self.ball_name(i);
                    if ui.selectable_label(self.report_ball == i, name).clicked() {
                        self.report_ball = i;
                    }
                }
            });
            ui.horizontal(|ui| {
                let armed = self.report_marking;
                if ui.selectable_label(armed, "📍 click table = correct position").clicked() {
                    self.report_marking = !armed;
                }
                if let Some(p) = self.report_pos {
                    ui.label(format!("({:.3}, {:.3})", p.x, p.y));
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.report_text.trim().is_empty() || self.report_pos.is_some(),
                                  egui::Button::new(egui::RichText::new("Save report").strong())).clicked() {
                    save = true;
                }
                if ui.button("Discard").clicked() {
                    save = false;
                    self.report_open = false;
                    self.report_text.clear();
                }
            });
        });
        if save {
            self.save_report();
        }
        if !open {
            self.report_open = false;
        }
    }

    fn save_report(&mut self) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut s = format!("# Shot report {ts}\n");
        if let Some(p) = &self.loaded_path {
            s += &format!("shot: {p}\n");
        }
        s += &format!("playhead: {:.2}\n", self.playhead);
        if let (Some(a), b) = (self.report_t0, self.report_t1) {
            s += &format!("bad_segment: {a:.2} - {}\n", b.map_or("end".into(), |b| format!("{b:.2}")));
        }
        if self.report_pos.is_some() || self.report_t0.is_some() {
            s += &format!("ball: {}\n", self.ball_name(self.report_ball));
        }
        if let Some(p) = self.report_pos {
            s += &format!("correct_position: {:.4}, {:.4}\n", p.x, p.y);
        }
        s += &format!("view: {}\n\n", match self.active_view {
            ActionView::Shot => "shot",
            ActionView::Repaired => "repaired",
            ActionView::Solved => "solved",
        });
        s += self.report_text.trim();
        s += "\n";
        let _ = std::fs::create_dir_all("reports");
        let path = format!("reports/report_{ts}.md");
        match std::fs::write(&path, s) {
            Ok(()) => {
                eprintln!("report saved to {path}");
                self.report_open = false;
                self.report_text.clear();
                self.report_t0 = None;
                self.report_t1 = None;
                self.report_pos = None;
                self.report_marking = false;
            }
            Err(e) => eprintln!("could not save report: {e}"),
        }
    }

    /// Load an action into the cue controls (aim/speed/spin).
    fn set_action(&mut self, a: &billiards_core::CueAction) {
        self.aim_deg = a.aim.to_degrees().rem_euclid(360.0);
        self.speed = a.speed;
        self.sidespin = a.sidespin;
        self.follow = a.follow;
    }

    /// Switch which action (shot / repaired / solved) the controls show.
    fn show_view(&mut self, v: ActionView) {
        let a = match v {
            ActionView::Shot => self.shot_action,
            ActionView::Repaired => self.repair.flatten().map(|r| r.action),
            ActionView::Solved => self.solution.as_ref().map(|s| s.action),
        };
        if let Some(a) = a {
            self.active_view = v;
            self.set_action(&a);
            self.playing = false;
            self.playhead = 0.0;
        }
    }

    /// Color for ball `i` (scene order), falling back to the default palette.
    fn ball_color(&self, i: usize) -> egui::Color32 {
        self.ball_colors.get(i).copied().unwrap_or(egui::Color32::GRAY)
    }

    fn action(&self) -> billiards_core::CueAction {
        billiards_core::CueAction {
            aim: self.aim_deg.to_radians(),
            speed: self.speed,
            sidespin: self.sidespin,
            follow: self.follow,
        }
    }

    fn ball_pos(&self, i: usize) -> DVec3 {
        if i == 0 { self.scene.cue } else { self.scene.objects[i - 1] }
    }

    fn set_ball_pos(&mut self, i: usize, p: DVec3) {
        if i == 0 {
            self.scene.cue = p;
        } else {
            self.scene.objects[i - 1] = p;
        }
    }

    fn apply_imported(&mut self, imp: ImportedScene) {
        self.scene = imp.scene;
        self.image_path = imp.image_path;
        self.tab2img = imp.tab2img;
        self.texture = None; // reload lazily in update()
        self.tracked = None;
        self.fit_rms = None;
        self.video = None;
        self.ball_colors = BALL_COLORS.to_vec(); // white cue, yellow, red
        // If we have the source frame, open on the actual view so the match is verifiable.
        self.view_mode = if self.image_path.is_some() { ViewMode::Actual } else { ViewMode::Reconstructed };
        self.playing = false;
        self.playhead = 0.0;
        self.solution = None;
        self.repair = None;
        self.shot_action = None; // a plain scene has no tracked shot to return to
        self.active_view = ActionView::Shot;
    }

    /// Import a tracked shot: set the scene from t=0, reconstruct the cue action,
    /// and keep the tracked trajectories for the actual-vs-reconstructed overlay.
    fn apply_shot(&mut self, shot: ImportedShot) {
        let tracks = shot.tracks;
        // Shared scene construction (same as calibration/verification): corrects an
        // object ball that was first detected mid-motion (occluded at address) back
        // to its rest, so the collision geometry starts from the right place.
        let Some(scene) = billiards_solver::shotfile::scene_from_tracks(&tracks, &self.table, self.ball.radius) else {
            return eprintln!("shot has no usable cue track");
        };
        self.scene = scene;
        self.ball_colors = shot.colors; // true color per ball (cue-first order)

        // Pin the aim to the cue's observed pre-collision heading (it's directly
        // measured and physics-independent); only speed + spin are fit. A wider
        // window let the fit rotate the start off the real heading to hide physics
        // error, so the reconstruction's starting vector disagreed with reality.
        let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
        let fit = fit_action(&self.scene, &tracks, &self.table, &self.ball, &self.phys, &cfg);
        self.set_action(&fit.action);
        self.fit_rms = Some(fit.rms_m);
        // Remember the reconstruction so repaired/solved views never destroy it,
        // and its verdict (a made shot has nothing to repair).
        self.shot_action = Some(fit.action);
        let shot_sim = simulate(&self.scene.ball_states(&fit.action), &self.table, &self.ball, &self.phys);
        self.shot_scored = three_cushion_score(&shot_sim, BallId(0));
        self.shot_prob = Some(success_probability(
            &self.scene, &fit.action, &self.table, &self.ball, &self.phys, &SolveConfig::default(),
        ));
        self.active_view = ActionView::Shot;
        self.repair = None;
        self.solution = None;

        self.tracked = Some(tracks);
        // The tracked overlay is opt-in: the reconstruction reads cleaner on its
        // own, and the video pane already shows the raw tracking against reality.
        self.view_mode = ViewMode::Reconstructed; // shots play in the top-down view
        self.image_path = None;
        self.texture = None;
        self.tab2img = None;
        // Link the source video clip, if this shot carried one (and its frames exist).
        self.video = shot.video.and_then(|vr| {
            let frames = list_frames(&vr.frames_dir);
            if frames.is_empty() {
                eprintln!("shot references frames in {} but none were found", vr.frames_dir);
                None
            } else {
                Some(VideoClip {
                    frames,
                    fps: vr.fps,
                    start: vr.start,
                    xform: vr.tab2img.as_ref().map_or((false, 0), video_xform),
                    tab2img: vr.tab2img,
                    current: None,
                })
            }
        });
        self.playing = false;
        self.playhead = 0.0;
        self.solution = None;
    }

    /// Load a `.scene` or a shot CSV (auto-detected) from the video pipeline.
    fn try_load_scene(&mut self, path: &str) {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => return eprintln!("could not read {path}: {e}"),
        };
        self.loaded_path = Some(path.to_string());
        self.report_t0 = None;
        self.report_t1 = None;
        self.report_pos = None;
        self.report_marking = false;
        if let Some(tracks) = parse_shot(&text, self.ball.radius) {
            self.apply_shot(tracks);
        } else if let Some(imp) = parse_scene(&text, self.ball.radius) {
            self.apply_imported(imp);
        } else {
            eprintln!("could not parse {path} as a scene or shot");
        }
    }

    /// Positions of all balls (cue first), for overlaying markers.
    fn all_balls(&self) -> Vec<DVec3> {
        std::iter::once(self.scene.cue).chain(self.scene.objects.iter().copied()).collect()
    }

    /// Load the actual frame into a GPU texture (once) when a scene has one.
    fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.texture.is_some() {
            return;
        }
        let Some(path) = self.image_path.clone() else { return };
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &rgba.into_raw());
                self.texture = Some(ctx.load_texture("actual_frame", ci, Default::default()));
            }
            Err(e) => {
                eprintln!("could not load frame {path}: {e}");
                self.image_path = None;
            }
        }
    }

    /// Upload the source-video frame for the current playhead time (only when it
    /// changes) so the actual clip tracks the reconstruction as it plays.
    fn ensure_video_frame(&mut self, ctx: &egui::Context) {
        let playhead = self.playhead;
        let Some(v) = self.video.as_mut() else { return };
        if v.frames.is_empty() {
            return;
        }
        let idx = v.frame_at(playhead);
        if v.current.as_ref().map(|(i, _)| *i) == Some(idx) {
            return; // already showing this frame
        }
        match image::open(&v.frames[idx]) {
            Ok(img) => {
                // Transform the frame so the video table lies the same way as
                // the reconstruction (overlays go through `xform_px` to match).
                let (flip, rot) = v.xform;
                let mut rgba = img.to_rgba8();
                if flip {
                    rgba = image::imageops::flip_vertical(&rgba);
                }
                let rgba = match rot {
                    1 => image::imageops::rotate90(&rgba),
                    2 => image::imageops::rotate180(&rgba),
                    3 => image::imageops::rotate270(&rgba),
                    _ => rgba,
                };
                let size = [rgba.width() as usize, rgba.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &rgba.into_raw());
                let tex = ctx.load_texture("video_frame", ci, Default::default());
                v.current = Some((idx, tex));
            }
            Err(e) => eprintln!("could not load frame {}: {e}", v.frames[idx].display()),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_texture(ctx);
        self.ensure_video_frame(ctx);
        let sim = simulate(&self.scene.ball_states(&self.action()), &self.table, &self.ball, &self.phys);
        let scored = three_cushion_score(&sim, BallId(0));
        let total = sim.settled_time();

        self.side_panel(ctx, scored, total);
        self.report_window(ctx);
        self.video_panel(ctx, &sim);
        self.table_panel(ctx, &sim);

        if self.playing && total > 0.0 {
            let dt = ctx.input(|i| i.stable_dt) as f64;
            self.playhead += dt * self.play_speed;
            if self.playhead >= total {
                self.playhead = 0.0;
            }
            ctx.request_repaint();
        }
    }
}

impl App {
    /// Browse a loaded match: prev/next and a shot list.
    fn match_browser(&mut self, ui: &mut egui::Ui) {
        // Multi-game match: pick which game to browse (players + make totals).
        if !self.games.is_empty() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Pick a game").strong());
            let mut pick: Option<usize> = None;
            egui::ScrollArea::vertical().id_salt("gamelist").max_height(96.0).show(ui, |ui| {
                for (gi, g) in self.games.iter().enumerate() {
                    let made = g.n_made.map(|m| format!("  ({m}/{} made)", g.n_shots))
                        .unwrap_or_else(|| format!("  ({} shots)", g.n_shots));
                    let label = format!("{} vs {}{}", g.left, g.right, made);
                    if ui.selectable_label(Some(gi) == self.current_game, label).clicked() {
                        pick = Some(gi);
                    }
                }
            });
            if let Some(gi) = pick {
                self.load_game(gi);
            }
            ui.separator();
        }

        if self.match_shots.is_empty() {
            return;
        }
        let n = self.match_shots.len();
        let cur = self.current_shot.unwrap_or(0);
        ui.add_space(4.0);
        ui.label(egui::RichText::new(format!("Match · {n} shots")).strong());
        // Make totals per player, if the scoreboard annotator has run.
        let annotated = self.match_shots.iter().any(|s| s.result.is_some());
        if annotated {
            let (mut lm, mut lt, mut rm, mut rt) = (0, 0, 0, 0);
            for s in &self.match_shots {
                let made = s.result.as_deref() == Some("make");
                match s.player.as_deref() {
                    Some("left") => { lt += 1; lm += made as i32; }
                    Some("right") => { rt += 1; rm += made as i32; }
                    _ => {}
                }
            }
            ui.label(egui::RichText::new(format!("left {lm}/{lt} made   ·   right {rm}/{rt} made"))
                .size(11.0).weak());
        }
        ui.label(egui::RichText::new(format!("physics: {}", self.calib_source)).size(11.0).weak());

        let mut goto: Option<usize> = None;
        ui.horizontal(|ui| {
            if ui.button("◀").clicked() && cur > 0 {
                goto = Some(cur - 1);
            }
            ui.label(format!("shot {cur} / {}", n - 1));
            if ui.button("▶").clicked() && cur + 1 < n {
                goto = Some(cur + 1);
            }
        });

        egui::ScrollArea::vertical().id_salt("shotlist").max_height(120.0).show(ui, |ui| {
            for (i, ms) in self.match_shots.iter().enumerate() {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, ms.cue_color);
                    ui.painter().circle_stroke(rect.center(), 5.0, egui::Stroke::new(1.0, egui::Color32::from_gray(30)));
                    let side = match ms.player.as_deref() {
                        Some("left") => "L", Some("right") => "R", _ => " ",
                    };
                    let label = format!("shot {i}  {side}");
                    if ui.selectable_label(Some(i) == self.current_shot, label).clicked() {
                        goto = Some(i);
                    }
                    // Make/miss verdict from the scoreboard: green check = the
                    // player's score changed after this shot, red cross = miss.
                    // Drawn (not text): the built-in font lacks ✓/✗ glyphs.
                    match ms.result.as_deref() {
                        Some("make") => {
                            let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                            let c = r.center();
                            let s = egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 200, 110));
                            ui.painter().line_segment([c + egui::vec2(-4.0, 0.0), c + egui::vec2(-1.0, 3.5)], s);
                            ui.painter().line_segment([c + egui::vec2(-1.0, 3.5), c + egui::vec2(4.5, -3.5)], s);
                        }
                        Some("miss") => {
                            let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                            let c = r.center();
                            let s = egui::Stroke::new(2.0, egui::Color32::from_rgb(210, 90, 90));
                            ui.painter().line_segment([c + egui::vec2(-3.5, -3.5), c + egui::vec2(3.5, 3.5)], s);
                            ui.painter().line_segment([c + egui::vec2(-3.5, 3.5), c + egui::vec2(3.5, -3.5)], s);
                        }
                        _ => {}
                    }
                });
            }
        });
        if let Some(i) = goto {
            self.load_shot_index(i);
        }
        ui.separator();
    }

    fn side_panel(&mut self, ctx: &egui::Context, scored: bool, total: f64) {
        egui::SidePanel::left("controls").exact_width(268.0).show(ctx, |ui| {
            ui.add_space(6.0);
            ui.heading("Three-Cushion Coach");
            ui.add_space(2.0);
            // Everything below scrolls, so the panel never outgrows the window.
            // `drag_to_scroll(false)`: otherwise the scroll area swallows drags and
            // the strike diagram (drag to place english/follow) stops responding.
            egui::ScrollArea::vertical().auto_shrink([false, false]).drag_to_scroll(false).show(ui, |ui| {
                self.source_section(ui);
                ui.separator();
                self.cue_section(ui, total);
                // (the reconstructed-hit readout lives in the table pane's header)
                ui.separator();
                self.coach_section(ui, scored);
                ui.add_space(8.0);
            });
        });
    }

    /// Load buttons + match browser + the Actual/Reconstructed view toggle.
    fn source_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("📂 Import…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("scene/shot", &["scene", "shot", "txt", "csv"])
                    .pick_file()
                {
                    self.match_shots.clear();
                    self.current_shot = None;
                    self.try_load_scene(&path.to_string_lossy());
                }
            }
            if ui.button("📁 Open match…").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.open_dir(&dir.to_string_lossy());
                }
            }
            if ui.button("🚩 Report").on_hover_text("mark a tracking/reconstruction problem on this shot").clicked() {
                self.report_open = true;
            }
        });
        self.match_browser(ui);
        if self.texture.is_some() {
            ui.horizontal(|ui| {
                ui.label("view:");
                ui.selectable_value(&mut self.view_mode, ViewMode::Actual, "Actual");
                ui.selectable_value(&mut self.view_mode, ViewMode::Reconstructed, "Reconstructed");
            });
        }
    }

    /// Cue action (aim/speed + strike diagram) and playback transport.
    fn cue_section(&mut self, ui: &mut egui::Ui, total: f64) {
        ui.label(egui::RichText::new("Cue action").strong());
        ui.add(egui::Slider::new(&mut self.aim_deg, 0.0..=360.0).text("aim °"));
        ui.add(egui::Slider::new(&mut self.speed, 0.5..=6.5).text("speed m/s"));
        self.strike_diagram(ui);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let label = if self.playing { "⏸ pause" } else { "▶ play" };
            if ui.button(label).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("⟲ reset").clicked() {
                self.playing = false;
                self.playhead = 0.0;
            }
            ui.add(egui::Slider::new(&mut self.playhead, 0.0..=total.max(1e-3)).text("s"));
        });
        // Playback pace — slow-mo to inspect exactly what a shot (or the tracker) does.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("pace").weak());
            for (label, s) in [("⅛×", 0.125), ("¼×", 0.25), ("½×", 0.5), ("1×", 1.0), ("2×", 2.0)] {
                if ui.selectable_label((self.play_speed - s).abs() < 1e-6, label).clicked() {
                    self.play_speed = s;
                }
            }
        });
    }

    /// Reconstructed-hit readout for an imported shot (force/spin + fit error).
    /// Score verdict + the coach tools (Solve / Repair) side by side.
    fn coach_section(&mut self, ui: &mut egui::Ui, scored: bool) {
        let (txt, color) = if scored {
            ("● THREE-CUSHION POINT", egui::Color32::from_rgb(90, 210, 120))
        } else {
            ("○ no score", egui::Color32::GRAY)
        };
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(txt).color(color).strong().size(15.0));
            if let Some(p) = self.shot_prob {
                ui.label(
                    egui::RichText::new(format!("~{:.0}% line", p * 100.0))
                        .weak()
                        .size(11.0),
                )
                .on_hover_text("make-probability of the attempted line under human execution noise");
            }
        });

        ui.add_space(4.0);
        // The three views: the tracked shot's reconstruction, its repaired
        // version (only offered for a miss), and the solver's best shot for the
        // layout. Repair/solve compute lazily on first selection; switching
        // never destroys anything.
        let have_shot = self.shot_action.is_some();
        let mut v = self.active_view;
        ui.horizontal(|ui| {
            if have_shot {
                ui.selectable_value(&mut v, ActionView::Shot, "shot");
                if !self.shot_scored {
                    ui.selectable_value(&mut v, ActionView::Repaired, "repaired");
                }
            }
            ui.selectable_value(&mut v, ActionView::Solved, "solved");
        });
        if v != self.active_view {
            match v {
                ActionView::Repaired if self.repair.is_none() => self.repair_shot(),
                ActionView::Solved if self.solution.is_none() => {
                    self.solution =
                        solve(&self.scene, &self.table, &self.ball, &self.phys, &SolveConfig::default());
                }
                _ => {}
            }
            // No-op when the search found nothing (stays on the current view and
            // the message below explains).
            self.show_view(v);
        }

        match self.active_view {
            ActionView::Solved => {
                if let Some(s) = &self.solution {
                    ui.label(egui::RichText::new("Best shot").strong());
                    ui.label(format!("success {:.0}%   ·   {} ({:.2})", s.success_prob * 100.0, s.category(), s.difficulty()));
                    ui.label(egui::RichText::new(format!("{} scoring options", s.scoring_cells)).weak());
                }
            }
            ActionView::Repaired => {
                if let Some(rep) = self.repair.flatten() {
                    ui.label(egui::RichText::new("Smallest fix to score:").strong());
                    for line in repair_advice(&rep) {
                        ui.label(format!("  • {line}"));
                    }
                    ui.label(egui::RichText::new(format!("→ scores ~{:.0}%", rep.success_prob * 100.0)).weak());
                }
            }
            ActionView::Shot => match self.repair {
                Some(None) => {
                    ui.colored_label(egui::Color32::from_rgb(224, 150, 90), "✗ No nearby scoring shot");
                    ui.label(egui::RichText::new("looks like the wrong shot, not a mis-hit").weak().size(11.0));
                }
                Some(Some(rep)) if rep.already_scores => {
                    ui.colored_label(egui::Color32::from_rgb(90, 210, 120), "✓ This shot already scores");
                }
                _ => {
                    ui.label(egui::RichText::new("Solve finds the most forgiving shot; Repair finds the smallest fix to your shot.").weak().size(11.0));
                }
            },
        }
    }

    /// Cue-ball strike diagram: shows (and lets you drag) the tip contact point,
    /// with a power bar. The contact point is derived from the current spin and
    /// *speed* — so it moves as you change how hard you hit, making the
    /// speed↔offset coupling (and miscue risk) visible.
    fn strike_diagram(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Cue tip — where & how hard").strong());

        let size = egui::vec2(ui.available_width().min(210.0), 148.0);
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        let rect = resp.rect;
        let face_r = (rect.height() * 0.5 - 8.0).min(rect.width() * 0.5 - 26.0);
        let face_c = egui::pos2(rect.left() + face_r + 6.0, rect.center().y);
        let radius = self.ball.radius;

        // Ball face, crosshair, and miscue-limit (~½R) ring.
        painter.circle_filled(face_c, face_r, egui::Color32::from_rgb(244, 244, 236));
        painter.circle_stroke(face_c, face_r, egui::Stroke::new(1.5, egui::Color32::from_gray(70)));
        let faint = egui::Stroke::new(1.0, egui::Color32::from_gray(205));
        painter.line_segment([face_c - egui::vec2(face_r, 0.0), face_c + egui::vec2(face_r, 0.0)], faint);
        painter.line_segment([face_c - egui::vec2(0.0, face_r), face_c + egui::vec2(0.0, face_r)], faint);
        painter.circle_stroke(face_c, face_r * 0.5, egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 170, 120)));

        // Dragging the face sets the spin from the tip offset at the current speed.
        if (resp.dragged() || resp.clicked()) && !self.playing {
            if let Some(p) = resp.interact_pointer_pos() {
                let mut h = ((p.x - face_c.x) / face_r) as f64;
                let mut v = (-(p.y - face_c.y) / face_r) as f64;
                let mag = (h * h + v * v).sqrt();
                if mag > 0.9 {
                    h *= 0.9 / mag;
                    v *= 0.9 / mag;
                }
                let a = billiards_core::CueAction::from_tip_offset(self.aim_deg.to_radians(), self.speed, h, v, radius);
                self.sidespin = a.sidespin;
                self.follow = a.follow;
                self.solution = None;
            }
        }

        // Contact dot from the current action (clamped to the face for display).
        let (fh, fv) = self.action().tip_offset(radius);
        let mag = (fh * fh + fv * fv).sqrt();
        let (dh, dv) = if mag > 1.0 { (fh / mag, fv / mag) } else { (fh, fv) };
        let dot = face_c + egui::vec2(dh as f32 * face_r, -(dv as f32) * face_r);
        let miscue = mag > 0.5;
        let dot_col = if miscue { egui::Color32::from_rgb(220, 80, 60) } else { egui::Color32::from_rgb(70, 160, 230) };
        painter.circle_filled(dot, 6.0, dot_col);
        painter.circle_stroke(dot, 6.0, egui::Stroke::new(1.0, egui::Color32::from_gray(30)));

        // Power bar (speed) on the right.
        let frac = (((self.speed - 0.5) / 6.0).clamp(0.0, 1.0)) as f32;
        let bx = rect.right() - 8.0;
        let (top, bot) = (rect.top() + 8.0, rect.bottom() - 8.0);
        let track = egui::Rect::from_min_max(egui::pos2(bx - 5.0, top), egui::pos2(bx + 5.0, bot));
        painter.rect_filled(track, egui::Rounding::same(3.0), egui::Color32::from_gray(60));
        let fill = egui::Rect::from_min_max(egui::pos2(bx - 5.0, bot - frac * (bot - top)), egui::pos2(bx + 5.0, bot));
        painter.rect_filled(fill, egui::Rounding::same(3.0), egui::Color32::from_rgb(120, 200, 255));

        ui.label(format!("english {:+.0}%R · follow {:+.0}%R", fh * 100.0, fv * 100.0));
        if miscue {
            ui.colored_label(egui::Color32::from_rgb(220, 120, 90), "⚠ past ½R — miscue / hit harder");
        }
        ui.label(egui::RichText::new("harder shots need less offset for the same spin").italics().weak().size(11.0));
    }

    /// The actual source-video clip, on the right, playing in sync with the
    /// reconstruction. Reconstructed ball positions are ringed on the frame so you
    /// can see, frame by frame, how the model lines up with reality.
    fn video_panel(&mut self, ctx: &egui::Context, sim: &billiards_core::Simulation) {
        if self.video.is_none() {
            return;
        }
        // Layout follows the video's displayed orientation: a landscape video
        // (the normal case — frames are turned to match the reconstruction)
        // stacks BELOW the table; a portrait one sits beside it.
        let landscape = self
            .video
            .as_ref()
            .and_then(|v| v.current.as_ref())
            .map(|(_, t)| {
                let [w, h] = t.size();
                w >= h
            })
            .unwrap_or(true);
        if landscape {
            // Size the panel to the footage each frame: full available width at
            // the video's aspect ratio, capped at half the window. Set EXACTLY —
            // egui's per-id panel memory once collapsed this panel to its header
            // (the id had persisted state from its side-panel days), so nothing
            // is left to remembered sizes.
            let avail = ctx.available_rect();
            let aspect = self
                .video
                .as_ref()
                .and_then(|v| v.current.as_ref())
                .map(|(_, t)| {
                    let [w, h] = t.size();
                    h as f32 / w.max(1) as f32
                })
                .unwrap_or(0.5);
            let header = 34.0;
            let ideal = (avail.width() * aspect + header).min(avail.height() * 0.5);
            egui::TopBottomPanel::bottom("video_stack")
                .exact_height(ideal)
                .show(ctx, |ui| self.video_panel_body(ui, sim));
        } else {
            egui::SidePanel::right("video_side")
                .resizable(true)
                .default_width(380.0)
                .show(ctx, |ui| self.video_panel_body(ui, sim));
        }
    }

    fn video_panel_body(&mut self, ui: &mut egui::Ui, sim: &billiards_core::Simulation) {
        {
            ui.add_space(4.0);
            // One compact header row — heading, frame info, and toggles — so the
            // footage itself gets every remaining pixel of the panel.
            let frame_info = {
                let v = self.video.as_ref().unwrap();
                v.current.as_ref().map(|(idx, _)| format!("frame {idx} · {:.2}s", self.playhead))
            };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Actual video").strong());
                if let Some(info) = frame_info {
                    ui.label(egui::RichText::new(info).weak().size(11.0));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.overlay_on_video, "recon");
                    ui.checkbox(&mut self.overlay_tracked_on_video, "track");
                    if self.overlay_tracked_on_video {
                        ui.checkbox(&mut self.show_gap_fill, "gap-fill");
                    }
                });
            });

            let playhead = self.playhead;
            let overlay = self.overlay_on_video;
            let overlay_tracked = self.overlay_tracked_on_video;
            let v = self.video.as_ref().unwrap();
            let Some((idx, tex)) = &v.current else {
                ui.weak("loading frame…");
                return;
            };
            let _ = idx;

            let [iw, ih] = tex.size();
            let (iw, ih) = (iw as f32, ih as f32);
            let avail = ui.available_size();
            let scale = (avail.x / iw).min((avail.y - 6.0).max(1.0) / ih);
            let (resp, painter) = ui
                .vertical_centered(|ui| {
                    ui.allocate_painter(egui::vec2(iw * scale, ih * scale), egui::Sense::hover())
                })
                .inner;
            let rect = resp.rect;
            painter.image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Table (meters) -> a point on the displayed frame. The homography
            // targets the ORIGINAL frame; the display is flipped/turned to match
            // the reconstruction, so points pass through the same transform.
            let xform = v.xform;
            let (ow, oh) = if xform.1 % 2 == 1 { (ih as f64, iw as f64) } else { (iw as f64, ih as f64) };
            let project = move |h: &Homography, x: f64, y: f64| {
                let (u, w) = h.apply(x, y);
                let (u, w) = xform_px(u, w, ow, oh, xform);
                rect.min + egui::vec2((u as f32 / iw) * rect.width(), (w as f32 / ih) * rect.height())
            };

            // Tracking layer: the raw detected path (line) + a crosshair at the
            // playhead. If the marker leaves the real ball, the tracker is wrong.
            if overlay_tracked {
                if let (Some(h), Some(tracks)) = (&v.tab2img, &self.tracked) {
                    for (i, track) in tracks.iter().enumerate() {
                        let color = self.ball_color(i);
                        tracked_runs(
                            track,
                            |p| project(h, p.x, p.y),
                            |run| {
                                painter.add(egui::Shape::line(run, egui::Stroke::new(1.2, color.gamma_multiply(0.7))));
                            },
                        );
                        if self.show_gap_fill {
                            let bounds = self.table.center_bounds(self.ball.radius);
                            track_gaps(track, |prev, s, e, fdt, gdt| {
                                if let Some(bridge) = physics_bridge(prev, s, e, fdt, gdt, bounds) {
                                    let pts: Vec<egui::Pos2> =
                                        bridge.iter().map(|p| project(h, p.x, p.y)).collect();
                                    painter.add(egui::Shape::dashed_line(
                                        &pts,
                                        egui::Stroke::new(1.1, color.gamma_multiply(0.55)),
                                        5.0,
                                        4.0,
                                    ));
                                }
                            });
                        }
                        if let Some(p) = interp(track, playhead) {
                            let c = project(h, p.x, p.y);
                            let s = 6.0;
                            let stroke = egui::Stroke::new(1.8, color);
                            painter.line_segment([c - egui::vec2(s, 0.0), c + egui::vec2(s, 0.0)], stroke);
                            painter.line_segment([c - egui::vec2(0.0, s), c + egui::vec2(0.0, s)], stroke);
                        }
                    }
                }
            }

            // Reconstruction layer: the model's predicted positions, as rings.
            if overlay {
                if let Some(h) = &v.tab2img {
                    for i in 0..sim.trajectories.len() {
                        let c = project(h, sim.trajectories[i].state_at(playhead).pos.x, sim.trajectories[i].state_at(playhead).pos.y);
                        let color = self.ball_color(i);
                        painter.circle_stroke(c, 10.0, egui::Stroke::new(2.0, color));
                        painter.circle_stroke(c, 11.5, egui::Stroke::new(1.0, egui::Color32::from_gray(20)));
                    }
                }
            }

            // Legend lives inside the frame's top-left corner — no extra row.
            let mut legend = Vec::new();
            if overlay_tracked { legend.push("✚ tracking"); }
            if overlay { legend.push("◯ recon"); }
            if !legend.is_empty() {
                painter.text(
                    rect.left_top() + egui::vec2(6.0, 4.0),
                    egui::Align2::LEFT_TOP,
                    legend.join("  "),
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_rgb(200, 230, 255),
                );
            }
        }
    }

    fn table_panel(&mut self, ctx: &egui::Context, sim: &billiards_core::Simulation) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Header row for the reconstruction — mirrors the video pane's: the
            // fitted hit's numbers on the left, its display toggles on the right.
            if let Some(rms) = self.fit_rms {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Reconstructed hit").strong());
                    if rms > 0.45 {
                        ui.colored_label(
                            egui::Color32::from_rgb(230, 160, 70),
                            egui::RichText::new("⚠ low confidence").size(11.0),
                        )
                        .on_hover_text("the fit couldn't match the tracked shot well — treat this reconstruction as unreliable");
                    }
                    let (h, v) = self.action().tip_offset(self.ball.radius);
                    ui.label(
                        egui::RichText::new(format!(
                            "force {:.2} m/s · english {:+.0}%R follow {:+.0}%R · fit error {:.0} mm",
                            self.speed, h * 100.0, v * 100.0, rms * 1000.0,
                        ))
                        .weak()
                        .size(11.0),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.checkbox(&mut self.show_tracked, "actual (tracked)");
                        if self.show_tracked {
                            ui.checkbox(&mut self.show_gap_fill, "gap bridge");
                        }
                        ui.label(egui::RichText::new("filled = recon · ring = actual").weak().size(11.0));
                    });
                });
            }
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            let rect = response.rect;
            if self.view_mode == ViewMode::Actual && self.texture.is_some() {
                self.draw_actual(&painter, rect);
                return;
            }
            let view = View::fit(
                (rect.center().x, rect.center().y),
                (rect.width(), rect.height()),
                &self.table,
                44.0,
            );
            let ball_px = (self.ball.radius as f32 * view.scale).max(4.0);

            self.draw_table(&painter, &view);
            self.draw_trajectories(&painter, &view, sim);
            if !self.playing {
                self.draw_aim(&painter, &view);
            }
            self.draw_balls(&painter, &view, sim, ball_px);
            if self.show_tracked {
                self.draw_tracked(&painter, &view, ball_px);
            }
            self.handle_drag(&response, &view, ball_px);
        });
    }

    /// Overlay the actual tracked shot: each ball's tracked path (solid) and its
    /// actual position at the playhead (a ring). Filled ball = reconstructed;
    /// ring on top of it = reconstruction matches reality.
    fn draw_tracked(&self, painter: &egui::Painter, view: &View, ball_px: f32) {
        let Some(tracks) = &self.tracked else { return };
        for (i, track) in tracks.iter().enumerate() {
            let color = self.ball_color(i);
            tracked_runs(
                track,
                |p| {
                    let (sx, sy) = view.to_screen(p.x, p.y);
                    egui::pos2(sx, sy)
                },
                |run| {
                    painter.add(egui::Shape::line(run, egui::Stroke::new(2.0, color)));
                },
            );
            // Dashed physics prediction across each detection gap (display only).
            if self.show_gap_fill {
                let bounds = self.table.center_bounds(self.ball.radius);
                track_gaps(track, |prev, s, e, fdt, gdt| {
                    if let Some(bridge) = physics_bridge(prev, s, e, fdt, gdt, bounds) {
                        let pts: Vec<egui::Pos2> = bridge
                            .iter()
                            .map(|p| {
                                let (sx, sy) = view.to_screen(p.x, p.y);
                                egui::pos2(sx, sy)
                            })
                            .collect();
                        painter.add(egui::Shape::dashed_line(
                            &pts,
                            egui::Stroke::new(1.6, color.gamma_multiply(0.65)),
                            6.0,
                            5.0,
                        ));
                    }
                });
            }
            // Ring the tracked position only when it's known (hidden inside a gap).
            if let Some(p) = interp(track, self.playhead) {
                let (sx, sy) = view.to_screen(p.x, p.y);
                painter.circle_stroke(egui::pos2(sx, sy), ball_px + 2.5, egui::Stroke::new(2.0, color));
            }
        }
    }

    /// Draw the actual video frame with the reconstructed ball positions ringed on
    /// top — toggle this against the reconstructed view to confirm they match.
    fn draw_actual(&self, painter: &egui::Painter, rect: egui::Rect) {
        let tex = self.texture.as_ref().unwrap();
        let [iw, ih] = tex.size();
        let (iw, ih) = (iw as f32, ih as f32);
        let scale = (rect.width() / iw).min(rect.height() / ih);
        let img_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(iw * scale, ih * scale));
        painter.image(
            tex.id(),
            img_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        if let Some(h) = &self.tab2img {
            for (i, ball) in self.all_balls().iter().enumerate() {
                let (u, v) = h.apply(ball.x, ball.y);
                let p = img_rect.min + egui::vec2((u as f32 / iw) * img_rect.width(), (v as f32 / ih) * img_rect.height());
                let color = self.ball_color(i);
                painter.circle_stroke(p, 11.0, egui::Stroke::new(2.5, color));
                painter.circle_stroke(p, 12.5, egui::Stroke::new(1.0, egui::Color32::from_gray(20)));
                painter.circle_filled(p, 1.5, color);
            }
        }
        painter.text(
            rect.left_top() + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            "ACTUAL frame · rings = reconstructed positions",
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgb(120, 220, 255),
        );
    }

    fn draw_table(&self, painter: &egui::Painter, view: &View) {
        let hl = (self.table.length / 2.0) as f32 * view.scale;
        let hw = (self.table.width / 2.0) as f32 * view.scale;
        let center = egui::pos2(view.cx, view.cy);
        let cloth = egui::Rect::from_center_size(center, egui::vec2(2.0 * hl, 2.0 * hw));
        let rail = cloth.expand(18.0);

        painter.rect_filled(rail, egui::Rounding::same(6.0), egui::Color32::from_rgb(74, 46, 24));
        painter.rect_filled(cloth, egui::Rounding::ZERO, egui::Color32::from_rgb(27, 104, 66));

        // Diamonds on the rail band.
        let diamond = egui::Color32::from_rgb(232, 224, 188);
        for k in 1..8 {
            let x = -self.table.length / 2.0 + k as f64 * DIAMOND;
            for edge in [self.table.width / 2.0 + 0.045, -self.table.width / 2.0 - 0.045] {
                let (sx, sy) = view.to_screen(x, edge);
                painter.circle_filled(egui::pos2(sx, sy), 2.5, diamond);
            }
        }
        for k in 1..4 {
            let y = -self.table.width / 2.0 + k as f64 * DIAMOND;
            for edge in [self.table.length / 2.0 + 0.045, -self.table.length / 2.0 - 0.045] {
                let (sx, sy) = view.to_screen(edge, y);
                painter.circle_filled(egui::pos2(sx, sy), 2.5, diamond);
            }
        }
    }

    fn draw_trajectories(&self, painter: &egui::Painter, view: &View, sim: &billiards_core::Simulation) {
        for (i, traj) in sim.trajectories.iter().enumerate() {
            let pts = sample(traj, view, 0.015);
            if pts.len() >= 2 {
                let color = self.ball_color(i).gamma_multiply(0.6);
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.6, color)));
            }
        }
    }

    fn draw_aim(&self, painter: &egui::Painter, view: &View) {
        let cue = self.scene.cue;
        let (sx, sy) = view.to_screen(cue.x, cue.y);
        let start = egui::pos2(sx, sy);
        let a = self.aim_deg.to_radians();
        let len = (self.speed as f32 * 0.10 * view.scale).max(24.0);
        let dir = egui::vec2(a.cos() as f32, -(a.sin() as f32));
        let tip = start + dir * len;
        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 220, 255));
        // Dashed extension to the first rail contact, with the diamond coordinate
        // at the hit — the line a player would actually aim through.
        if let Some((hit, rail)) = aim_rail_hit(cue, a, self.table.center_bounds(self.ball.radius)) {
            let (hx, hy) = view.to_screen(hit.x, hit.y);
            let hp = egui::pos2(hx, hy);
            painter.add(egui::Shape::dashed_line(
                &[tip, hp],
                egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 220, 255).gamma_multiply(0.7)),
                7.0,
                6.0,
            ));
            painter.circle_stroke(hp, 5.0, egui::Stroke::new(1.8, egui::Color32::from_rgb(120, 220, 255)));
            let coord = match rail {
                0 | 1 => (hit.x + self.table.length / 2.0) / (self.table.length / 8.0),
                _ => (hit.y + self.table.width / 2.0) / (self.table.width / 4.0),
            };
            let off = match rail {
                0 => egui::vec2(0.0, -14.0),
                1 => egui::vec2(0.0, 14.0),
                2 => egui::vec2(14.0, 0.0),
                _ => egui::vec2(-14.0, 0.0),
            };
            painter.text(
                hp + off,
                egui::Align2::CENTER_CENTER,
                format!("{coord:.1}"),
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(160, 230, 255),
            );
        }
        painter.line_segment([start, tip], stroke);
        // Arrowhead.
        let perp = egui::vec2(-dir.y, dir.x);
        painter.line_segment([tip, tip - dir * 8.0 + perp * 5.0], stroke);
        painter.line_segment([tip, tip - dir * 8.0 - perp * 5.0], stroke);
    }

    fn draw_balls(&self, painter: &egui::Painter, view: &View, sim: &billiards_core::Simulation, ball_px: f32) {
        for i in 0..sim.trajectories.len() {
            let p = sim.trajectories[i].state_at(self.playhead).pos;
            let (sx, sy) = view.to_screen(p.x, p.y);
            let center = egui::pos2(sx, sy);
            painter.circle_filled(center, ball_px, self.ball_color(i));
            painter.circle_stroke(center, ball_px, egui::Stroke::new(1.4, egui::Color32::from_gray(25)));
        }
    }

    fn handle_drag(&mut self, response: &egui::Response, view: &View, ball_px: f32) {
        // Armed report marker: the next press on the table records the reported
        // ball's correct position instead of grabbing a ball.
        if self.report_marking {
            if response.drag_started() || response.clicked() {
                if let Some(p) = response.interact_pointer_pos() {
                    let (wx, wy) = view.to_world(p.x, p.y);
                    self.report_pos = Some(DVec3::new(wx, wy, self.ball.radius));
                    self.report_marking = false;
                }
            }
            return;
        }
        if response.drag_started() {
            self.playing = false;
            self.playhead = 0.0;
            if let Some(p) = response.interact_pointer_pos() {
                self.dragging = (0..1 + self.scene.objects.len())
                    .map(|i| {
                        let bp = self.ball_pos(i);
                        let (sx, sy) = view.to_screen(bp.x, bp.y);
                        (i, (egui::pos2(sx, sy) - p).length())
                    })
                    .filter(|(_, d)| *d <= ball_px * 1.8)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i);
            }
        }
        if response.dragged() {
            if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos()) {
                let (wx, wy) = view.to_world(p.x, p.y);
                let [min_x, max_x, min_y, max_y] = self.table.center_bounds(self.ball.radius);
                let pos = DVec3::new(wx.clamp(min_x, max_x), wy.clamp(min_y, max_y), self.ball.radius);
                self.set_ball_pos(i, pos);
                self.solution = None; // scene changed under it
                self.repair = None;
            }
        }
        if response.drag_stopped() {
            self.dragging = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_xform_aligns_vertical_inset_with_recon() {
        // The MASA 1080p inset: portrait crop, table length along image v, with
        // mirrored chirality relative to the editor's drawing convention.
        let corners = [(19.0, 19.0), (264.0, 15.0), (270.0, 492.0), (23.0, 492.0)];
        let h = table_to_image(corners, "vertical").unwrap();
        let xf = video_xform(&h);
        assert!(xf.1 % 2 == 1, "a portrait inset needs a quarter turn, got {xf:?}");
        // After the transform: +x right AND +y up on screen, like the recon.
        let (ow, oh) = (290.0, 515.0);
        let o = h.apply(0.0, 0.0);
        let dx = h.apply(0.4, 0.0);
        let dy = h.apply(0.0, 0.4);
        let a = xform_px(o.0, o.1, ow, oh, xf);
        let bx = xform_px(dx.0, dx.1, ow, oh, xf);
        let by = xform_px(dy.0, dy.1, ow, oh, xf);
        assert!(bx.0 > a.0, "+x should point right, got {xf:?}");
        assert!(by.1 < a.1, "+y should point up (screen v down), got {xf:?}");
    }

    #[test]
    fn parses_match_manifest() {
        let text = "{\n  \"video\": \"x.mp4\", \"fps\": 30, \"name\": \"masa4\",\n  \
                    \"games\": [\n    {\"id\": 0, \"dir\": \"game_00\", \"left\": \"EMRAH ERINCIK\", \
                    \"right\": \"SINAN YASAR\", \"t0\": 0.0, \"t1\": 474.0, \"n_shots\": 13, \"n_made\": 3},\n    \
                    {\"id\": 1, \"dir\": \"game_01\", \"left\": \"A B\", \"right\": \"C D\", \
                    \"n_shots\": 8, \"n_made\": null}\n  ]\n}";
        let games = parse_match_games(text);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].dir, "game_00");
        assert_eq!(games[0].left, "EMRAH ERINCIK");
        assert_eq!(games[0].right, "SINAN YASAR");
        assert_eq!(games[0].n_shots, 13);
        assert_eq!(games[0].n_made, Some(3));
        assert_eq!(games[1].dir, "game_01");
        assert_eq!(games[1].n_made, None, "null parses as no count");
    }

    #[test]
    fn parses_scene_file() {
        let text = "# billiards scene\nimage /tmp/f.png\ncorners 287,172 988,162 990,560 285,567\n\
                    orient horizontal\nwhite -1.13 -0.20\nyellow -0.75 -0.18\nred -1.19 -0.53\n";
        let imp = parse_scene(text, 0.03075).unwrap();
        assert!((imp.scene.cue.x + 1.13).abs() < 1e-9, "white is the cue");
        assert_eq!(imp.scene.objects.len(), 2);
        assert!((imp.scene.objects[0].x + 0.75).abs() < 1e-9, "yellow is object 0");
        assert!((imp.scene.objects[1].x + 1.19).abs() < 1e-9, "red is object 1");
        assert_eq!(imp.image_path.as_deref(), Some("/tmp/f.png"));
        assert!(imp.tab2img.is_some(), "homography built from corners");
    }

    #[test]
    fn parses_shot_csv() {
        // Yellow is the cue here; red is the (still) object ball. The cue must come
        // first and each ball must keep its true color.
        let text = "cue yellow\nyellow,0.00,-1.0,-0.3\nyellow,0.03,-0.9,-0.28\n\
                    red,0.00,0.5,0.2\nred,0.03,0.5,0.2\n";
        let shot = parse_shot(text, 0.03075).unwrap();
        assert_eq!(shot.tracks.len(), 2);
        assert_eq!(shot.tracks[0].len(), 2, "cue has 2 samples");
        assert!((shot.tracks[0][0].1.x + 1.0).abs() < 1e-9, "yellow cue is first");
        assert_eq!(shot.colors[0], BALL_COLORS[1], "cue drawn yellow, not white");
        assert_eq!(shot.colors[1], BALL_COLORS[2], "object drawn red");
        assert!(shot.video.is_none(), "no frames header -> no video link");
        // A .scene (space-separated) must NOT parse as a shot.
        assert!(parse_shot("white -1.0 -0.3\n", 0.03075).is_none());
    }

    #[test]
    fn parses_shot_with_video_link() {
        let text = "cue white\nframes /tmp/frames\nfps 15\nstart 42\n\
                    corners 287,172 988,162 990,560 285,567\norient vertical\ncolor,t,x,y\n\
                    white,0.00,-1.0,-0.3\nwhite,0.03,-0.9,-0.28\nred,0.00,0.5,0.2\nred,0.03,0.5,0.2\n";
        let shot = parse_shot(text, 0.03075).unwrap();
        let v = shot.video.expect("frames header -> video link");
        assert_eq!(v.frames_dir, "/tmp/frames");
        assert_eq!(v.fps, 15.0);
        assert_eq!(v.start, 42);
        assert!(v.tab2img.is_some(), "homography built from corners");
        // The header must not leak into the ball data.
        assert_eq!(shot.tracks.len(), 2);
        assert!((shot.tracks[0][0].1.x + 1.0).abs() < 1e-9, "white cue first");
    }

    #[test]
    fn interp_midpoint() {
        // Frame-spaced samples interpolate; a gap (dt > TRACK_GAP_S) returns None.
        let track = [(0.0, DVec3::new(0.0, 0.0, 0.0)), (0.06, DVec3::new(2.0, 0.0, 0.0))];
        assert!((interp(&track, 0.03).unwrap().x - 1.0).abs() < 1e-9);
        let gapped = [(0.0, DVec3::new(0.0, 0.0, 0.0)), (0.5, DVec3::new(2.0, 0.0, 0.0))];
        assert!(interp(&gapped, 0.25).is_none(), "no position inside a detection gap");
    }
}

/// Sample a trajectory into screen-space points at a fixed time step.
fn sample(traj: &Trajectory, view: &View, step: f64) -> Vec<egui::Pos2> {
    let total = traj.time_to_rest();
    let n = ((total / step).ceil() as usize).max(1);
    (0..=n)
        .map(|k| {
            let t = (k as f64 * step).min(total);
            let p = traj.state_at(t).pos;
            let (sx, sy) = view.to_screen(p.x, p.y);
            egui::pos2(sx, sy)
        })
        .collect()
}
