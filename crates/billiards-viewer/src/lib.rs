//! Browser viewer for published match bundles.
//!
//! The native app (`billiards-ui`) is the full editor; this is the shareable
//! read-only sibling: it fetches a *published bundle* (see `python/publish_bundle.py`)
//! over HTTP and runs the same physics engine + solver client-side (wasm), so
//! anyone with the link can browse a match — games, shots, make/miss, the
//! reconstruction, and solve/repair — without installing anything.
//!
//! Everything IO is async fetch (`ehttp`): responses land in a shared inbox the
//! UI polls each frame. No filesystem, no threads (the solver runs through its
//! sequential wasm shim with a trimmed search grid).

use std::sync::{Arc, Mutex};

use billiards_core::math::DVec3;
use billiards_core::{
    BallId, BallSpec, PhysicsParams, Scene, TableSpec, three_cushion_score,
};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action};
use billiards_solver::shotfile;
use billiards_solver::{Repair, RepairConfig, SolveConfig, Solution, repair, solve, success_probability};
use serde::Deserialize;

const BALL_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(240, 240, 240), // white
    egui::Color32::from_rgb(240, 210, 70),  // yellow
    egui::Color32::from_rgb(220, 60, 50),   // red
];

fn color_of(name: &str) -> egui::Color32 {
    match name {
        "white" => BALL_COLORS[0],
        "yellow" => BALL_COLORS[1],
        _ => BALL_COLORS[2],
    }
}

#[derive(Deserialize, Clone)]
struct GameEntry {
    dir: String,
    left: String,
    right: String,
    #[serde(default)]
    n_shots: i64,
    #[serde(default)]
    n_made: Option<i64>,
}

#[derive(Deserialize)]
struct MatchManifest {
    games: Vec<GameEntry>,
}

#[derive(Deserialize, Clone)]
struct ShotEntry {
    file: String,
    #[serde(default)]
    player: Option<String>,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    mp4: Option<String>,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Calibration {
    cushion_restitution: f64,
    cushion_friction: f64,
    mu_slide: f64,
    mu_roll: f64,
}

/// A fetched-and-parsed shot ready to draw.
struct LoadedShot {
    scene: Scene,
    observed: Vec<Vec<(f64, DVec3)>>,
    colors: Vec<egui::Color32>,
    action: billiards_core::CueAction,
    rms: f64,
    /// Footage: browser-decoded MP4 + the table→frame mapping for overlays.
    video: Option<webvideo::WebVideo>,
    homography: Option<billiards_vision::Homography>,
    /// Video time of shot t = 0 (the stroke) inside the clip.
    video_t0: f64,
}

/// Table→image homography from the `.shot` header's corner calibration.
fn table_to_image(corners: [(f64, f64); 4], orient: &str) -> Option<billiards_vision::Homography> {
    let (hl, hw) = (1.42, 0.71);
    let tc = if orient == "vertical" {
        [(-hl, hw), (-hl, -hw), (hl, -hw), (hl, hw)]
    } else {
        [(-hl, hw), (hl, hw), (hl, -hw), (-hl, -hw)]
    };
    billiards_vision::Homography::from_correspondences(tc, corners)
}

/// Async fetch results, filled by ehttp callbacks, drained by the UI.
#[derive(Default)]
struct Inbox {
    manifest: Option<Result<MatchManifest, String>>,
    shots_index: Option<Result<Vec<ShotEntry>, String>>,
    calibration: Option<Calibration>,
    shot_text: Option<Result<(usize, String), String>>,
}

/// Which of the three actions the cue controls are showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActionView {
    Shot,
    Repaired,
    Solved,
}

pub struct ViewerApp {
    base: String,
    inbox: Arc<Mutex<Inbox>>,
    games: Vec<GameEntry>,
    current_game: Option<usize>,
    shots: Vec<ShotEntry>,
    current_shot: Option<usize>,
    loading: bool,
    /// Measured height of the coach/controls block last frame — reserved
    /// ahead of the shot list so the controls always fit un-scrolled.
    coach_h: f32,
    error: Option<String>,

    table: TableSpec,
    ball: BallSpec,
    phys: PhysicsParams,
    loaded: Option<LoadedShot>,
    playhead: f64,
    playing: bool,
    play_speed: f64,
    show_tracked: bool,
    /// Live cue action (editable — sliders, strike diagram, as-if edits).
    aim_deg: f64,
    speed: f64,
    sidespin: f64,
    follow: f64,
    /// Which action the controls show (shot / repaired / solved).
    active_view: ActionView,
    /// Whether the loaded reconstruction scores (no "repaired" view for a make).
    shot_scored: bool,
    /// Make-probability of the player's line under execution noise.
    shot_prob: Option<f64>,
    /// Ball being dragged for an as-if scenario (0 = cue).
    dragging: Option<usize>,
    /// Reconstructed positions ringed on the footage.
    show_video_recon: bool,
    /// Tracker overlay on the FOOTAGE (paths + crosshairs); on by default —
    /// checking the tracker against reality is what the video panel is for.
    show_video_track: bool,
    solution: Option<Option<Solution>>,
    repair_found: Option<Option<Repair>>,
    /// MP4 URL for the shot currently being fetched (paired up on arrival).
    pending_mp4: Option<String>,
}

impl ViewerApp {
    pub fn new(base: String) -> Self {
        let app = Self {
            base,
            inbox: Arc::new(Mutex::new(Inbox::default())),
            games: Vec::new(),
            current_game: None,
            shots: Vec::new(),
            current_shot: None,
            loading: true,
            coach_h: 220.0,
            error: None,
            table: TableSpec::carom_match(),
            ball: BallSpec::carom(),
            phys: PhysicsParams::carom_calibrated(),
            loaded: None,
            playhead: 0.0,
            playing: false,
            play_speed: 1.0,
            show_tracked: false,
            aim_deg: 0.0,
            speed: 2.0,
            sidespin: 0.0,
            follow: 0.0,
            active_view: ActionView::Shot,
            shot_scored: false,
            shot_prob: None,
            dragging: None,
            show_video_recon: false,
            show_video_track: true,
            solution: None,
            repair_found: None,
            pending_mp4: None,
        };
        app.fetch_manifest();
        app
    }

    fn url(&self, rel: &str) -> String {
        format!("{}/{}", self.base.trim_end_matches('/'), rel)
    }

    fn fetch_manifest(&self) {
        let inbox = self.inbox.clone();
        ehttp::fetch(ehttp::Request::get(self.url("match.json")), move |res| {
            let out = res
                .map_err(|e| e.to_string())
                .and_then(|r| serde_json::from_slice(&r.bytes).map_err(|e| e.to_string()));
            inbox.lock().unwrap().manifest = Some(out);
        });
    }

    fn open_game(&mut self, gi: usize) {
        self.current_game = Some(gi);
        self.shots.clear();
        self.current_shot = None;
        self.loaded = None;
        self.loading = true;
        let dir = self.games[gi].dir.clone();
        let inbox = self.inbox.clone();
        ehttp::fetch(ehttp::Request::get(self.url(&format!("{dir}/shots.json"))), move |res| {
            let out = res
                .map_err(|e| e.to_string())
                .and_then(|r| serde_json::from_slice(&r.bytes).map_err(|e| e.to_string()));
            inbox.lock().unwrap().shots_index = Some(out);
        });
        let inbox = self.inbox.clone();
        ehttp::fetch(ehttp::Request::get(self.url(&format!("{dir}/calibration.json"))), move |res| {
            if let Ok(r) = res {
                if let Ok(c) = serde_json::from_slice::<Calibration>(&r.bytes) {
                    inbox.lock().unwrap().calibration = Some(c);
                }
            }
        });
    }

    fn open_shot(&mut self, si: usize) {
        let (Some(gi), Some(entry)) = (self.current_game, self.shots.get(si)) else { return };
        self.current_shot = Some(si);
        self.loading = true;
        self.solution = None;
        self.repair_found = None;
        self.pending_mp4 = entry
            .mp4
            .as_ref()
            .map(|m| self.url(&format!("{}/{}", self.games[gi].dir, m)));
        let url = self.url(&format!("{}/{}", self.games[gi].dir, entry.file));
        let inbox = self.inbox.clone();
        ehttp::fetch(ehttp::Request::get(url), move |res| {
            let out = res.map_err(|e| e.to_string()).and_then(|r| {
                String::from_utf8(r.bytes).map(|t| (si, t)).map_err(|e| e.to_string())
            });
            inbox.lock().unwrap().shot_text = Some(out);
        });
    }

    /// Drain async arrivals into app state.
    fn poll_inbox(&mut self, ctx: &egui::Context) {
        let mut inbox = self.inbox.lock().unwrap();
        if let Some(m) = inbox.manifest.take() {
            self.loading = false;
            match m {
                Ok(m) => self.games = m.games,
                Err(e) => self.error = Some(format!("match.json: {e}")),
            }
            ctx.request_repaint();
        }
        if let Some(s) = inbox.shots_index.take() {
            self.loading = false;
            match s {
                Ok(s) => {
                    self.shots = s;
                    if !self.shots.is_empty() {
                        drop(inbox);
                        self.open_shot(0);
                        return;
                    }
                }
                Err(e) => self.error = Some(format!("shots.json: {e}")),
            }
            ctx.request_repaint();
        }
        if let Some(c) = inbox.calibration.take() {
            self.phys = PhysicsParams {
                cushion_restitution: c.cushion_restitution,
                cushion_friction: c.cushion_friction,
                mu_slide: c.mu_slide,
                mu_roll: c.mu_roll,
                ..PhysicsParams::carom_calibrated()
            };
        }
        if let Some(t) = inbox.shot_text.take() {
            self.loading = false;
            match t {
                Ok((si, text)) if Some(si) == self.current_shot => {
                    drop(inbox);
                    self.apply_shot_text(&text);
                    ctx.request_repaint();
                    return;
                }
                Ok(_) => {} // stale (user clicked another shot meanwhile)
                Err(e) => self.error = Some(format!("shot: {e}")),
            }
            ctx.request_repaint();
        }
    }

    fn action(&self) -> billiards_core::CueAction {
        billiards_core::CueAction {
            aim: self.aim_deg.to_radians(),
            speed: self.speed,
            sidespin: self.sidespin,
            follow: self.follow,
        }
    }

    fn set_action(&mut self, a: &billiards_core::CueAction) {
        self.aim_deg = a.aim.to_degrees().rem_euclid(360.0);
        self.speed = a.speed;
        self.sidespin = a.sidespin;
        self.follow = a.follow;
    }

    /// Switch shot / repaired / solved, computing lazily on first use.
    fn show_view(&mut self, v: ActionView) {
        let Some(shot) = &self.loaded else { return };
        match v {
            ActionView::Repaired if self.repair_found.is_none() => {
                self.repair_found = Some(repair(
                    &shot.scene, &shot.action, &self.table, &self.ball, &self.phys,
                    &RepairConfig::default(),
                ));
            }
            ActionView::Solved if self.solution.is_none() => {
                let cfg = SolveConfig { aim_steps: 90, mc_samples: 32, ..SolveConfig::default() };
                self.solution = Some(solve(&shot.scene, &self.table, &self.ball, &self.phys, &cfg));
            }
            _ => {}
        }
        let a = match v {
            ActionView::Shot => Some(shot.action),
            ActionView::Repaired => self.repair_found.clone().flatten().map(|r| r.action),
            ActionView::Solved => self.solution.clone().flatten().map(|s| s.action),
        };
        if let Some(a) = a {
            self.active_view = v;
            self.set_action(&a);
            self.playing = false;
            self.playhead = 0.0;
        }
    }

    fn apply_shot_text(&mut self, text: &str) {
        let Some(parsed) = shotfile::parse(text, &self.table, self.ball.radius) else {
            self.error = Some("could not parse shot".into());
            return;
        };
        // Trimmed fit for single-threaded wasm (the full grid is a native luxury).
        let cfg = FitConfig {
            aim_window: 0.03,
            multistart: 8,
            refine_iters: 40,
            ..FitConfig::default()
        };
        let fit = fit_action(&parsed.scene, &parsed.observed, &self.table, &self.ball, &self.phys, &cfg);
        // Footage headers (corner calibration, clip timing) — shotfile ignores
        // them, so read them off the raw text here.
        let mut corners: Option<[(f64, f64); 4]> = None;
        let mut orient = "horizontal";
        let mut fps = 30.0_f64;
        let mut start = 0.0_f64;
        for line in text.lines() {
            let mut it = line.split_whitespace();
            match it.next().unwrap_or("") {
                "corners" => {
                    let pts: Vec<(f64, f64)> = it
                        .filter_map(|p| {
                            let mut c = p.split(',');
                            Some((c.next()?.parse().ok()?, c.next()?.parse().ok()?))
                        })
                        .collect();
                    if pts.len() == 4 {
                        corners = Some([pts[0], pts[1], pts[2], pts[3]]);
                    }
                }
                "orient" if it.next() == Some("vertical") => orient = "vertical",
                "fps" => fps = it.next().and_then(|v| v.parse().ok()).unwrap_or(30.0),
                "start" => start = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                _ => {}
            }
        }
        let homography = corners.and_then(|c| table_to_image(c, orient));
        let xform = homography.as_ref().map(video_xform).unwrap_or((false, 0));
        let video = self.pending_mp4.take().and_then(|u| webvideo::WebVideo::new(&u, xform));
        self.loaded = Some(LoadedShot {
            colors: parsed.order.iter().map(|c| color_of(c)).collect(),
            scene: parsed.scene,
            observed: parsed.observed,
            action: fit.action,
            rms: fit.rms_m,
            video,
            homography,
            video_t0: start / fps.max(1.0),
        });
        let loaded = self.loaded.as_ref().unwrap();
        let a = loaded.action;
        let sim = simulate(&loaded.scene.ball_states(&a), &self.table, &self.ball, &self.phys);
        self.shot_scored = three_cushion_score(&sim, BallId(0));
        // one-off (~0.3s in wasm): grade the player's line under execution noise
        let mc = SolveConfig { mc_samples: 48, ..SolveConfig::default() };
        self.shot_prob = Some(success_probability(
            &loaded.scene, &a, &self.table, &self.ball, &self.phys, &mc,
        ));
        self.set_action(&a);
        self.active_view = ActionView::Shot;
        self.playhead = 0.0;
        self.playing = true;
        self.error = None;
    }

    fn draw_table_view(&mut self, ui: &mut egui::Ui) {
        let Some(shot) = &self.loaded else {
            ui.centered_and_justified(|ui| {
                ui.label(if self.loading { "loading…" } else { "pick a game and shot" });
            });
            return;
        };
        // small clones so ball-dragging below can mutate self.loaded freely
        let scene = shot.scene.clone();
        let observed = shot.observed.clone();
        let colors = shot.colors.clone();
        let rms = shot.rms;
        let sim = simulate(&scene.ball_states(&self.action()), &self.table, &self.ball, &self.phys);
        let total = sim.settled_time();
        let scored = three_cushion_score(&sim, BallId(0));

        ui.horizontal(|ui| {
            let verdict = if scored {
                ("● THREE-CUSHION POINT", egui::Color32::from_rgb(90, 210, 120))
            } else {
                ("○ no score", egui::Color32::GRAY)
            };
            ui.colored_label(verdict.1, egui::RichText::new(verdict.0).strong());
            if let Some(p) = self.shot_prob {
                ui.label(egui::RichText::new(format!("~{:.0}% line", p * 100.0)).weak().size(11.0));
            }
            if rms > 0.45 {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 160, 70),
                    egui::RichText::new("⚠ low-confidence reconstruction").size(11.0),
                )
                .on_hover_text("the fit couldn't match the tracked shot well — treat this reconstruction as unreliable");
            }
            ui.label(
                egui::RichText::new(format!(
                    "aim {:.1}° · {:.2} m/s · fit {:.0} mm",
                    self.aim_deg, self.speed, rms * 1000.0
                ))
                .weak()
                .size(11.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_tracked, "tracked");
            });
        });

        // playback row
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "⏸" } else { "▶" }).clicked() {
                self.playing = !self.playing;
            }
            let mut ph = self.playhead;
            if ui.add(egui::Slider::new(&mut ph, 0.0..=total.max(0.01)).show_value(false)).changed() {
                self.playhead = ph;
                self.playing = false;
            }
            ui.label(egui::RichText::new(format!("{:.2}s / {total:.1}s", self.playhead)).weak().size(11.0));
            for (label, s) in [("⅛×", 0.125), ("¼×", 0.25), ("½×", 0.5), ("1×", 1.0), ("2×", 2.0)] {
                if ui.selectable_label(self.play_speed == s, label).clicked() {
                    self.play_speed = s;
                }
            }
        });

        let avail = ui.available_size();
        let (resp, painter) = ui.allocate_painter(avail, egui::Sense::click_and_drag());
        let rect = resp.rect;
        // fit the table into the rect
        let margin = 30.0_f32;
        let sx = (rect.width() - 2.0 * margin) / self.table.length as f32;
        let sy = (rect.height() - 2.0 * margin) / self.table.width as f32;
        let scale = sx.min(sy);
        let c = rect.center();
        let to_screen = |x: f64, y: f64| {
            egui::pos2(c.x + x as f32 * scale, c.y - y as f32 * scale)
        };

        // cloth + rail
        let hl = self.table.length as f32 / 2.0 * scale;
        let hw = self.table.width as f32 / 2.0 * scale;
        let cloth = egui::Rect::from_center_size(c, egui::vec2(2.0 * hl, 2.0 * hw));
        painter.rect_filled(cloth.expand(14.0), 4.0, egui::Color32::from_rgb(74, 46, 24));
        painter.rect_filled(cloth, 0.0, egui::Color32::from_rgb(27, 104, 66));
        // rail diamonds — the reference marks players aim by
        let diamond = egui::Color32::from_rgb(232, 224, 188);
        let l = self.table.length;
        let w = self.table.width;
        for k in 1..8 {
            let x = -l / 2.0 + k as f64 * l / 8.0;
            for ey in [w / 2.0 + 0.045, -w / 2.0 - 0.045] {
                painter.circle_filled(to_screen(x, ey), 2.5, diamond);
            }
        }
        for k in 1..4 {
            let y = -w / 2.0 + k as f64 * w / 4.0;
            for ex in [l / 2.0 + 0.045, -l / 2.0 - 0.045] {
                painter.circle_filled(to_screen(ex, y), 2.5, diamond);
            }
        }

        // simulated trajectories
        for (i, traj) in sim.trajectories.iter().enumerate() {
            let col = colors.get(i).copied().unwrap_or(egui::Color32::GRAY).gamma_multiply(0.6);
            let mut pts = Vec::new();
            let mut t = 0.0;
            while t <= total {
                let p = traj.state_at(t).pos;
                pts.push(to_screen(p.x, p.y));
                t += 0.02;
            }
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.6, col)));
            }
        }
        // tracked overlay (runs broken at gaps)
        if self.show_tracked {
            for (i, track) in observed.iter().enumerate() {
                let col = colors.get(i).copied().unwrap_or(egui::Color32::GRAY);
                let mut run: Vec<egui::Pos2> = Vec::new();
                let mut last_t: Option<f64> = None;
                for &(t, p) in track {
                    if let Some(pt) = last_t {
                        if t - pt > 0.1 {
                            if run.len() >= 2 {
                                painter.add(egui::Shape::line(std::mem::take(&mut run), egui::Stroke::new(1.2, col)));
                            }
                            run.clear();
                        }
                    }
                    run.push(to_screen(p.x, p.y));
                    last_t = Some(t);
                }
                if run.len() >= 2 {
                    painter.add(egui::Shape::line(run, egui::Stroke::new(1.2, col)));
                }
            }
        }
        // dashed physics bridges across tracking gaps (display-only predictions)
        if self.show_tracked {
            let bounds = self.table.center_bounds(self.ball.radius);
            for (i, track) in observed.iter().enumerate() {
                let col = colors.get(i).copied().unwrap_or(egui::Color32::GRAY);
                track_gaps(track, |prev, s0, e, fdt, gdt| {
                    if let Some(bridge) = physics_bridge(prev, s0, e, fdt, gdt, bounds) {
                        let pts: Vec<egui::Pos2> =
                            bridge.iter().map(|p| to_screen(p.x, p.y)).collect();
                        painter.add(egui::Shape::dashed_line(
                            &pts,
                            egui::Stroke::new(1.4, col.gamma_multiply(0.65)),
                            6.0,
                            5.0,
                        ));
                    }
                });
            }
        }

        // aim arrow while paused, extended to its first rail contact so the
        // player can read the line against the diamonds
        if !self.playing {
            let cue = scene.cue;
            let start = to_screen(cue.x, cue.y);
            let a = self.aim_deg.to_radians();
            let len = (self.speed as f32 * 0.10 * scale).max(24.0);
            let dir = egui::vec2(a.cos() as f32, -(a.sin() as f32));
            let tip = start + dir * len;
            let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 220, 255));
            if let Some((hit, rail)) = aim_rail_hit(cue, a, self.table.center_bounds(self.ball.radius)) {
                let hp = to_screen(hit.x, hit.y);
                painter.add(egui::Shape::dashed_line(
                    &[tip, hp],
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 220, 255).gamma_multiply(0.7)),
                    7.0,
                    6.0,
                ));
                painter.circle_stroke(hp, 5.0, egui::Stroke::new(1.8, egui::Color32::from_rgb(120, 220, 255)));
                // diamond coordinate along that rail (0 at the left/bottom corner)
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
            let perp = egui::vec2(-dir.y, dir.x);
            painter.line_segment([tip, tip - dir * 8.0 + perp * 5.0], stroke);
            painter.line_segment([tip, tip - dir * 8.0 - perp * 5.0], stroke);
        }

        // balls at the playhead
        let ball_px = (self.ball.radius as f32 * scale).max(3.5);
        for (i, traj) in sim.trajectories.iter().enumerate() {
            let p = traj.state_at(self.playhead).pos;
            let col = colors.get(i).copied().unwrap_or(egui::Color32::GRAY);
            painter.circle_filled(to_screen(p.x, p.y), ball_px, col);
            painter.circle_stroke(to_screen(p.x, p.y), ball_px, egui::Stroke::new(1.2, egui::Color32::from_gray(25)));
        }

        // drag balls for as-if scenarios (from-world inverse of to_screen)
        let to_world = |p: egui::Pos2| -> (f64, f64) {
            (((p.x - c.x) / scale) as f64, (-(p.y - c.y) / scale) as f64)
        };
        if resp.drag_started() {
            self.playing = false;
            self.playhead = 0.0;
            if let Some(p) = resp.interact_pointer_pos() {
                let n = 1 + scene.objects.len();
                self.dragging = (0..n)
                    .map(|i| {
                        let bp = if i == 0 { scene.cue } else { scene.objects[i - 1] };
                        (i, (to_screen(bp.x, bp.y) - p).length())
                    })
                    .filter(|(_, d)| *d < ball_px + 12.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i);
            }
        }
        if resp.dragged() {
            if let (Some(i), Some(p)) = (self.dragging, resp.interact_pointer_pos()) {
                let (wx, wy) = to_world(p);
                let [min_x, max_x, min_y, max_y] = self.table.center_bounds(self.ball.radius);
                let pos = DVec3::new(wx.clamp(min_x, max_x), wy.clamp(min_y, max_y), self.ball.radius);
                if let Some(shot) = &mut self.loaded {
                    if i == 0 {
                        shot.scene.cue = pos;
                    } else if let Some(o) = shot.scene.objects.get_mut(i - 1) {
                        *o = pos;
                    }
                }
                // scene changed under them — stale searches must not linger
                self.solution = None;
                self.repair_found = None;
            }
        }
        if resp.drag_stopped() {
            self.dragging = None;
        }

        if self.playing && total > 0.0 && !self.video_is_master() {
            self.playhead += ui.input(|i| i.stable_dt) as f64 * self.play_speed;
            if self.playhead >= total {
                self.playhead = 0.0;
            }
            ui.ctx().request_repaint();
        }
    }

    /// Is the browser <video> the playback clock? (While it plays, the playhead
    /// follows its currentTime, so footage and simulation stay in lockstep.)
    fn video_is_master(&self) -> bool {
        self.loaded.as_ref().and_then(|s| s.video.as_ref()).is_some_and(|v| v.ready())
    }

    /// Bottom panel: the shot's footage, aligned with the reconstruction, with
    /// tracked-position crosshairs projected through the same transform.
    fn video_panel(&mut self, ctx: &egui::Context) {
        let playing = self.playing;
        let speed = self.play_speed;
        let playhead0 = self.playhead;
        // reconstructed positions at the playhead (rings on the footage)
        let recon_pts: Vec<DVec3> = if self.show_video_recon {
            self.loaded
                .as_ref()
                .map(|shot| {
                    let sim = simulate(
                        &shot.scene.ball_states(&self.action()),
                        &self.table,
                        &self.ball,
                        &self.phys,
                    );
                    sim.trajectories.iter().map(|t| t.state_at(self.playhead).pos).collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut new_playhead = None;
        {
            let Some(shot) = &mut self.loaded else { return };
            let t0 = shot.video_t0;
            let Some(video) = &mut shot.video else { return };
            if video.ready() {
                if playing {
                    new_playhead = Some((video.current_time() - t0).max(0.0));
                }
                video.sync(t0 + new_playhead.unwrap_or(playhead0), playing, speed);
            }
        }
        if let Some(p) = new_playhead {
            self.playhead = p;
        }
        let playhead = self.playhead;

        let avail = ctx.available_rect();
        let shot = self.loaded.as_mut().unwrap();
        let homography = &shot.homography;
        let observed = &shot.observed;
        let colors = &shot.colors;
        let video = shot.video.as_mut().unwrap();
        let (ow, oh) = video.out_dims();
        let tex = video.texture(ctx);
        let header_h = 26.0;
        let ideal = if ow > 0 {
            (avail.width() * oh as f32 / ow as f32 + header_h).min(avail.height() * 0.45)
        } else {
            120.0
        };
        let mut vt = self.show_video_track;
        let mut vr = self.show_video_recon;
        egui::TopBottomPanel::bottom("video")
            .exact_height(ideal)
            .show(ctx, |ui| {
                // the toggles live WITH the footage they control
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Actual video").strong().size(12.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.checkbox(&mut vr, "◯ recon");
                        ui.checkbox(&mut vt, "✚ track");
                    });
                });
                let (Some(tex), true) = (tex, ow > 0) else {
                    ui.centered_and_justified(|ui| ui.weak("footage loading…"));
                    return;
                };
                let availv = ui.available_size();
                let scale = (availv.x / ow as f32).min(availv.y / oh as f32);
                let size = egui::vec2(ow as f32 * scale, oh as f32 * scale);
                let (resp, painter) = ui
                    .vertical_centered(|ui| ui.allocate_painter(size, egui::Sense::hover()))
                    .inner;
                let rect = resp.rect;
                painter.image(
                    tex,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                // tracked overlay, through the same table→frame→display path:
                // per-ball path runs (broken at detection gaps) + a crosshair
                if vt {
                    if let Some(h) = homography {
                        let (sw, sh) = video.src_dims();
                        let project = |p: &DVec3| {
                            let (u, v) = h.apply(p.x, p.y);
                            let (u, v) = xform_px(u, v, sw as f64, sh as f64, video.xform);
                            rect.min
                                + egui::vec2(
                                    (u as f32 / ow as f32) * rect.width(),
                                    (v as f32 / oh as f32) * rect.height(),
                                )
                        };
                        for (i, track) in observed.iter().enumerate() {
                            let col = shot_color(colors, i);
                            let mut run: Vec<egui::Pos2> = Vec::new();
                            let mut last_t: Option<f64> = None;
                            for (t, p) in track {
                                if let Some(pt) = last_t {
                                    if t - pt > 0.1 {
                                        if run.len() >= 2 {
                                            painter.add(egui::Shape::line(
                                                std::mem::take(&mut run),
                                                egui::Stroke::new(1.1, col.gamma_multiply(0.7)),
                                            ));
                                        }
                                        run.clear();
                                    }
                                }
                                run.push(project(p));
                                last_t = Some(*t);
                            }
                            if run.len() >= 2 {
                                painter.add(egui::Shape::line(run, egui::Stroke::new(1.1, col.gamma_multiply(0.7))));
                            }
                        }
                        for (i, track) in observed.iter().enumerate() {
                            let Some(p) = interp_track(track, playhead) else { continue };
                            let (u, v) = h.apply(p.x, p.y);
                            let (u, v) = xform_px(u, v, sw as f64, sh as f64, video.xform);
                            let c = rect.min
                                + egui::vec2(
                                    (u as f32 / ow as f32) * rect.width(),
                                    (v as f32 / oh as f32) * rect.height(),
                                );
                            let col = colors.get(i).copied().unwrap_or(egui::Color32::GRAY);
                            let s = egui::Stroke::new(1.8, col);
                            painter.line_segment([c - egui::vec2(6.0, 0.0), c + egui::vec2(6.0, 0.0)], s);
                            painter.line_segment([c - egui::vec2(0.0, 6.0), c + egui::vec2(0.0, 6.0)], s);
                        }
                    }
                }
                // reconstruction rings (model positions) on the footage
                if vr {
                    if let Some(h) = homography {
                        let (sw, sh) = video.src_dims();
                        for (i, p) in recon_pts.iter().enumerate() {
                            let (u, v) = h.apply(p.x, p.y);
                            let (u, v) = xform_px(u, v, sw as f64, sh as f64, video.xform);
                            let c = rect.min
                                + egui::vec2(
                                    (u as f32 / ow as f32) * rect.width(),
                                    (v as f32 / oh as f32) * rect.height(),
                                );
                            let col = shot_color(colors, i);
                            painter.circle_stroke(c, 9.0, egui::Stroke::new(2.0, col));
                            painter.circle_stroke(c, 10.5, egui::Stroke::new(1.0, egui::Color32::from_gray(20)));
                        }
                    }
                }
            });
        self.show_video_track = vt;
        self.show_video_recon = vr;
        if playing {
            ctx.request_repaint(); // stream frames
        }
    }

    fn coach_panel(&mut self, ui: &mut egui::Ui) {
        if self.loaded.is_none() {
            return;
        }
        ui.separator();

        // cue action controls (live — the table view simulates these)
        ui.label(egui::RichText::new("Cue action").strong());
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.aim_deg, 0.0..=360.0).text("aim °"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.speed, 0.5..=6.5).text("m/s"))
            .changed();
        if changed {
            self.solution = None;
            self.repair_found = None;
            self.playing = false;
            self.playhead = 0.0;
        }
        self.strike_diagram(ui);

        ui.separator();
        // shot / repaired / solved — computed lazily, never destructive
        let mut v = self.active_view;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut v, ActionView::Shot, "shot");
            if !self.shot_scored {
                ui.selectable_value(&mut v, ActionView::Repaired, "repaired");
            }
            ui.selectable_value(&mut v, ActionView::Solved, "solved");
        });
        if v != self.active_view {
            self.show_view(v);
        }
        match self.active_view {
            ActionView::Solved => {
                if let Some(Some(sol)) = &self.solution {
                    ui.label(egui::RichText::new("Best shot").strong());
                    ui.label(format!(
                        "success {:.0}% · {} ({:.2})",
                        sol.success_prob * 100.0,
                        sol.category(),
                        sol.difficulty()
                    ));
                    ui.label(
                        egui::RichText::new(format!("{} scoring options", sol.scoring_cells))
                            .weak()
                            .size(11.0),
                    );
                } else if matches!(self.solution, Some(None)) {
                    ui.weak("no scoring shot found for this layout");
                }
            }
            ActionView::Repaired => {
                if let Some(Some(rep)) = &self.repair_found {
                    if rep.already_scores {
                        ui.weak("already scores");
                    } else {
                        ui.label(egui::RichText::new("Smallest fix to score:").strong());
                        for line in repair_advice(rep) {
                            ui.label(format!("  • {line}"));
                        }
                        ui.label(
                            egui::RichText::new(format!("→ scores ~{:.0}%", rep.success_prob * 100.0))
                                .weak(),
                        );
                    }
                } else if matches!(self.repair_found, Some(None)) {
                    ui.colored_label(egui::Color32::from_rgb(224, 150, 90), "✗ no nearby scoring shot");
                }
            }
            ActionView::Shot => {
                ui.label(
                    egui::RichText::new("shot = reconstruction · repaired = smallest fix · solved = best shot")
                        .weak()
                        .size(11.0),
                );
            }
        }
    }

    /// Cue-tip strike diagram: drag to place english/follow; power bar shows speed.
    fn strike_diagram(&mut self, ui: &mut egui::Ui) {
        let size = egui::vec2(ui.available_width().min(200.0), 132.0);
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
        let rect = resp.rect;
        let face_r = (rect.height() * 0.5 - 8.0).min(rect.width() * 0.5 - 26.0);
        let face_c = egui::pos2(rect.left() + face_r + 6.0, rect.center().y);
        let radius = self.ball.radius;

        painter.circle_filled(face_c, face_r, egui::Color32::from_rgb(244, 244, 236));
        painter.circle_stroke(face_c, face_r, egui::Stroke::new(1.5, egui::Color32::from_gray(70)));
        let faint = egui::Stroke::new(1.0, egui::Color32::from_gray(205));
        painter.line_segment([face_c - egui::vec2(face_r, 0.0), face_c + egui::vec2(face_r, 0.0)], faint);
        painter.line_segment([face_c - egui::vec2(0.0, face_r), face_c + egui::vec2(0.0, face_r)], faint);
        painter.circle_stroke(face_c, face_r * 0.5, egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 170, 120)));

        if (resp.dragged() || resp.clicked()) && !self.playing {
            if let Some(p) = resp.interact_pointer_pos() {
                let mut h = ((p.x - face_c.x) / face_r) as f64;
                let mut v = (-(p.y - face_c.y) / face_r) as f64;
                let mag = (h * h + v * v).sqrt();
                if mag > 0.9 {
                    h *= 0.9 / mag;
                    v *= 0.9 / mag;
                }
                let a = billiards_core::CueAction::from_tip_offset(
                    self.aim_deg.to_radians(),
                    self.speed,
                    h,
                    v,
                    radius,
                );
                self.sidespin = a.sidespin;
                self.follow = a.follow;
                self.solution = None;
                self.repair_found = None;
            }
        }

        let (fh, fv) = self.action().tip_offset(radius);
        let mag = (fh * fh + fv * fv).sqrt();
        let (dh, dv) = if mag > 1.0 { (fh / mag, fv / mag) } else { (fh, fv) };
        let dot = face_c + egui::vec2(dh as f32 * face_r, -(dv as f32) * face_r);
        let miscue = mag > 0.5;
        let dot_col = if miscue {
            egui::Color32::from_rgb(220, 80, 60)
        } else {
            egui::Color32::from_rgb(70, 160, 230)
        };
        painter.circle_filled(dot, 6.0, dot_col);
        painter.circle_stroke(dot, 6.0, egui::Stroke::new(1.0, egui::Color32::from_gray(30)));

        let frac = (((self.speed - 0.5) / 6.0).clamp(0.0, 1.0)) as f32;
        let bx = rect.right() - 8.0;
        let (top, bot) = (rect.top() + 8.0, rect.bottom() - 8.0);
        let track = egui::Rect::from_min_max(egui::pos2(bx - 5.0, top), egui::pos2(bx + 5.0, bot));
        painter.rect_filled(track, egui::Rounding::same(3.0), egui::Color32::from_gray(60));
        let fill = egui::Rect::from_min_max(
            egui::pos2(bx - 5.0, bot - frac * (bot - top)),
            egui::pos2(bx + 5.0, bot),
        );
        painter.rect_filled(fill, egui::Rounding::same(3.0), egui::Color32::from_rgb(120, 200, 255));

        ui.label(
            egui::RichText::new(format!("english {:+.0}%R · follow {:+.0}%R", fh * 100.0, fv * 100.0))
                .size(11.0),
        );
        if miscue {
            ui.colored_label(egui::Color32::from_rgb(220, 120, 90), "⚠ past ½R — miscue");
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_inbox(ctx);

        egui::SidePanel::left("nav").default_width(230.0).show(ctx, |ui| {
            ui.add_space(6.0);
            ui.heading("Billiards Coach");
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(220, 120, 90), e);
            }
            ui.separator();
            ui.label(egui::RichText::new("Games").strong());
            if self.games.is_empty() && self.error.is_none() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.weak(if self.loading { "fetching games…" } else { "no games in bundle" });
                });
                if ui.small_button("retry").clicked() {
                    self.loading = true;
                    self.fetch_manifest();
                }
            }
            let mut open_game = None;
            for (gi, g) in self.games.iter().enumerate() {
                let made = g.n_made.map(|m| format!(" · {m}/{} made", g.n_shots)).unwrap_or_default();
                if ui
                    .selectable_label(self.current_game == Some(gi), format!("{} vs {}{}", g.left, g.right, made))
                    .clicked()
                {
                    open_game = Some(gi);
                }
            }
            if let Some(gi) = open_game {
                self.open_game(gi);
            }
            ui.separator();
            // The controls below get first claim on the panel's height (their
            // measured height from last frame); the shot list — 100+ entries
            // for a full game — lives in whatever remains and scrolls.
            let list_max = (ui.available_height() - self.coach_h - 16.0).max(60.0);
            let mut open_shot = None;
            egui::ScrollArea::vertical()
                .id_salt("shot_list")
                .max_height(list_max)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                // One column per player, each holding that player's shots in
                // order — a scoresheet reading. Shots without an attributed
                // side sit in the left column marked "?".
                let (lname, rname) = self
                    .current_game
                    .and_then(|gi| self.games.get(gi))
                    .map(|g| (g.left.clone(), g.right.clone()))
                    .unwrap_or_else(|| ("left".into(), "right".into()));
                let glyph = |ui: &mut egui::Ui, result: Option<&str>| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    let c = r.center();
                    match result {
                        Some("make") => {
                            let s = egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 200, 110));
                            ui.painter().line_segment([c + egui::vec2(-4.0, 0.0), c + egui::vec2(-1.0, 3.5)], s);
                            ui.painter().line_segment([c + egui::vec2(-1.0, 3.5), c + egui::vec2(4.5, -3.5)], s);
                        }
                        Some("miss") => {
                            let s = egui::Stroke::new(2.0, egui::Color32::from_rgb(210, 90, 90));
                            ui.painter().line_segment([c + egui::vec2(-3.5, -3.5), c + egui::vec2(3.5, 3.5)], s);
                            ui.painter().line_segment([c + egui::vec2(-3.5, 3.5), c + egui::vec2(3.5, -3.5)], s);
                        }
                        _ => {}
                    }
                };
                ui.columns(2, |cols| {
                    for (col, name) in [(0usize, &lname), (1, &rname)] {
                        cols[col].add(
                            egui::Label::new(egui::RichText::new(name).strong().small())
                                .truncate(),
                        );
                    }
                    for (si, s) in self.shots.iter().enumerate() {
                        let (col, tag) = match s.player.as_deref() {
                            Some("left") => (0, format!("shot {si}")),
                            Some("right") => (1, format!("shot {si}")),
                            _ => (0, format!("shot {si} ?")),
                        };
                        cols[col].horizontal(|ui| {
                            if ui
                                .selectable_label(self.current_shot == Some(si), tag)
                                .clicked()
                            {
                                open_shot = Some(si);
                            }
                            glyph(ui, s.result.as_deref());
                        });
                    }
                });
            });
            if let Some(si) = open_shot {
                self.open_shot(si);
            }
            let r = ui.scope(|ui| self.coach_panel(ui)).response.rect;
            self.coach_h = r.height();
        });

        self.video_panel(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_table_view(ui);
        });
    }
}

// ---- browser video --------------------------------------------------------
//
// The bundle carries one small MP4 per shot. The browser decodes it in a hidden
// <video>; each egui frame we draw the current video frame through an offscreen
// 2D canvas — applying the same flip/quarter-turn that aligns the footage with
// the reconstruction — and upload the pixels as an egui texture. While playing,
// the <video> clock is the master (playhead follows it); scrubbing seeks it.

/// Quarter-turn+flip aligning footage with the reconstruction (+x right, +y up).
fn video_xform(h: &billiards_vision::Homography) -> (bool, u8) {
    let o = h.apply(0.0, 0.0);
    let dx = h.apply(0.4, 0.0);
    let dy = h.apply(0.0, 0.4);
    let (dxu, dxv) = (dx.0 - o.0, dx.1 - o.1);
    let (dyu, dyv) = (dy.0 - o.0, dy.1 - o.1);
    let nx = (dxu * dxu + dxv * dxv).sqrt().max(1e-9);
    let ny = (dyu * dyu + dyv * dyv).sqrt().max(1e-9);
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
            let score = x.0 / nx - y.1 / ny;
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

#[cfg(target_arch = "wasm32")]
mod webvideo {
    use wasm_bindgen::JsCast;

    pub struct WebVideo {
        video: web_sys::HtmlVideoElement,
        canvas: web_sys::HtmlCanvasElement,
        ctx: web_sys::CanvasRenderingContext2d,
        pub xform: (bool, u8),
        texture: Option<egui::TextureHandle>,
    }

    impl WebVideo {
        pub fn new(url: &str, xform: (bool, u8)) -> Option<Self> {
            let doc = web_sys::window()?.document()?;
            let video: web_sys::HtmlVideoElement =
                doc.create_element("video").ok()?.dyn_into().ok()?;
            video.set_src(url);
            video.set_muted(true);
            video.set_preload("auto");
            let _ = video.set_attribute("playsinline", "");
            let _ = video.style().set_property("display", "none");
            doc.body()?.append_child(&video).ok()?;
            let canvas: web_sys::HtmlCanvasElement =
                doc.create_element("canvas").ok()?.dyn_into().ok()?;
            let ctx: web_sys::CanvasRenderingContext2d =
                canvas.get_context("2d").ok()??.dyn_into().ok()?;
            Some(Self { video, canvas, ctx, xform, texture: None })
        }

        pub fn ready(&self) -> bool {
            self.video.ready_state() >= 2 && self.video.video_width() > 0
        }

        /// Source (original) frame dims.
        pub fn src_dims(&self) -> (u32, u32) {
            (self.video.video_width(), self.video.video_height())
        }

        /// Displayed (transformed) dims.
        pub fn out_dims(&self) -> (u32, u32) {
            let (w, h) = self.src_dims();
            if self.xform.1 % 2 == 1 { (h, w) } else { (w, h) }
        }

        pub fn current_time(&self) -> f64 {
            self.video.current_time()
        }

        pub fn sync(&self, target: f64, playing: bool, rate: f64) {
            if playing {
                self.video.set_playback_rate(rate);
                if self.video.paused() {
                    let _ = self.video.play();
                }
                // only correct on real drift (a seek mid-play stutters)
                if (self.video.current_time() - target).abs() > 0.4 {
                    self.video.set_current_time(target);
                }
            } else {
                if !self.video.paused() {
                    let _ = self.video.pause();
                }
                if (self.video.current_time() - target).abs() > 0.04 {
                    self.video.set_current_time(target);
                }
            }
        }

        /// Pull the current frame through the aligning transform into a texture.
        pub fn texture(&mut self, ctx_egui: &egui::Context) -> Option<egui::TextureId> {
            if !self.ready() {
                return self.texture.as_ref().map(|t| t.id());
            }
            let (sw, sh) = self.src_dims();
            let (ow, oh) = self.out_dims();
            self.canvas.set_width(ow);
            self.canvas.set_height(oh);
            // Derive the canvas transform from xform_px on basis points.
            let f = |u: f64, v: f64| super::xform_px(u, v, sw as f64, sh as f64, self.xform);
            let o = f(0.0, 0.0);
            let bx = f(1.0, 0.0);
            let by = f(0.0, 1.0);
            let _ = self.ctx.set_transform(bx.0 - o.0, bx.1 - o.1, by.0 - o.0, by.1 - o.1, o.0, o.1);
            let _ = self.ctx.draw_image_with_html_video_element(&self.video, 0.0, 0.0);
            let _ = self.ctx.reset_transform();
            let data = self.ctx.get_image_data(0.0, 0.0, ow as f64, oh as f64).ok()?.data();
            let img = egui::ColorImage::from_rgba_unmultiplied([ow as usize, oh as usize], &data);
            match &mut self.texture {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => {
                    self.texture =
                        Some(ctx_egui.load_texture("webvideo", img, egui::TextureOptions::LINEAR));
                }
            }
            self.texture.as_ref().map(|t| t.id())
        }
    }

    impl Drop for WebVideo {
        fn drop(&mut self) {
            let _ = self.video.pause();
            self.video.remove();
            self.canvas.remove();
        }
    }
}

/// Native stub so the shared code compiles; the native harness has no video.
#[cfg(not(target_arch = "wasm32"))]
mod webvideo {
    pub struct WebVideo {
        pub xform: (bool, u8),
    }
    impl WebVideo {
        pub fn new(_url: &str, xform: (bool, u8)) -> Option<Self> {
            Some(Self { xform })
        }
        pub fn ready(&self) -> bool {
            false
        }
        pub fn src_dims(&self) -> (u32, u32) {
            (0, 0)
        }
        pub fn out_dims(&self) -> (u32, u32) {
            (0, 0)
        }
        pub fn current_time(&self) -> f64 {
            0.0
        }
        pub fn sync(&self, _t: f64, _p: bool, _r: f64) {}
        pub fn texture(&mut self, _ctx: &egui::Context) -> Option<egui::TextureId> {
            None
        }
    }
}

// ---- web entry point ------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod web {
    use super::ViewerApp;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn start() {
        // Bundle lives next to the page: <origin>/<path>/bundle
        let base = web_sys::window()
            .and_then(|w| w.location().href().ok())
            .map(|href| {
                let root = href.split(['?', '#']).next().unwrap_or(&href);
                format!("{}bundle", root.trim_end_matches("index.html"))
            })
            .unwrap_or_else(|| "bundle".into());
        let options = eframe::WebOptions::default();
        wasm_bindgen_futures::spawn_local(async move {
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("viewer_canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .expect("canvas #viewer_canvas not found");
            eframe::WebRunner::new()
                .start(
                    canvas,
                    options,
                    Box::new(move |_cc| Ok(Box::new(ViewerApp::new(base)))),
                )
                .await
                .expect("failed to start viewer");
        });
    }
}


/// Invoke `f(prev, s, e, frame_dt, gap_dt)` for each detection gap in a track
/// (ported from the native editor; display-only physics predictions).
fn track_gaps(track: &[(f64, DVec3)], mut f: impl FnMut(DVec3, DVec3, DVec3, f64, f64)) {
    for i in 2..track.len() {
        let (tp, pp) = track[i - 2];
        let (t0, p0) = track[i - 1];
        let (t1, p1) = track[i];
        if t1 - t0 > 0.1 && t0 - tp <= 0.1 {
            f(pp, p0, p1, t0 - tp, t1 - t0);
        }
    }
}

/// Roll-and-bounce dead reckoning across a tracking gap; `None` when a simple
/// reflection path can't reconnect (a collision lives in the gap).
fn physics_bridge(
    prev: DVec3, s: DVec3, e: DVec3, frame_dt: f64, gap_dt: f64, bounds: [f64; 4],
) -> Option<Vec<DVec3>> {
    let [min_x, max_x, min_y, max_y] = bounds;
    let (dx, dy) = (s.x - prev.x, s.y - prev.y);
    let seg = (dx * dx + dy * dy).sqrt();
    let entry_speed = seg / frame_dt.max(1e-6);
    if entry_speed < 0.05 || seg < 1e-6 {
        return None;
    }
    let (mut ux, mut uy) = (dx / seg, dy / seg);
    let step = 0.005;
    let max_len = entry_speed * gap_dt * 1.4;
    let n = ((max_len / step).ceil() as usize).min(4000);
    let (mut px, mut py) = (s.x, s.y);
    let mut pts = vec![s];
    let mut best_d = ((px - e.x).powi(2) + (py - e.y).powi(2)).sqrt();
    let mut best_i = 0usize;
    for k in 1..=n {
        px += ux * step;
        py += uy * step;
        if px < min_x { px = 2.0 * min_x - px; ux = -ux; }
        else if px > max_x { px = 2.0 * max_x - px; ux = -ux; }
        if py < min_y { py = 2.0 * min_y - py; uy = -uy; }
        else if py > max_y { py = 2.0 * max_y - py; uy = -uy; }
        pts.push(DVec3::new(px, py, s.z));
        let d = ((px - e.x).powi(2) + (py - e.y).powi(2)).sqrt();
        if d < best_d { best_d = d; best_i = k; }
    }
    if best_i < 1 || best_d > 0.09 {
        return None;
    }
    pts.truncate(best_i + 1);
    pts.push(e);
    Some(pts)
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

fn shot_color(colors: &[egui::Color32], i: usize) -> egui::Color32 {
    colors.get(i).copied().unwrap_or(egui::Color32::GRAY)
}

/// Linear interpolation along a tracked path (None inside >0.1s gaps).
fn interp_track(track: &[(f64, DVec3)], t: f64) -> Option<DVec3> {
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
            if t1 - t0 > 0.1 {
                return None;
            }
            let a = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return Some(p0 + (p1 - p0) * a);
        }
    }
    Some(track.last().unwrap().1)
}
