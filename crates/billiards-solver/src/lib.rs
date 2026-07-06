//! Three-cushion shot solver.
//!
//! Given a [`Scene`], search the cue-action space using the engine as the
//! forward model, and return the shot that not only scores but is the **most
//! forgiving** under realistic execution error.
//!
//! The spin search is parameterised as a 2D **tip offset** `(horizontal =
//! english, vertical = follow/draw)` over the miscue disk (radius ≈ ½R) — the
//! same quantity the strike diagram shows, and the thing a player actually
//! controls. This is speed-independent and automatically excludes un-hittable
//! strikes.
//!
//! The key idea (see `docs/DESIGN.md`): the **size of a shot's success basin
//! under execution noise** is one quantity that ranks candidate shots (widest
//! basin wins), defines the scene's *difficulty*, and is the *expected success
//! rate* a coach would quote. We estimate it by Monte-Carlo over the player's
//! execution noise. The grid and robustness passes are parallel (rayon) but
//! fully deterministic: every candidate gets a fixed seed and ties break by index.

pub mod calibrate;
pub mod fit;
mod par;
mod rng;
pub mod shotfile;

pub use calibrate::{CalibConfig, CalibShot, calibrate};
pub use fit::{FitConfig, FitResult, fit_action};
pub use shotfile::LoadedShot;

use billiards_core::{
    BallId, BallSpec, CueAction, PhysicsParams, Scene, TableSpec, three_cushion_score,
};
use billiards_engine::simulate;
use crate::par::*;
use rng::Rng;

/// How precisely the player executes an intended action (Gaussian standard
/// deviations). `offset_sd` is tip-placement error as a fraction of the radius.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionNoise {
    pub aim_sd: f64,    // rad
    pub speed_sd: f64,  // m/s
    pub offset_sd: f64, // fraction of ball radius
}

impl Default for ExecutionNoise {
    fn default() -> Self {
        Self { aim_sd: 0.025, speed_sd: 0.15, offset_sd: 0.04 }
    }
}

/// Search resolution and robustness settings.
#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    pub aim_steps: usize,
    pub speed_min: f64,
    pub speed_max: f64,
    pub speed_steps: usize,
    /// Steps per tip-offset axis (english and follow/draw each).
    pub offset_steps: usize,
    /// Miscue limit: max tip offset as a fraction of the ball radius.
    pub miscue_limit: f64,
    pub mc_samples: usize,
    pub noise: ExecutionNoise,
    pub seed: u64,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            aim_steps: 180,
            speed_min: 1.0,
            speed_max: 6.0,
            speed_steps: 11,
            offset_steps: 7,
            miscue_limit: 0.5,
            mc_samples: 64,
            noise: ExecutionNoise::default(),
            seed: 0x5EED,
        }
    }
}

/// The solver's recommendation for a scene.
#[derive(Clone, Copy, Debug)]
pub struct Solution {
    /// The most forgiving scoring action.
    pub action: CueAction,
    /// Estimated probability the shot scores under execution noise (0..1).
    pub success_prob: f64,
    /// How many grid cells scored — a coarse count of distinct scoring options.
    pub scoring_cells: usize,
}

impl Solution {
    /// Difficulty in [0, 1]: 0 = a sitter, 1 = essentially impossible.
    pub fn difficulty(&self) -> f64 {
        1.0 - self.success_prob
    }

    /// A human-facing difficulty label.
    pub fn category(&self) -> &'static str {
        match self.success_prob {
            p if p >= 0.70 => "easy",
            p if p >= 0.45 => "medium",
            p if p >= 0.20 => "hard",
            _ => "very hard",
        }
    }
}

pub(crate) fn lerp(a: f64, b: f64, i: usize, n: usize) -> f64 {
    if n <= 1 { 0.5 * (a + b) } else { a + (b - a) * (i as f64 / (n - 1) as f64) }
}

fn scores(scene: &Scene, action: &CueAction, table: &TableSpec, ball: &BallSpec, phys: &PhysicsParams) -> bool {
    let states = scene.ball_states(action);
    let sim = simulate(&states, table, ball, phys);
    three_cushion_score(&sim, BallId(0))
}

/// The tip-offset grid points (english, follow) inside the miscue disk.
fn offset_grid(cfg: &SolveConfig) -> Vec<(f64, f64)> {
    let lim = cfg.miscue_limit;
    let mut pts = Vec::new();
    for hi in 0..cfg.offset_steps {
        let h = lerp(-lim, lim, hi, cfg.offset_steps);
        for vi in 0..cfg.offset_steps {
            let v = lerp(-lim, lim, vi, cfg.offset_steps);
            if h * h + v * v <= lim * lim + 1e-9 {
                pts.push((h, v));
            }
        }
    }
    pts
}

/// Estimate the fraction of executions that still score when `base` is perturbed
/// by the execution noise (aim, speed, and tip placement).
fn robustness(
    scene: &Scene,
    base: &CueAction,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &SolveConfig,
    rng: &mut Rng,
) -> f64 {
    let (bh, bv) = base.tip_offset(ball.radius);
    let mut hits = 0usize;
    for _ in 0..cfg.mc_samples {
        let aim = base.aim + cfg.noise.aim_sd * rng.normal();
        let speed = (base.speed + cfg.noise.speed_sd * rng.normal()).max(0.2);
        let h = bh + cfg.noise.offset_sd * rng.normal();
        let v = bv + cfg.noise.offset_sd * rng.normal();
        let action = CueAction::from_tip_offset(aim, speed, h, v, ball.radius);
        if scores(scene, &action, table, ball, phys) {
            hits += 1;
        }
    }
    hits as f64 / cfg.mc_samples as f64
}

/// Estimated probability that `action` scores on `scene` under human execution
/// noise — the same Monte-Carlo the solver ranks candidates with, exposed so a
/// PLAYER's reconstructed shot can be graded ("how forgiving was the line they
/// chose?"). Deterministic for a given config.
pub fn success_probability(
    scene: &Scene,
    action: &CueAction,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &SolveConfig,
) -> f64 {
    let mut rng = Rng::new(cfg.seed ^ 0xB0B5_CA7E);
    robustness(scene, action, table, ball, phys, cfg, &mut rng)
}

/// Solve a scene: returns the most robust scoring shot, or `None` if the search
/// found no scoring action at this resolution.
pub fn solve(
    scene: &Scene,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &SolveConfig,
) -> Option<Solution> {
    let offsets = offset_grid(cfg);

    // 1. Coarse grid search (parallel over aim; order preserved for determinism).
    let scoring: Vec<CueAction> = (0..cfg.aim_steps)
        .into_par_iter()
        .flat_map_iter(|ai| {
            let aim = std::f64::consts::TAU * ai as f64 / cfg.aim_steps as f64;
            let mut local = Vec::new();
            for si in 0..cfg.speed_steps {
                let speed = lerp(cfg.speed_min, cfg.speed_max, si, cfg.speed_steps);
                for &(h, v) in &offsets {
                    let action = CueAction::from_tip_offset(aim, speed, h, v, ball.radius);
                    if scores(scene, &action, table, ball, phys) {
                        local.push(action);
                    }
                }
            }
            local
        })
        .collect();
    if scoring.is_empty() {
        return None;
    }

    // 2. Rank by robustness. Each candidate gets a fixed seed, so the estimate is
    // reproducible regardless of thread scheduling; ties break by lowest index.
    let scored: Vec<(f64, usize, CueAction)> = scoring
        .par_iter()
        .enumerate()
        .map(|(i, action)| {
            let mut rng = Rng::new(cfg.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            (robustness(scene, action, table, ball, phys, cfg, &mut rng), i, *action)
        })
        .collect();

    scored
        .into_iter()
        .max_by(|x, y| {
            x.0.partial_cmp(&y.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| y.1.cmp(&x.1))
        })
        .map(|(p, _, action)| Solution { action, success_prob: p, scoring_cells: scoring.len() })
}

/// Settings for [`repair`]: how far around the player's shot to look, and how to
/// weigh each kind of change (the search returns the *smallest* change that
/// scores, so the weights are the coaching cost of each adjustment).
#[derive(Clone, Copy, Debug)]
pub struct RepairConfig {
    pub aim_window: f64,   // rad, ± around the player's aim
    pub speed_window: f64, // m/s, ± around the player's speed
    pub aim_steps: usize,
    pub speed_steps: usize,
    pub offset_steps: usize,
    pub miscue_limit: f64,
    /// "Just-noticeable" scales that normalize each axis into a comparable cost.
    pub aim_scale: f64,    // rad
    pub speed_scale: f64,  // m/s
    pub offset_scale: f64, // fraction of R
    pub mc_samples: usize,
    pub noise: ExecutionNoise,
    pub seed: u64,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            aim_window: 0.35, // ~20°
            speed_window: 2.0,
            aim_steps: 41,
            speed_steps: 17,
            offset_steps: 7,
            miscue_limit: 0.5,
            aim_scale: 0.087, // ~5°
            speed_scale: 0.5,
            offset_scale: 0.15,
            mc_samples: 48,
            noise: ExecutionNoise::default(),
            seed: 0x5EED,
        }
    }
}

/// A coaching repair for a (missed) shot: the nearest action that scores, and
/// how it differs from what the player did.
#[derive(Clone, Copy, Debug)]
pub struct Repair {
    /// The closest scoring action to the player's.
    pub action: CueAction,
    /// Signed adjustments from the player's shot to this one.
    pub d_aim: f64,     // rad (wrapped to [-π, π])
    pub d_speed: f64,   // m/s
    pub d_english: f64, // fraction of R (+ = more running-side)
    pub d_follow: f64,  // fraction of R (+ = more follow)
    /// Robustness of the repaired shot (0..1) and the weighted change magnitude.
    pub success_prob: f64,
    pub cost: f64,
    /// True if the player's own shot already scored (no change needed).
    pub already_scores: bool,
}

fn wrap_pi(a: f64) -> f64 {
    let mut a = a % std::f64::consts::TAU;
    if a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    } else if a < -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

/// Shot repair for coaching: search near the player's action for the *smallest*
/// change that turns a miss into a score. Returns `None` if nothing in the
/// window scores — i.e. it was the wrong shot, not just mis-hit.
///
/// If the player's own shot already scores, returns it with zero adjustments.
pub fn repair(
    scene: &Scene,
    player: &CueAction,
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &RepairConfig,
) -> Option<Repair> {
    let r = ball.radius;
    let (ph, pv) = player.tip_offset(r);
    let offsets = offset_grid(&SolveConfig {
        offset_steps: cfg.offset_steps,
        miscue_limit: cfg.miscue_limit,
        ..SolveConfig::default()
    });

    let already = scores(scene, player, table, ball, phys);

    // Weighted coaching cost of moving from the player's shot to `(aim,speed,h,v)`.
    let cost = |aim: f64, speed: f64, h: f64, v: f64| {
        (wrap_pi(aim - player.aim) / cfg.aim_scale).powi(2)
            + ((speed - player.speed) / cfg.speed_scale).powi(2)
            + ((h - ph) / cfg.offset_scale).powi(2)
            + ((v - pv) / cfg.offset_scale).powi(2)
    };

    // Search the neighborhood; keep the lowest-cost scoring action (deterministic:
    // parallel over aim, ties break by lowest cost then earliest index).
    let best = (0..cfg.aim_steps)
        .into_par_iter()
        .filter_map(|ai| {
            let aim = lerp(player.aim - cfg.aim_window, player.aim + cfg.aim_window, ai, cfg.aim_steps);
            let mut local: Option<(f64, CueAction)> = None;
            for si in 0..cfg.speed_steps {
                let speed = lerp(
                    (player.speed - cfg.speed_window).max(0.4),
                    player.speed + cfg.speed_window,
                    si,
                    cfg.speed_steps,
                );
                for &(h, v) in &offsets {
                    let action = CueAction::from_tip_offset(aim, speed, h, v, r);
                    if scores(scene, &action, table, ball, phys) {
                        let c = cost(aim, speed, h, v);
                        if local.map_or(true, |(bc, _)| c < bc) {
                            local = Some((c, action));
                        }
                    }
                }
            }
            local
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))?;

    let (c, action) = best;
    let (h, v) = action.tip_offset(r);
    let mut rng = Rng::new(cfg.seed);
    let sp = robustness(
        scene,
        &action,
        table,
        ball,
        phys,
        &SolveConfig { mc_samples: cfg.mc_samples, noise: cfg.noise, ..SolveConfig::default() },
        &mut rng,
    );
    Some(Repair {
        action,
        d_aim: wrap_pi(action.aim - player.aim),
        d_speed: action.speed - player.speed,
        d_english: h - ph,
        d_follow: v - pv,
        success_prob: sp,
        cost: c.sqrt(),
        already_scores: already,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use billiards_core::math::DVec3;

    fn setup() -> (Scene, TableSpec, BallSpec, PhysicsParams) {
        let ball = BallSpec::carom();
        let r = ball.radius;
        let scene = Scene::new(
            DVec3::new(-1.0, -0.3, r),
            vec![DVec3::new(0.5, 0.15, r), DVec3::new(0.9, -0.25, r)],
        );
        (scene, TableSpec::carom_match(), ball, PhysicsParams::default())
    }

    #[test]
    fn repair_recovers_a_nearby_scoring_action() {
        let (scene, table, ball, phys) = setup();
        // A known scoring action for this scene (the solver's best).
        let sol = solve(&scene, &table, &ball, &phys, &SolveConfig::default())
            .expect("scene should have a scoring shot");
        assert!(scores(&scene, &sol.action, &table, &ball, &phys));

        // Nudge the aim off so it (likely) misses; repair should find its way back.
        let missed = CueAction { aim: sol.action.aim + 0.12, ..sol.action };
        let rep = repair(&scene, &missed, &table, &ball, &phys, &RepairConfig::default())
            .expect("a small change should score");
        assert!(scores(&scene, &rep.action, &table, &ball, &phys), "repaired action scores");
        // The fix stays within the search window (a genuinely small adjustment).
        assert!(rep.d_aim.abs() <= 0.35 + 1e-9);
    }

    #[test]
    fn repair_flags_an_already_scoring_shot() {
        let (scene, table, ball, phys) = setup();
        let sol = solve(&scene, &table, &ball, &phys, &SolveConfig::default()).unwrap();
        let rep = repair(&scene, &sol.action, &table, &ball, &phys, &RepairConfig::default()).unwrap();
        assert!(rep.already_scores, "an already-scoring shot is reported as such");
    }
}
