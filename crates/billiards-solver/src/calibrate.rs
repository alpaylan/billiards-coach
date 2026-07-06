//! Phase 4: physics calibration (system identification).
//!
//! Fit the engine's physics parameters so *simulated* trajectories match
//! *observed* real ones — the sim-to-real bridge. This is a nested optimization:
//! the **outer** loop searches the physics parameters; the **inner** [`fit_action`]
//! recovers each shot's (unknown) cue action under those parameters. The physics
//! that let the fitted actions reproduce the observations are the real table's.
//!
//! We calibrate the four parameters a cue trajectory constrains: cushion
//! restitution `e_c` (speed lost per bounce), cushion friction `f_c` (english's
//! effect on rebound angle), cloth sliding friction `μ_s` (the slide→roll
//! transition), and rolling resistance `μ_r` (how far it rolls). Ball–ball
//! parameters need collision-rich data and are left at their defaults here.

use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec};

use crate::fit::{FitConfig, ObservedTrack, fit_action};

/// One calibration example: the initial scene and the observed ball tracks.
pub struct CalibShot {
    pub scene: Scene,
    pub observed: Vec<ObservedTrack>,
}

/// Settings for calibration.
#[derive(Clone, Copy, Debug)]
pub struct CalibConfig {
    /// Inner action-fit resolution (kept coarse — we need residuals, not perfect actions).
    pub fit: FitConfig,
    pub iters: usize,
}

impl Default for CalibConfig {
    fn default() -> Self {
        Self {
            fit: FitConfig {
                aim_steps: 31,
                speed_steps: 8,
                offset_steps: 5,
                multistart: 8,
                refine_iters: 30,
                ..FitConfig::default()
            },
            iters: 50,
        }
    }
}

// [cushion_restitution, cushion_friction, mu_slide, mu_roll] and their bounds.
const BOUNDS: [(f64, f64); 4] = [(0.6, 0.98), (0.05, 0.4), (0.1, 0.35), (0.003, 0.02)];

fn pack(p: &PhysicsParams) -> [f64; 4] {
    [p.cushion_restitution, p.cushion_friction, p.mu_slide, p.mu_roll]
}

fn unpack(base: &PhysicsParams, x: &[f64; 4]) -> PhysicsParams {
    PhysicsParams {
        cushion_restitution: x[0],
        cushion_friction: x[1],
        mu_slide: x[2],
        mu_roll: x[3],
        ..*base
    }
}

/// Total (RMS) trajectory mismatch across all shots at these parameters, each
/// shot fitted for its best action.
fn objective(
    x: &[f64; 4],
    shots: &[CalibShot],
    table: &TableSpec,
    ball: &BallSpec,
    base: &PhysicsParams,
    fit: &FitConfig,
) -> f64 {
    let phys = unpack(base, x);
    let mut sse = 0.0;
    for s in shots {
        let r = fit_action(&s.scene, &s.observed, table, ball, &phys, fit);
        sse += r.rms_m * r.rms_m;
    }
    (sse / shots.len() as f64).sqrt()
}

/// Recover the physics parameters that best reproduce the observed shots,
/// starting from `base` (e.g. the literature defaults). Coordinate descent with
/// shrinking, bounded steps.
pub fn calibrate(
    shots: &[CalibShot],
    table: &TableSpec,
    ball: &BallSpec,
    base: &PhysicsParams,
    cfg: &CalibConfig,
) -> PhysicsParams {
    let mut x = pack(base);
    let mut step = [0.03, 0.03, 0.02, 0.002];
    let mut err = objective(&x, shots, table, ball, base, &cfg.fit);

    for _ in 0..cfg.iters {
        let mut improved = false;
        for k in 0..4 {
            for dir in [1.0, -1.0] {
                let mut cand = x;
                cand[k] = (cand[k] + dir * step[k]).clamp(BOUNDS[k].0, BOUNDS[k].1);
                if (cand[k] - x[k]).abs() < 1e-12 {
                    continue;
                }
                let e = objective(&cand, shots, table, ball, base, &cfg.fit);
                if e < err {
                    x = cand;
                    err = e;
                    improved = true;
                }
            }
        }
        if !improved {
            for s in &mut step {
                *s *= 0.5;
            }
            if step.iter().all(|&s| s < 1e-4) {
                break;
            }
        }
    }
    unpack(base, &x)
}
