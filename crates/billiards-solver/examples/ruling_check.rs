//! Ruling agreement: does each shot's RECONSTRUCTION score exactly when the
//! referee said it scored? The banner-derived labels (validated against the
//! final scoreboards) are ground truth for the collision structure — a "make"
//! requires the cue to contact both object balls with 3+ cushions before the
//! second — so agreement here measures the physics + fit end to end, and every
//! disagreement is a shot where we mis-detected a contact (close pass read as
//! a hit, or vice versa) or mis-simulated the path.
//!
//! Beyond the point verdict, the fit's multistart population gives a
//! PROBABILITY: near-tied candidates are alternative explanations of the same
//! tracks, so P(score) is their loss-weighted vote. A shot whose ruling is
//! genuinely undecidable from the data (feather touch, borderline 3rd cushion,
//! chaotic tail) shows up as p ≈ 0.5 instead of a confident coin flip. The
//! probabilistic metrics — Brier, log-loss, reliability, abstention curve —
//! measure what the boolean agreement can't: whether the pipeline KNOWS which
//! rulings it knows.
//!
//!   cargo run -p billiards-solver --example ruling_check --release -- data/masa4_day2/game_0*
//!
//! ENSEMBLE_T (default 0.04): loss→weight temperature(s), comma-separated to
//! compare several. Interpreted as the model-error scale in meters: candidates
//! within ~T of the best loss are credible alternatives.
//! PHYS_JITTER (default 0.5, 0 = off): scale of the ±σ physics-parameter
//! replays each candidate votes under (see [`jittered`]). Both defaults were
//! chosen by reading the reliability table on the masa4_day2 corpus: T=0.04 /
//! k=0.5 keeps every bin except [0,0.2) calibrated. That bottom bin claims ~4%
//! and delivers ~15% — the honest signature of the failure modes no parameter
//! can widen (feather touches below tracking resolution, fits that never
//! matched the track); fixing it needs observation-level uncertainty, not
//! more physics spread.

use std::{env, fs};

use billiards_core::{BallId, BallSpec, PhysicsParams, TableSpec, three_cushion_score};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action_ensemble};
use billiards_solver::shotfile;

fn load_calibration(dir: &str) -> PhysicsParams {
    let path = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let Ok(text) = fs::read_to_string(path) else { return PhysicsParams::carom_calibrated() };
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>()
            .parse()
            .ok()
    };
    match (get("cushion_restitution"), get("cushion_friction"), get("mu_slide"), get("mu_roll")) {
        (Some(e), Some(f), Some(s), Some(r)) => PhysicsParams {
            cushion_restitution: e,
            cushion_friction: f,
            mu_slide: s,
            mu_roll: r,
            ..PhysicsParams::carom_calibrated()
        },
        _ => PhysicsParams::carom_calibrated(),
    }
}

/// Balanced ±1σ design over (cushion_restitution, cushion_friction, mu_slide,
/// mu_roll): 8 orthogonal sign rows, so each parameter is high in half the
/// replays and no two move in lockstep.
const JITTER_SIGNS: [[f64; 4]; 8] = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0],
];

/// One ±1σ variation of the calibrated physics, at `k`×σ (base σ: e_c 0.02,
/// f_c 0.03, cloth friction 10% — the scale of game-to-game calibration
/// spread). The right k is an empirical question the reliability table
/// answers: too small leaves the extremes overconfident, too large drags
/// well-determined verdicts toward 0.5.
fn jittered(p: &PhysicsParams, s: [f64; 4], k: f64) -> PhysicsParams {
    PhysicsParams {
        cushion_restitution: (p.cushion_restitution + 0.02 * k * s[0]).clamp(0.5, 0.99),
        cushion_friction: (p.cushion_friction + 0.03 * k * s[1]).max(0.0),
        mu_slide: p.mu_slide * (1.0 + 0.10 * k * s[2]),
        mu_roll: p.mu_roll * (1.0 + 0.10 * k * s[3]),
        ..p.clone()
    }
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let mut cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    // SPEED_SCALE: experimental launch-speed-pin multiplier (see FitConfig).
    if let Some(v) = env::var("SPEED_SCALE").ok().and_then(|s| s.parse().ok()) {
        cfg.speed_scale = v;
    }

    // Same physics-sweep overrides as verify_match (see there).
    let envf = |k: &str| env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
    let (bf, bf_b, bf_c, mu_sp) = (envf("BF"), envf("BF_B"), envf("BF_C"), envf("MU_SPIN"));
    let eb = envf("EB");
    let bb_steps: Option<u32> = env::var("BB_STEPS").ok().and_then(|s| s.parse().ok());
    let cushion_steps: Option<u32> = env::var("CUSHION_STEPS").ok().and_then(|s| s.parse().ok());
    let ec_slope = envf("EC_SLOPE");
    // Cushion-map overrides (replace the per-game calibrated values — used to
    // trial event-locally fitted params, see cushion_events.rs).
    let (ec, fc) = (envf("EC"), envf("FC"));
    let ecs_t = envf("ECS_T");
    let ecd = envf("ECD");

    // Temperatures to evaluate the ensemble probability at (see module docs).
    let temps: Vec<f64> = env::var("ENSEMBLE_T")
        .unwrap_or_else(|_| "0.04".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    // PHYS_JITTER: jitter scale k (0 disables; default 0.5×σ).
    let jitter_k: f64 = env::var("PHYS_JITTER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);

    let (mut tp, mut tn, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    // Per labeled shot: (ruled make?, point verdict, candidate (Δloss, vote)
    // outcomes — everything needed to form P(score) at any temperature).
    let mut records: Vec<(bool, bool, Vec<(f64, f64)>)> = Vec::new();
    for dir in &dirs {
        let mut phys = load_calibration(dir);
        if let Some(v) = bf {
            phys.ball_friction = v;
        }
        if let Some(v) = bf_b {
            phys.ball_friction_b = v;
        }
        if let Some(v) = bf_c {
            phys.ball_friction_c = v;
        }
        if let Some(v) = mu_sp {
            phys.mu_spin = v;
        }
        if let Some(v) = eb {
            phys.ball_restitution = v;
        }
        if let Some(v) = bb_steps {
            phys.ball_contact_steps = v;
        }
        if let Some(v) = cushion_steps {
            phys.cushion_contact_steps = v;
        }
        if let Some(v) = ec_slope {
            phys.cushion_restitution_slope = v;
        }
        if let Some(v) = ec {
            phys.cushion_restitution = v;
        }
        if let Some(v) = fc {
            phys.cushion_friction = v;
        }
        if let Some(v) = ecs_t {
            phys.cushion_restitution_slope_t = v;
        }
        if let Some(v) = ecd {
            phys.cushion_restitution_chain = v;
        }
        let mut paths: Vec<String> = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".shot"))
            .collect();
        paths.sort();
        let (mut gtp, mut gtn, mut gfp, mut gfn) = (0usize, 0usize, 0usize, 0usize);
        let mut wrong: Vec<String> = Vec::new();
        for p in &paths {
            let Ok(text) = fs::read_to_string(p) else { continue };
            let label = text
                .lines()
                .find_map(|l| l.strip_prefix("result "))
                .map(str::trim)
                .unwrap_or("");
            if label != "make" && label != "miss" {
                continue; // spurious/unlabeled
            }
            let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };
            let ens = fit_action_ensemble(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
            let sim = simulate(&shot.scene.ball_states(&ens.best.action), &table, &ball, &phys);
            let scored = three_cushion_score(&sim, BallId(0));
            // Every credible candidate votes; store (Δloss, vote) so the vote
            // can be weighted at any temperature after the loop. Beyond Δloss
            // 1.0 the weight is negligible at every temperature we use (and a
            // skeleton violation costs 0.7-1.5 — causally wrong candidates are
            // meant to be excluded here). A candidate's vote is the fraction
            // of its physics-jittered replays that score: the per-game
            // calibration is itself uncertain, and late-shot outcomes are
            // chaotic in exactly those parameters, so a verdict that flips
            // under ±1σ parameter noise carries less conviction than one that
            // survives it.
            let l0 = ens.candidates.first().map_or(0.0, |c| c.loss);
            let outcomes: Vec<(f64, f64)> = ens
                .candidates
                .iter()
                .take_while(|c| c.loss - l0 <= 1.0)
                .map(|c| {
                    let states = shot.scene.ball_states(&c.action);
                    let mut yes = 0usize;
                    let mut total = 0usize;
                    let mut tally = |ph: &PhysicsParams| {
                        let s = simulate(&states, &table, &ball, ph);
                        total += 1;
                        if three_cushion_score(&s, BallId(0)) {
                            yes += 1;
                        }
                    };
                    tally(&phys);
                    if jitter_k > 0.0 {
                        for signs in JITTER_SIGNS {
                            tally(&jittered(&phys, signs, jitter_k));
                        }
                    }
                    (c.loss - l0, yes as f64 / total as f64)
                })
                .collect();
            records.push((label == "make", scored, outcomes));
            match (label == "make", scored) {
                (true, true) => gtp += 1,
                (false, false) => gtn += 1,
                (false, true) => {
                    gfp += 1;
                    wrong.push(format!("{}: recon SCORES but ruled miss", p.rsplit('/').next().unwrap()));
                }
                (true, false) => {
                    gfn += 1;
                    wrong.push(format!("{}: recon fails but ruled MAKE", p.rsplit('/').next().unwrap()));
                }
            }
        }
        let n = gtp + gtn + gfp + gfn;
        println!(
            "{}: {}/{} agree ({:.0}%) · make✓ {gtp} miss✓ {gtn} · false-score {gfp} · missed-score {gfn}",
            dir.trim_end_matches('/').rsplit('/').next().unwrap(),
            gtp + gtn,
            n,
            100.0 * (gtp + gtn) as f64 / n.max(1) as f64,
        );
        for w in wrong.iter().take(4) {
            println!("    {w}");
        }
        tp += gtp;
        tn += gtn;
        fp += gfp;
        fn_ += gfn;
    }
    let n = tp + tn + fp + fn_;
    println!(
        "=== ruling agreement: {}/{} ({:.1}%) · false-score {fp} · missed-score {fn_} ===",
        tp + tn,
        n,
        100.0 * (tp + tn) as f64 / n.max(1) as f64
    );

    // Context for the proper scores: what the base-rate predictor achieves.
    let rate = records.iter().filter(|(y, ..)| *y).count() as f64 / records.len().max(1) as f64;
    println!(
        "\nbase rate {:.2} → constant predictor Brier {:.3} · log-loss {:.3}",
        rate,
        rate * (1.0 - rate),
        -(rate * rate.ln() + (1.0 - rate) * (1.0 - rate).ln()),
    );

    for &t in &temps {
        // P(score): loss-weighted candidate vote at temperature t.
        let probs: Vec<(bool, f64)> = records
            .iter()
            .map(|(y, point, outcomes)| {
                let (mut wsum, mut wyes) = (0.0f64, 0.0f64);
                for &(dl, frac) in outcomes {
                    let w = (-dl / t).exp();
                    wsum += w;
                    wyes += w * frac;
                }
                let p = if wsum > 0.0 { wyes / wsum } else if *point { 1.0 } else { 0.0 };
                (*y, p)
            })
            .collect();
        let nn = probs.len().max(1) as f64;
        let agree = probs.iter().filter(|&&(y, p)| (p >= 0.5) == y).count();
        let brier: f64 =
            probs.iter().map(|&(y, p)| (p - y as u8 as f64).powi(2)).sum::<f64>() / nn;
        let logloss: f64 = probs
            .iter()
            .map(|&(y, p)| {
                let pc = p.clamp(1e-3, 1.0 - 1e-3);
                if y { -pc.ln() } else { -(1.0 - pc).ln() }
            })
            .sum::<f64>()
            / nn;
        let conf_wrong =
            probs.iter().filter(|&&(y, p)| (p >= 0.9 && !y) || (p <= 0.1 && y)).count();
        println!(
            "\n=== ensemble T={t}: p≥0.5 agreement {agree}/{} ({:.1}%) · Brier {brier:.3} · log-loss {logloss:.3} · confident-wrong {conf_wrong} ===",
            probs.len(),
            100.0 * agree as f64 / nn,
        );
        // Reliability: within each predicted band, did that fraction make?
        println!("  reliability (bin · n · mean p · make rate):");
        for b in 0..5 {
            let (lo, hi) = (b as f64 / 5.0, (b + 1) as f64 / 5.0);
            let sel: Vec<&(bool, f64)> =
                probs.iter().filter(|(_, p)| *p >= lo && (*p < hi || b == 4)).collect();
            if sel.is_empty() {
                continue;
            }
            let mp = sel.iter().map(|(_, p)| p).sum::<f64>() / sel.len() as f64;
            let mr = sel.iter().filter(|(y, _)| *y).count() as f64 / sel.len() as f64;
            println!("    [{lo:.1},{hi:.1}) {:4}  {mp:.2}  {mr:.2}", sel.len());
        }
        // Abstention: rule only the most confident X% — what's the agreement?
        let mut by_conf = probs.clone();
        by_conf.sort_by(|a, b| (b.1 - 0.5).abs().total_cmp(&(a.1 - 0.5).abs()));
        print!("  abstention:");
        for cov in [1.0, 0.9, 0.75, 0.5] {
            let k = ((probs.len() as f64) * cov).round() as usize;
            let a = by_conf[..k].iter().filter(|&&(y, p)| (p >= 0.5) == y).count();
            print!("  {:.0}% cover → {:.1}%", cov * 100.0, 100.0 * a as f64 / k.max(1) as f64);
        }
        println!();
    }
}
