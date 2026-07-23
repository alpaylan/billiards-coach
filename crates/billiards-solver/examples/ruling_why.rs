//! Why does each ruling disagreement happen? For every shot where the
//! reconstruction's score contradicts the referee, report the evidence:
//!
//!   - fit quality (cue-only RMS: did the reconstruction even match the track?)
//!   - the observed skeleton's testimony (did the 2nd object verifiably move?
//!     how many cue cushion bounces are confidently visible before it did?)
//!   - the sim's margin at the decisive event (near-graze distance when it
//!     missed a ball the referee saw hit; cushion count and contact thinness
//!     when it scored a shot the referee called a miss)
//!
//! The point is a taxonomy, not a number: which disagreements are data limits
//! (the decisive event is below tracking resolution), which are fit/physics
//! failures (the tracks contain the answer and we still get it wrong), and
//! which look like label problems (tracks contradict the referee).
//!
//!   cargo run -p billiards-solver --example ruling_why --release -- data/masa4_day2/game_0*

use std::{env, fs};

use billiards_core::{BallId, BallSpec, ContactKind, PhysicsParams, Simulation, TableSpec, three_cushion_score};
use billiards_engine::simulate;
use billiards_solver::fit::{FirstMove, FitConfig, ObservedEvents, ObservedTrack, fit_action};
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

/// The sim's scoring story for the cue ball: (time, object index 1-based) of
/// each first contact with a distinct object, and cue cushion count before each.
struct SimStory {
    /// (time, ball index in trajectory order, cue cushions strictly before it)
    contacts: Vec<(f64, usize, usize)>,
    total_cushions: usize,
}

fn sim_story(sim: &Simulation, cue: BallId) -> SimStory {
    let mut cushions = 0usize;
    let mut contacts: Vec<(f64, usize, usize)> = Vec::new();
    let mut seen: Vec<BallId> = Vec::new();
    for e in &sim.events {
        match e.kind {
            ContactKind::Cushion { ball } if ball == cue => cushions += 1,
            ContactKind::BallBall { a, b } if a == cue || b == cue => {
                let other = if a == cue { b } else { a };
                if !seen.contains(&other) {
                    seen.push(other);
                    contacts.push((e.time, other.0 as usize, cushions));
                }
            }
            _ => {}
        }
    }
    SimStory { contacts, total_cushions: cushions }
}

/// Closest center-to-center approach (minus 2R) between the sim cue and a
/// target ball over the whole sim — how near the miss was, in meters of gap.
fn min_gap(sim: &Simulation, target: usize, radius: f64) -> f64 {
    let end = sim.settled_time();
    let (cue, tgt) = (&sim.trajectories[0], &sim.trajectories[target]);
    let mut t = 0.0;
    let mut best = f64::INFINITY;
    while t <= end {
        let d = (cue.state_at(t).pos - tgt.state_at(t).pos).length();
        best = best.min(d);
        t += 0.004;
    }
    best - 2.0 * radius
}

/// Cue-only RMS between sim and observed — did the reconstruction match the
/// track it was fit to?
fn cue_rms(sim: &Simulation, observed: &[ObservedTrack]) -> f64 {
    let (mut sse, mut n) = (0.0, 0usize);
    for &(t, p) in &observed[0] {
        sse += (sim.trajectories[0].state_at(t).pos - p).length_squared();
        n += 1;
    }
    if n == 0 { f64::INFINITY } else { (sse / n as f64).sqrt() }
}

fn fm_str(fm: FirstMove) -> String {
    match fm {
        FirstMove::Still => "still".into(),
        FirstMove::Unreliable => "unrel".into(),
        FirstMove::At(t) => format!("moved@{t:.2}"),
    }
}

fn main() {
    let dirs: Vec<String> = env::args().skip(1).collect();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };

    // category counters
    let mut tags: Vec<(String, String)> = Vec::new(); // (tag, shot id) per mismatch
    let (mut agree, mut total) = (0usize, 0usize);

    for dir in &dirs {
        let phys = load_calibration(dir);
        let mut paths: Vec<String> = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".shot"))
            .collect();
        paths.sort();
        for p in &paths {
            let Ok(text) = fs::read_to_string(p) else { continue };
            let label = text
                .lines()
                .find_map(|l| l.strip_prefix("result "))
                .map(str::trim)
                .unwrap_or("");
            if label != "make" && label != "miss" {
                continue;
            }
            let Some(shot) = shotfile::parse(&text, &table, ball.radius) else { continue };
            let f = fit_action(&shot.scene, &shot.observed, &table, &ball, &phys, &cfg);
            let sim = simulate(&shot.scene.ball_states(&f.action), &table, &ball, &phys);
            let scored = three_cushion_score(&sim, BallId(0));
            total += 1;
            if (label == "make") == scored {
                agree += 1;
                continue;
            }

            let id = format!(
                "{}/{}",
                dir.trim_end_matches('/').rsplit('/').next().unwrap(),
                p.rsplit('/').next().unwrap().trim_end_matches(".shot")
            );
            let events = ObservedEvents::from_tracks(&shot.observed, table.center_bounds(ball.radius));
            let story = sim_story(&sim, BallId(0));
            let crms = cue_rms(&sim, &shot.observed);
            let track_end = shot.observed[0].last().map_or(0.0, |s| s.0);

            // Observed testimony: per-object first-move, and the count of cue
            // cushion Vs confidently visible before the LAST object's move
            // (conservative undercount — only clear bounces are listed).
            let obj_fm: Vec<FirstMove> = events.first_move.clone();
            let second_move_t = obj_fm
                .iter()
                .filter_map(|fm| match fm {
                    FirstMove::At(t) => Some(*t),
                    _ => None,
                })
                .fold(f64::NAN, f64::max);
            let obs_cushions_before = if second_move_t.is_nan() {
                events.cushions[0].len()
            } else {
                events.cushions[0].iter().filter(|&&(t, _)| t < second_move_t).count()
            };

            let mut line = format!(
                "{id}: {} | fit_rms {:.3} cue_rms {:.3} | obs: obj1 {} obj2 {} cue_Vs_before {} (total {}) | sim: ",
                if label == "make" { "ruled MAKE, sim fails" } else { "ruled miss, sim SCORES" },
                f.rms_m,
                crms,
                fm_str(*obj_fm.first().unwrap_or(&FirstMove::Unreliable)),
                fm_str(*obj_fm.get(1).unwrap_or(&FirstMove::Unreliable)),
                obs_cushions_before,
                events.cushions[0].len(),
            );

            let tag: &str;
            match story.contacts.len() {
                2 => {
                    let (t2, b2, cush) = story.contacts[1];
                    // Thinness of the decisive contact: struck-ball speed just
                    // after, as a fraction of the cue's speed just before.
                    let v2 = sim.trajectories[b2].state_at(t2 + 1e-3).vel.length();
                    let vc = sim.trajectories[0].state_at(t2 - 1e-3).vel.length();
                    let thin = if vc > 1e-6 { v2 / vc } else { 0.0 };
                    line += &format!(
                        "hit both, 2nd@{t2:.2} after {cush} cushions (total {}), transfer {:.2}",
                        story.total_cushions, thin
                    );
                    if label == "make" {
                        // Sim hit both but failed the cushion requirement.
                        tag = "cushion-short";
                    } else if cush == 3 && obs_cushions_before < 3 {
                        tag = "cushion-borderline";
                    } else if thin < 0.12 {
                        tag = "thin-graze-scored";
                    } else if crms > 0.35 {
                        tag = "bad-fit";
                    } else if t2 > track_end {
                        tag = "beyond-track";
                    } else {
                        tag = "solid-false-score";
                    }
                }
                n @ (0 | 1) => {
                    // The sim never reached the 2nd (or any) object ball.
                    let missed: Vec<usize> = (1..sim.trajectories.len())
                        .filter(|i| !story.contacts.iter().any(|&(_, b, _)| b == *i))
                        .collect();
                    let gap = missed
                        .iter()
                        .map(|&i| min_gap(&sim, i, ball.radius))
                        .fold(f64::INFINITY, f64::min);
                    line += &format!(
                        "hit {n} ball(s), nearest miss gap {:.3} m, {} cushions",
                        gap, story.total_cushions
                    );
                    if label == "miss" {
                        // Scored with <2 contacts is impossible; this arm is
                        // only reachable for ruled-MAKE shots. Defensive:
                        tag = "impossible";
                    } else if crms > 0.35 {
                        tag = "bad-fit";
                    } else if gap < 0.04 {
                        tag = "near-graze-missed";
                    } else if matches!(obj_fm.get(1), Some(FirstMove::Unreliable))
                        || matches!(obj_fm.first(), Some(FirstMove::Unreliable))
                    {
                        tag = "track-corrupt";
                    } else if matches!(obj_fm.get(1), Some(FirstMove::Still))
                        || matches!(obj_fm.first(), Some(FirstMove::Still))
                    {
                        // Referee saw a hit; the track says that ball never
                        // moved. Either an invisible touch or a wrong label.
                        tag = "obs-contradicts-label";
                    } else {
                        tag = "wide-miss";
                    }
                }
                _ => {
                    tag = "multi";
                }
            }
            line += &format!("  => [{tag}]");
            println!("{line}");
            tags.push((tag.to_string(), id));
        }
    }

    println!("\n=== {agree}/{total} agree · {} mismatches ===", tags.len());
    let mut by_tag: Vec<(String, usize)> = Vec::new();
    for (t, _) in &tags {
        match by_tag.iter_mut().find(|(bt, _)| bt == t) {
            Some((_, c)) => *c += 1,
            None => by_tag.push((t.clone(), 1)),
        }
    }
    by_tag.sort_by(|a, b| b.1.cmp(&a.1));
    for (t, c) in &by_tag {
        println!("  {t:24} {c}");
    }
}
