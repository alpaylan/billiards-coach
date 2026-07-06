//! Grade every shot of a game and aggregate per player.
//!
//! For each tracked shot: fit the player's action, estimate its make
//! probability under human execution noise (`success_probability` — the same
//! Monte-Carlo that ranks solver candidates), and note whether it actually
//! scored (turn-sequence annotation in results.json). Per player this yields:
//!
//!   * expected makes  Σp  — how many points their CHOSEN lines were worth
//!   * actual makes        — what they converted
//!   * delta               — execution above/below the noise model
//!   * avg p               — how ambitious/forgiving their shot selection runs
//!
//!   cargo run -p billiards-solver --example player_stats --release -- data/<game_dir>

use std::{env, fs};

use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_solver::shotfile;
use billiards_solver::{SolveConfig, success_probability};
use billiards_solver::fit::{FitConfig, fit_action};

fn load_phys(dir: &str) -> PhysicsParams {
    let path = format!("{}/calibration.json", dir.trim_end_matches('/'));
    let Ok(text) = fs::read_to_string(path) else { return PhysicsParams::carom_calibrated() };
    let get = |k: &str| -> Option<f64> {
        let i = text.find(&format!("\"{k}\""))?;
        let a = &text[i + k.len() + 2..];
        let a = &a[a.find(':')? + 1..];
        a.chars().skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
            .collect::<String>().parse().ok()
    };
    match (get("cushion_restitution"), get("cushion_friction"), get("mu_slide"), get("mu_roll")) {
        (Some(e), Some(f), Some(s), Some(r)) => PhysicsParams {
            cushion_restitution: e, cushion_friction: f, mu_slide: s, mu_roll: r,
            ..PhysicsParams::carom_calibrated()
        },
        _ => PhysicsParams::carom_calibrated(),
    }
}

fn main() {
    let dir = env::args().nth(1).expect("usage: player_stats <game_dir>");
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let phys = load_phys(&dir);
    let results = fs::read_to_string(format!("{}/results.json", dir.trim_end_matches('/')))
        .expect("results.json (run annotate_results first)");

    // shot -> (player, made) from the annotation
    let mut meta: Vec<(String, String, bool)> = Vec::new(); // (file, player, made)
    for chunk in results.split('{').skip(2) {
        let get = |k: &str| -> Option<String> {
            let i = chunk.find(&format!("\"{k}\""))?;
            let a = &chunk[i + k.len() + 2..];
            let a = &a[a.find(':')? + 1..];
            let q = a.find('"')?;
            let rest = &a[q + 1..];
            Some(rest[..rest.find('"')?].to_string())
        };
        if let (Some(shot), Some(player), Some(result)) = (get("shot"), get("player"), get("result")) {
            meta.push((shot, player, result == "make"));
        }
    }

    let fit_cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let mc = SolveConfig { mc_samples: 96, ..SolveConfig::default() };

    struct Acc {
        n: usize,
        made: usize,
        exp: f64,
        hardest_made: f64,
        softest_missed: f64,
    }
    let mut acc = std::collections::BTreeMap::<String, Acc>::new();

    println!("{:<16} {:>6} {:>7} {:>6}", "shot", "player", "p(make)", "made");
    for (file, player, made) in &meta {
        let path = format!("{}/{}", dir.trim_end_matches('/'), file);
        let Ok(text) = fs::read_to_string(&path) else { continue };
        let Some(s) = shotfile::parse(&text, &table, ball.radius) else { continue };
        let fit = fit_action(&s.scene, &s.observed, &table, &ball, &phys, &fit_cfg);
        let p = success_probability(&s.scene, &fit.action, &table, &ball, &phys, &mc);
        println!("{:<16} {:>6} {:>6.0}% {:>6}", file, player, p * 100.0, if *made { "✓" } else { "–" });
        let a = acc.entry(player.clone()).or_insert(Acc {
            n: 0,
            made: 0,
            exp: 0.0,
            hardest_made: 1.0,
            softest_missed: 0.0,
        });
        a.n += 1;
        a.made += *made as usize;
        a.exp += p;
        if *made && p < a.hardest_made {
            a.hardest_made = p;
        }
        if !*made && p > a.softest_missed {
            a.softest_missed = p;
        }
    }

    println!();
    for (player, a) in &acc {
        println!(
            "{player}: {} shots · expected {:.1} makes · actual {} ({:+.1}) · avg line {:.0}% · hardest make {:.0}% · softest miss {:.0}%",
            a.n, a.exp, a.made, a.made as f64 - a.exp, 100.0 * a.exp / a.n.max(1) as f64,
            a.hardest_made * 100.0, a.softest_missed * 100.0,
        );
    }
}
