//! `billiards` — headless CLI over the engine.
//!
//! Demo runner: simulates a full three-ball shot and prints each ball's
//! trajectory summary, the ordered contact log, and a first-cut three-cushion
//! scoring verdict computed from that log. Real subcommands (`sim`, `solve`,
//! `reconstruct`) and an arg parser arrive with later phases.

use std::collections::HashSet;

use billiards_core::math::DVec3;
use billiards_core::{BallColor, BallId, BallSpec, BallState, ContactEvent, ContactKind, PhysicsParams, TableSpec};
use billiards_engine::simulate;

#[cfg(feature = "track")]
mod match_cmd;
#[cfg(feature = "track")]
mod track_cmd;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "track") {
        #[cfg(feature = "track")]
        {
            if args.iter().any(|a| a == "--match") {
                match_cmd::run(&args[1..]);
            } else {
                track_cmd::run(&args[1..]);
            }
            return;
        }
        #[cfg(not(feature = "track"))]
        {
            eprintln!("this build lacks the tracker — rebuild with `--features track`");
            std::process::exit(2);
        }
    }
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let table = TableSpec::carom_match();
    let r = ball.radius;

    // BallId(0) = cue (white). The other two are the object balls.
    let labels = [BallColor::White, BallColor::Yellow, BallColor::Red];
    // A shot the engine found that scores: aim ~7°, 5.15 m/s, running english.
    let aim = 7f64.to_radians();
    let speed = 5.15;
    let balls = [
        BallState { pos: DVec3::new(-1.15, -0.45, r), vel: DVec3::new(speed * aim.cos(), speed * aim.sin(), 0.0), angular_vel: DVec3::new(0.0, 0.0, 50.0) },
        BallState { pos: DVec3::new(0.65, 0.28, r), vel: DVec3::ZERO, angular_vel: DVec3::ZERO },
        BallState { pos: DVec3::new(1.05, -0.10, r), vel: DVec3::ZERO, angular_vel: DVec3::ZERO },
    ];

    let sim = simulate(&balls, &table, &ball, &phys);

    println!("three-cushion engine — three-ball shot");
    println!("  table: {:.2} m × {:.2} m", table.length, table.width);
    for (i, traj) in sim.trajectories.iter().enumerate() {
        let rest = traj.final_state().pos;
        println!(
            "  {:?} (BallId {i}): {} segments, rests at ({:+.3},{:+.3}) t={:.2}s, {} cushions",
            labels[i],
            traj.segments.len(),
            rest.x,
            rest.y,
            traj.time_to_rest(),
            sim.cushion_count(BallId(i as u8)),
        );
    }

    println!("  contact log:");
    for e in &sim.events {
        match e.kind {
            ContactKind::Cushion { ball } => println!("    t={:.3}s  cushion   by {:?}", e.time, labels[ball.0 as usize]),
            ContactKind::BallBall { a, b } => println!("    t={:.3}s  ball⇄ball {:?}–{:?}", e.time, labels[a.0 as usize], labels[b.0 as usize]),
        }
    }

    let scored = three_cushion_scored(&sim.events, BallId(0));
    println!("  settled after {:.2}s — three-cushion point: {}", sim.settled_time(), if scored { "YES ✓" } else { "no" });
}

/// A three-cushion point: the cue ball contacts both object balls, with at least
/// three cue-ball cushion contacts before it reaches the *second* object ball.
fn three_cushion_scored(events: &[ContactEvent], cue: BallId) -> bool {
    let mut cue_cushions = 0usize;
    let mut objects_hit: HashSet<u8> = HashSet::new();
    for e in events {
        match e.kind {
            ContactKind::Cushion { ball } if ball == cue => cue_cushions += 1,
            ContactKind::BallBall { a, b } if a == cue || b == cue => {
                let other = if a == cue { b } else { a };
                if objects_hit.insert(other.0) && objects_hit.len() == 2 {
                    return cue_cushions >= 3;
                }
            }
            _ => {}
        }
    }
    false
}
