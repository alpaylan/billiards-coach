//! Three-cushion scoring, read off a [`Simulation`]'s contact-event log.
//!
//! A point is scored when the cue ball contacts **both** object balls and makes
//! **at least three cushion contacts before touching the second** object ball.
//! (The three cushions may fall anywhere before that second contact.)

use crate::ball::BallId;
use crate::simulation::{ContactKind, Simulation};

/// Whether the given cue ball scores a three-cushion point in this simulation.
pub fn three_cushion_score(sim: &Simulation, cue: BallId) -> bool {
    let mut cue_cushions = 0usize;
    let mut objects_hit: Vec<BallId> = Vec::new();

    for event in &sim.events {
        match event.kind {
            ContactKind::Cushion { ball } if ball == cue => cue_cushions += 1,
            ContactKind::BallBall { a, b } if a == cue || b == cue => {
                let other = if a == cue { b } else { a };
                if !objects_hit.contains(&other) {
                    objects_hit.push(other);
                    if objects_hit.len() == 2 {
                        // Second distinct object ball contacted.
                        return cue_cushions >= 3;
                    }
                }
            }
            _ => {}
        }
    }
    false
}
