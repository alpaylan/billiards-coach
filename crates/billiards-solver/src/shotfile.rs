//! Shared loader for tracked `.shot` files (the MASA pipeline's output).
//!
//! Every consumer — the editor, calibration, verification, diagnostics — must
//! agree on how a shot file becomes a [`Scene`] + observed tracks; five separate
//! hand-rolled parsers had already drifted once. This is the one implementation.
//!
//! Beyond parsing, it fixes a real correctness problem in the scene setup:
//! **late-picked object balls**. An object ball is often occluded by the player
//! and cue stick during address, so its first detection can be *after* the cue
//! has already struck it — first tracked while moving, up to ~29 cm from where
//! it actually rested. Initializing the scene at that first detection puts the
//! collision geometry visibly wrong. [`scene_from_tracks`] back-extrapolates
//! such a ball to its rest: run its initial velocity backwards to the cue's
//! nearest preceding bend (the moment of contact). The *observed* samples are
//! left untouched — only the scene initialization (an estimate either way) uses
//! the correction, so no fabricated data ever enters the error metric.

use std::collections::HashMap;

use billiards_core::math::DVec3;
use billiards_core::{Scene, TableSpec};

use crate::fit::ObservedTrack;

const COLORS: [&str; 3] = ["white", "yellow", "red"];

/// A parsed shot: ball colors cue-first, their observed tracks, and the scene
/// (with corrected object rests) to reconstruct from.
pub struct LoadedShot {
    /// Ball color names, cue first — aligned with `observed` and the scene order.
    pub order: Vec<String>,
    pub observed: Vec<ObservedTrack>,
    pub scene: Scene,
}

/// Parse a color-labeled `.shot` text (`cue COLOR` header + `COLOR,t,x,y` data
/// lines; other header lines are ignored here). Returns `None` without a usable
/// cue track.
pub fn parse(text: &str, table: &TableSpec, radius: f64) -> Option<LoadedShot> {
    let mut cue = "white".to_string();
    let mut by_color: HashMap<String, ObservedTrack> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(c) = line.strip_prefix("cue ") {
            cue = c.trim().to_ascii_lowercase();
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 4 {
            continue;
        }
        let (Ok(t), Ok(x), Ok(y)) = (f[1].trim().parse(), f[2].trim().parse(), f[3].trim().parse()) else {
            continue;
        };
        by_color.entry(f[0].trim().to_string()).or_default().push((t, DVec3::new(x, y, radius)));
    }
    // The header's cue label comes from an upstream heuristic that is sometimes
    // wrong (a real case: "cue yellow" while yellow never moves and white is
    // already flying at frame one). The cue is, by definition, the ball the
    // player sets in motion — so when the labeled cue verifiably never leaves
    // its rest while another (white/yellow) ball travels the table, relabel to
    // the earliest mover. Red is never a cue ball in carom, so it can't win.
    let travel = |tr: &ObservedTrack| -> f64 {
        tr.first()
            .map_or(0.0, |&(_, p0)| tr.iter().map(|&(_, p)| (p - p0).length()).fold(0.0, f64::max))
    };
    let first_move_t = |tr: &ObservedTrack| -> Option<f64> {
        let &(_, p0) = tr.first()?;
        tr.iter().find(|&&(_, p)| (p - p0).length() > 0.03).map(|&(t, _)| t)
    };
    if by_color.get(&cue).is_none_or(|tr| travel(tr) < 0.05) {
        let real = by_color
            .iter()
            .filter(|(c, tr)| matches!(c.as_str(), "white" | "yellow") && **c != cue && travel(tr) > 0.30)
            .filter_map(|(c, tr)| first_move_t(tr).map(|t| (c.clone(), t)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((c, _)) = real {
            cue = c;
        }
    }
    let order: Vec<String> = std::iter::once(cue.clone())
        .chain(COLORS.iter().map(|s| s.to_string()).filter(|c| *c != cue))
        .filter(|c| by_color.contains_key(c))
        .collect();
    let observed: Vec<ObservedTrack> = order.iter().map(|c| by_color[c].clone()).collect();
    let scene = scene_from_tracks(&observed, table, radius)?;
    Some(LoadedShot { order, observed, scene })
}

/// Build the initial [`Scene`] from cue-first tracks, correcting late-picked
/// object rests (see the module docs). Public so the editor can share it.
pub fn scene_from_tracks(tracks: &[ObservedTrack], table: &TableSpec, radius: f64) -> Option<Scene> {
    let cue_track = tracks.first().filter(|t| t.len() >= 2)?;
    let cue_pos = cue_track[0].1;
    let bounds = table.center_bounds(radius);
    let objects: Vec<DVec3> = tracks[1..]
        .iter()
        .filter_map(|tr| rest_of(tr, cue_track, bounds))
        .collect();
    Some(Scene::new(cue_pos, objects))
}

/// An object ball's rest position: its first detection if it was genuinely
/// still, else a back-extrapolation to the cue's contact moment.
fn rest_of(track: &ObservedTrack, cue: &ObservedTrack, bounds: [f64; 4]) -> Option<DVec3> {
    let &(t0, p0) = track.first()?;
    // A late pickup means the ball is moving *immediately* at its first samples.
    // A ball that sits still for even a few frames before moving was genuinely at
    // rest where first detected — even if the (fast) cue reaches it soon after —
    // so the test must be "moving at pickup", not "moved within some window".
    let t_move = track.iter().find(|(_, p)| (*p - p0).length() > 0.02).map(|&(t, _)| t);
    let Some(t_move) = t_move else { return Some(p0) }; // never moves at all
    if t_move - t0 > 0.08 || track.len() < 3 {
        return Some(p0); // still for ≥2-3 frames — a real rest
    }
    // Moving at pickup (occluded at address): initial velocity from a short
    // least-squares fit so one blurred step doesn't set the backtrack direction.
    let k = track.len().min(4);
    let (mut num, mut den) = (DVec3::ZERO, 0.0);
    for &(t, p) in &track[1..k] {
        let dt = t - t0;
        num += (p - p0) * dt;
        den += dt * dt;
    }
    if den <= 1e-9 {
        return Some(p0);
    }
    let v = num / den;
    let speed = v.length();
    if speed < 0.15 {
        return Some(p0); // drifting within noise — treat as rest
    }
    // Contact time ≈ the cue's latest bend at/before this ball appeared (a
    // ball-ball hit turns the cue); without a bend, assume two frames earlier.
    let t_contact = cue_bend_before(cue, t0 + 0.05).unwrap_or(t0 - 0.067);
    let dt = (t0 - t_contact).clamp(0.0, 0.25);
    // Constant-speed backtrack (it decelerated since contact, so this slightly
    // under-corrects — far better than not correcting at all), capped and kept
    // on the table.
    let back = (speed * dt).min(0.30);
    let rest = p0 - v / speed * back;
    let [min_x, max_x, min_y, max_y] = bounds;
    let rest = DVec3::new(rest.x.clamp(min_x, max_x), rest.y.clamp(min_y, max_y), p0.z);
    // Plausibility: the backtrack model assumes this ball was struck *by the
    // cue* at `t_contact`, so the cue must actually have been nearby then (if it
    // wasn't — the ball was moved by the other object ball, or the "bend" we
    // found was a cushion — the model doesn't apply; keep the detection). And
    // the corrected rest must land *closer* to the cue's contact position than
    // the raw detection: farther means the observed velocity points away from
    // the true rest (e.g. the ball banked off a rail before being detected).
    if let Some(&(_, cue_c)) = cue.iter().min_by(|a, b| {
        (a.0 - t_contact).abs().total_cmp(&(b.0 - t_contact).abs())
    }) {
        if (p0 - cue_c).length() > 0.25 || (rest - cue_c).length() > (p0 - cue_c).length() + 0.01 {
            // The velocity backtrack is implausible (typically: the ball banked
            // off a rail before it was ever detected, so its observed velocity
            // points away from where it rested). Second strategy, from the CUE
            // side only: the momentum the cue lost at its bend, v_in − v_out,
            // points along the line of centers — so the struck ball rested one
            // ball-ball distance from the cue's contact position along it. The
            // cue path is well observed both sides of the bend.
            if (p0 - cue_c).length() < 0.25 {
                if let Some(rest2) = rest_from_cue_deflection(cue, t_contact, cue_c) {
                    if (rest2 - p0).length() < 0.25 {
                        return Some(rest2);
                    }
                }
            }
            return Some(p0);
        }
    }
    Some(rest)
}

/// Rest of a struck ball from the cue's own deflection at `t_contact`: the line
/// of centers is the direction of the cue's momentum change, `v_in − v_out`.
fn rest_from_cue_deflection(cue: &ObservedTrack, t_contact: f64, cue_c: DVec3) -> Option<DVec3> {
    let vel = |t_lo: f64, t_hi: f64| -> Option<DVec3> {
        let pts: Vec<&(f64, DVec3)> =
            cue.iter().filter(|(t, _)| *t >= t_lo && *t <= t_hi).collect();
        let (a, b) = (pts.first()?, pts.last()?);
        let dt = b.0 - a.0;
        (dt > 0.05).then(|| (b.1 - a.1) / dt)
    };
    let v_in = vel(t_contact - 0.30, t_contact - 0.03)?;
    let v_out = vel(t_contact + 0.03, t_contact + 0.30)?;
    let j = v_in - v_out;
    if j.length() < 0.3 || v_in.length() < 0.3 {
        return None; // too soft a deflection to define the line of centers
    }
    let r = 0.030_75; // carom ball radius (m) — matches BallSpec::carom()
    Some(cue_c + j / j.length() * (2.0 * r))
}

/// Latest time ≤ `t_max` at which the cue's path bends by more than ~20°
/// between consecutive travel segments (a collision or cushion).
fn cue_bend_before(cue: &ObservedTrack, t_max: f64) -> Option<f64> {
    let mut best = None;
    for w in cue.windows(3) {
        let (_, p0) = w[0];
        let (t1, p1) = w[1];
        let (_, p2) = w[2];
        if t1 > t_max {
            break;
        }
        let a = p1 - p0;
        let b = p2 - p1;
        if a.length() < 0.01 || b.length() < 0.01 {
            continue; // too little travel to define a direction
        }
        if a.dot(b) / (a.length() * b.length()) < 0.94 {
            best = Some(t1);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use billiards_core::BallSpec;

    fn table() -> TableSpec {
        TableSpec::carom_match()
    }

    #[test]
    fn still_object_keeps_first_detection() {
        let r = BallSpec::carom().radius;
        // Cue rolls +x; object sits still for 0.4s then is pushed.
        let cue: ObservedTrack = (0..12).map(|i| (i as f64 / 30.0, DVec3::new(-0.5 + 0.05 * i as f64, 0.0, r))).collect();
        let obj: ObservedTrack = (0..12)
            .map(|i| {
                let t = i as f64 / 30.0;
                let x = if t < 0.3 { 0.4 } else { 0.4 + (t - 0.3) * 1.0 };
                (t, DVec3::new(x, 0.0, r))
            })
            .collect();
        let scene = scene_from_tracks(&[cue, obj], &table(), r).unwrap();
        assert!((scene.objects[0].x - 0.4).abs() < 1e-9, "still ball keeps its detected rest");
    }

    #[test]
    fn late_pickup_backtracks_to_contact() {
        let r = BallSpec::carom().radius;
        // Cue travels +x at 3 m/s and bends (collision) at t=0.2, x=0.1.
        let cue: ObservedTrack = (0..12)
            .map(|i| {
                let t = i as f64 / 30.0;
                let p = if t <= 0.2 {
                    DVec3::new(-0.5 + 3.0 * t, 0.0, r)
                } else {
                    DVec3::new(0.1 - 1.0 * (t - 0.2), 0.8 * (t - 0.2), r) // deflected
                };
                (t, p)
            })
            .collect();
        // Object actually rested at (0.16, 0.0) but is only detected from t=0.3,
        // already moving +x at 1.0 m/s (so first seen at 0.16 + 0.1 = 0.26).
        let obj: ObservedTrack = (9..15)
            .map(|i| {
                let t = i as f64 / 30.0;
                (t, DVec3::new(0.16 + 1.0 * (t - 0.2), 0.0, r))
            })
            .collect();
        let scene = scene_from_tracks(&[cue, obj], &table(), r).unwrap();
        let rest = scene.objects[0];
        assert!(
            (rest.x - 0.16).abs() < 0.03,
            "back-extrapolated rest ~0.16, got {:.3} (first detection was {:.3})",
            rest.x,
            0.16 + 1.0 * (9.0 / 30.0 - 0.2)
        );
    }
}
