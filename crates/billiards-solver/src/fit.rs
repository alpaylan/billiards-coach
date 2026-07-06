//! Hit reconstruction: recover the cue action from observed ball trajectories.
//!
//! Given the ball positions at the start of a shot (a [`Scene`], from tracking)
//! and the observed paths the balls took, find the [`CueAction`] — aim, speed,
//! english, follow/draw — whose *simulated* trajectory best matches what was
//! observed. This is the solver's forward-model-as-oracle idea run in reverse:
//! instead of searching for an action that *scores*, we search for the action
//! that *reproduces reality*.
//!
//! The search mirrors the solver — a parallel grid over (aim, speed, 2D tip
//! offset) — scored by trajectory mismatch instead of by scoring, then polished
//! with a coordinate-descent refinement.
//!
//! Candidates are scored on RMS **plus the event skeleton** ([`ObservedEvents`],
//! [`event_penalty`]): the ball contacts and cushion bounces the tracks
//! verifiably show. Pointwise RMS alone is shape-blind — on a long track the
//! resting tail dominates, so a path telling the wrong physical story (other
//! rails, an invented contact) can out-score the faithful one. Skeleton
//! violations cost more than a wrong-family path ever saves in RMS, making the
//! score effectively lexicographic: right causal class first, closeness second.
//!
//! Identifiability: the cue's cushion rebounds encode english; a ball–ball
//! contact reveals follow/draw via the cue's post-collision path. A shot with a
//! collision and a cushion is well-posed; a short straight roll is not.

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, CueAction, PhysicsParams, Scene, Simulation, TableSpec};
use billiards_engine::simulate;
use crate::par::*;

use crate::lerp;

/// One ball's observed path: (time seconds, position). Aligned with the scene's
/// ball order (index 0 = cue).
pub type ObservedTrack = Vec<(f64, DVec3)>;

/// Search resolution for the fit.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    pub aim_steps: usize,
    pub speed_min: f64,
    pub speed_max: f64,
    pub speed_steps: usize,
    pub offset_steps: usize,
    pub miscue_limit: f64,
    pub refine_iters: usize,
    /// Refine from this many best grid candidates. Trajectory-matching across
    /// collisions is highly non-convex, so a single start lands in the wrong
    /// basin; multi-start escapes it.
    pub multistart: usize,
    /// Half-width (rad) of the aim search window centered on the aim read
    /// directly from the cue's pre-collision heading.
    pub aim_window: f64,
    /// Fractional half-width of the speed search around the launch speed read
    /// directly from the cue's free flight (e.g. 0.2 = ±20%). The launch speed is
    /// observed, so — like the aim — it anchors the fit instead of floating free.
    pub speed_window: f64,
    /// Enforce the observed cushion skeleton (rails each ball verifiably
    /// bounced off). Off = legacy scoring (RMS + ball-contact structure only).
    pub skeleton: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            aim_steps: 41,
            speed_min: 0.8,
            speed_max: 6.5,
            speed_steps: 14,
            offset_steps: 7,
            miscue_limit: 0.5,
            refine_iters: 60,
            multistart: 24,
            aim_window: 0.18,   // ±~10°
            // ±22% around the observed launch speed — room for measurement noise and
            // the slide→roll adjustment, but far too tight to inflate speed 3-5x to
            // hide physics error (which is what the free speed search was doing).
            speed_window: 0.22,
            skeleton: true,
        }
    }
}

/// The recovered hit and how well it fits.
#[derive(Clone, Copy, Debug)]
pub struct FitResult {
    pub action: CueAction,
    /// RMS distance (meters) between observed and simulated ball positions.
    pub rms_m: f64,
}

/// The observed event skeleton: which physical events the tracks actually show.
/// For each object ball, when it FIRST moved (None = it never moved); for every
/// ball, the cushion bounces confidently visible in its track. Extracted with
/// sustained-motion / V-shape tests so single-frame jitter doesn't fabricate an
/// event. This is what a candidate simulation must agree with *causally* —
/// a similar-looking path built from the wrong elements (different rails, a
/// contact that never happened) is a wrong reconstruction no matter its RMS.
/// What an object ball's track testifies about it being struck.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FirstMove {
    /// Verifiably never left its rest.
    Still,
    /// Verifiably struck, first moving at this time (s).
    At(f64),
    /// The track claims motion but the claim is untrustworthy — an identity
    /// swap (its "movement" rides another ball's blob) or too little evidence
    /// of a real departure. Constrains nothing, in either direction.
    Unreliable,
}

#[derive(Clone, Debug)]
pub struct ObservedEvents {
    /// Per object (cue-first order, so index 0 here = observed[1]).
    pub first_move: Vec<FirstMove>,
    /// Per ball (same order as the tracks, 0 = cue): observed cushion bounces
    /// as (time, rail), rail ∈ 0..4 = -x, +x, -y, +y. Conservative: only
    /// bounces with a clear approach AND departure are listed.
    pub cushions: Vec<Vec<(f64, u8)>>,
    /// Center bounds [min_x, max_x, min_y, max_y] the rails were read against.
    bounds: [f64; 4],
    /// The tracks themselves, kept so a *simulated* bounce can be tested for
    /// positive contradiction (ball observed far from that rail at that time)
    /// rather than punished merely for falling in a tracking gap.
    tracks: Vec<ObservedTrack>,
}

/// Distance of a ball-center position to one rail's center-bound line.
/// Rail encoding matches [`ObservedEvents::cushions`]: 0 = -x, 1 = +x, 2 = -y, 3 = +y.
fn rail_dist(p: DVec3, rail: u8, b: [f64; 4]) -> f64 {
    match rail {
        0 => p.x - b[0],
        1 => b[1] - p.x,
        2 => p.y - b[2],
        _ => b[3] - p.y,
    }
}

/// Cushion bounces confidently visible in one track: a V in the distance-to-rail
/// series — the ball comes within `NEAR` of the rail and verifiably approaches
/// AND departs (≥ `PROM` prominence within `SIDE_T` on both sides, with real
/// perpendicular speed both ways — mid-table a V like that has no other cause).
/// `NEAR` is generous because near-rail positions carry the worst measurement
/// bias (rough corners, parallax): a real bounce can appear to turn 10+ cm short
/// of the nose. A ball that rolls up to a rail and stays, starts near a rail, or
/// merely jitters while resting there produces no V, so none of those fabricate
/// a bounce; a bounce inside a tracking gap has no nearby samples and is
/// (honestly) not claimed. Returns (time, rail, ball position at the V).
fn observed_bounces(track: &[(f64, DVec3)], bounds: [f64; 4]) -> Vec<(f64, u8, DVec3)> {
    const NEAR: f64 = 0.16; // a V this close to the nose ⇒ contact plausible (m)
    const PROM: f64 = 0.03; // required rise on both sides — above tracking jitter (m)
    const V_MIN: f64 = 0.20; // and at a real perpendicular speed (m/s), both ways
    const SIDE_T: f64 = 0.40; // look this far (s) around the minimum for the rise
    const CORNER_T: f64 = 0.10; // two rails' V-minima this close are ONE event (s)
    let mut out: Vec<(f64, u8, DVec3, f64)> = Vec::new(); // (time, rail, pos, impact speed)
    for rail in 0..4u8 {
        let d: Vec<f64> = track.iter().map(|&(_, p)| rail_dist(p, rail, bounds)).collect();
        let mut i = 0;
        while i < track.len() {
            if d[i] >= NEAR {
                i += 1;
                continue;
            }
            let mut j = i; // the whole below-NEAR stretch is one candidate bounce
            while j + 1 < track.len() && d[j + 1] < NEAR {
                j += 1;
            }
            let m = (i..=j).min_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap();
            let tm = track[m].0;
            // Perpendicular closing/opening speed around the V: fastest slope of
            // d toward the minimum on each side. Confirms the rise (≥ PROM) and,
            // at a corner, tells the rail that was HIT (large on both sides — a
            // collision) from one merely grazed (the ball only drifts in that axis).
            let v_in = track[..m]
                .iter()
                .zip(&d[..m])
                .rev()
                .take_while(|(s, _)| tm - s.0 <= SIDE_T)
                .filter(|&(_, &dk)| dk > d[m] + PROM)
                .map(|(s, &dk)| (dk - d[m]) / (tm - s.0).max(1e-6))
                .fold(0.0, f64::max);
            let v_out = track[m + 1..]
                .iter()
                .zip(&d[m + 1..])
                .take_while(|(s, _)| s.0 - tm <= SIDE_T)
                .filter(|&(_, &dk)| dk > d[m] + PROM)
                .map(|(s, &dk)| (dk - d[m]) / (s.0 - tm).max(1e-6))
                .fold(0.0, f64::max);
            if v_in > V_MIN && v_out > V_MIN {
                out.push((tm, rail, track[m].1, v_in + v_out));
            }
            i = j + 1;
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Corner disambiguation: a bounce beside an adjacent rail can reverse the
    // tangential velocity too (english), giving BOTH rails a V at the same
    // moment though only one was touched. Claim only the harder impact — the
    // unclaimed rail, if it really was a double corner kiss, is still free for
    // the sim (the contradiction test can't fire with the ball right there).
    let mut keep = vec![true; out.len()];
    for a in 0..out.len() {
        for b in a + 1..out.len() {
            if out[b].0 - out[a].0 > CORNER_T {
                break;
            }
            if keep[a] && keep[b] && out[a].1 != out[b].1 {
                let drop = if out[a].3 >= out[b].3 { b } else { a };
                keep[drop] = false;
            }
        }
    }
    out.iter()
        .zip(&keep)
        .filter(|&(_, &k)| k)
        .map(|(&(t, rail, p, _), _)| (t, rail, p))
        .collect()
}

impl ObservedEvents {
    pub fn from_tracks(observed: &[ObservedTrack], bounds: [f64; 4]) -> Self {
        // A first-move event needs BOTH: sustained motion (two consecutive
        // samples clear of the rest, going further — jitter can't fake it) AND
        // physical causality — some OTHER ball's track passes within reach of
        // this ball's rest around that moment. A ball "moving" with nothing
        // near it is a tracking artifact, and accepting it would FORCE the fit
        // to fabricate the very collisions this structure exists to forbid.
        // 2R (61 mm) + a frame of travel: at 30 fps a 2 m/s striker covers
        // ~7 cm between samples, and the struck ball has already left its rest
        // by the first post-contact frame — 0.12 verifiably missed a real
        // strike whose nearest striker sample was 12.5 cm from the rest.
        const REACH: f64 = 0.16;
        // A mover is an artifact only when every other ball POSITIVELY excludes
        // itself: observed in the window and observed far away. A ball with no
        // samples there (blur gap right at impact is common) could have been at
        // the contact — absence of evidence must not veto a real collision.
        // The window opens at `t_prev` — the mover's own last sample before the
        // motion — not a fixed offset before `t`: a hard-struck ball can fly
        // undetected for over half a second, and by its re-detection the
        // striker has long left the contact zone. Returns the best estimate of
        // the CONTACT time: the striker's closest approach to the rest inside
        // the window (falling back to `t` when no striker was seen there).
        let explainable = |ki: usize, p0: DVec3, t_prev: f64, t: f64| -> Option<f64> {
            let (lo, hi) = (t_prev - 0.35, t + 0.12);
            let mut best: Option<(f64, f64)> = None; // (dist, time)
            let mut silent_witness = false;
            for (j, other) in observed.iter().enumerate() {
                if j == ki {
                    continue;
                }
                let mut seen = false;
                for &(tt, p) in other.iter().filter(|(tt, _)| *tt >= lo && *tt <= hi) {
                    seen = true;
                    let d = (p - p0).length();
                    if d < REACH && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, tt));
                    }
                }
                if !seen {
                    silent_witness = true; // could have been at the contact
                }
            }
            match best {
                Some((_, tc)) => Some(tc.clamp(t_prev, t)),
                None if silent_witness => Some(t),
                None => None,
            }
        };
        // Two ball centers can never be closer than 2R (~61 mm). A "mover"
        // whose motion samples stay essentially ON another ball's path is the
        // tracker re-labeling that ball (identity swap) — the classic case is
        // the cue rolling near this ball's rest, which also satisfies the
        // reach test above (the nearby cue is exactly what caused the swap).
        // The swap usually steals the other ball's detections too (its track
        // goes silent), so probe its *interpolated* position; a really-struck
        // ball separates immediately — the 2R floor keeps it outside OVERLAP.
        const OVERLAP: f64 = 0.045;
        let interp_at = |track: &ObservedTrack, t: f64| -> Option<DVec3> {
            let after = track.iter().position(|(tt, _)| *tt >= t);
            match after {
                Some(0) => Some(track[0].1),
                Some(bi) => {
                    let (ta, pa) = track[bi - 1];
                    let (tb, pb) = track[bi];
                    Some(pa + (pb - pa) * ((t - ta) / (tb - ta).max(1e-6)))
                }
                None => track.last().map(|&(_, p)| p), // silent to the end — last seen
            }
        };
        // Is the motion starting at `t0` a real departure? Judge it on the
        // following samples that are actually away from the rest (a frame back
        // AT the rest is evidence of stillness, not motion): enough of them
        // must exist, and they must not ride another ball. A fast-struck ball
        // is often a set of sparse post-gap detections, so count the next few
        // samples wherever they fall (up to 1.2 s) rather than a fixed 0.35 s.
        let credible_move = |ki: usize, p0: DVec3, t0: f64| -> bool {
            let moving: Vec<&(f64, DVec3)> = observed[ki]
                .iter()
                .filter(|&&(t, p)| t >= t0 && t <= t0 + 1.2 && (p - p0).length() > 0.025)
                .take(6)
                .collect();
            if moving.len() < 3 {
                return false; // one or two stray frames don't prove a strike
            }
            !observed.iter().enumerate().any(|(j, other)| {
                if j == ki || other.is_empty() {
                    return false;
                }
                let on = moving
                    .iter()
                    .filter(|&&&(t, p)| interp_at(other, t).is_some_and(|q| (q - p).length() < OVERLAP))
                    .count();
                on * 4 >= moving.len() * 3 // mostly on another ball = a swap
            })
        };
        let first_move = observed
            .iter()
            .enumerate()
            .skip(1)
            .map(|(ki, tr)| {
                let Some(&(_, p0)) = tr.first() else { return FirstMove::Unreliable };
                let mut tainted = false;
                for i in 1..tr.len().saturating_sub(1) {
                    let d0 = (tr[i].1 - p0).length();
                    let d1 = (tr[i + 1].1 - p0).length();
                    if d0 > 0.02 && d1 > d0 {
                        // The last sample before the motion — the contact lies
                        // in (t_prev, t], however wide detection blur made it.
                        let t_prev = tr[i - 1].0;
                        if let Some(tc) = explainable(ki, p0, t_prev, tr[i].0) {
                            if credible_move(ki, p0, tr[i].0) {
                                return FirstMove::At(tc);
                            }
                            // A physically-explainable motion claim that fails
                            // the credibility test poisons the whole track's
                            // testimony: we can neither demand nor forbid a
                            // contact for this ball.
                            tainted = true;
                        }
                        // artifact — keep scanning for a later, explainable event
                    }
                }
                if tainted { FirstMove::Unreliable } else { FirstMove::Still }
            })
            .collect();
        // A V beside a rail proves a reversal, not its cause: colliding with a
        // ball that happens to rest near that rail reverses the path just the
        // same. Claim the rail only when no other ball was within reach at that
        // moment — dropping the claim is safe (an unclaimed real bounce charges
        // nothing), fabricating one would punish every faithful candidate.
        let cushions = observed
            .iter()
            .enumerate()
            .map(|(ki, tr)| {
                observed_bounces(tr, bounds)
                    .into_iter()
                    .filter(|&(tm, _, pm)| {
                        !observed.iter().enumerate().any(|(j, other)| {
                            j != ki
                                && other.iter().any(|&(t, p)| {
                                    (t - tm).abs() < 0.2 && (p - pm).length() < REACH
                                })
                        })
                    })
                    .map(|(t, rail, _)| (t, rail))
                    .collect()
            })
            .collect();
        Self { first_move, cushions, bounds, tracks: observed.to_vec() }
    }
}

/// A simulation's cushion bounces, per ball, as (time, rail) — the rail read
/// from where the ball actually is at the event (exact in a simulation).
fn sim_bounces(sim: &Simulation, n_balls: usize, bounds: [f64; 4]) -> Vec<Vec<(f64, u8)>> {
    let mut per_ball: Vec<Vec<(f64, u8)>> = vec![Vec::new(); n_balls];
    for e in &sim.events {
        let billiards_core::ContactKind::Cushion { ball } = e.kind else { continue };
        let bi = ball.0 as usize;
        if bi >= n_balls {
            continue;
        }
        let p = sim.trajectories[bi].state_at(e.time).pos;
        let rail = (0..4u8)
            .min_by(|&a, &b| rail_dist(p, a, bounds).total_cmp(&rail_dist(p, b, bounds)))
            .unwrap();
        per_ball[bi].push((e.time, rail));
    }
    per_ball
}

/// Penalty (meters, added to the RMS) for a simulation whose event structure
/// contradicts the observed skeleton. Ball contacts: hitting a ball that
/// verifiably sat still is a FABRICATED collision; leaving an observably-struck
/// ball untouched is a MISSING one; matching contacts should also roughly match
/// in time. Cushions: every observed bounce must appear in the sim (same ball,
/// same rail, near in time), and a sim bounce the tracks positively contradict
/// is charged too. This is what keeps the fit *causally* faithful instead of
/// merely shape-matching.
pub fn event_penalty(sim: &Simulation, events: &ObservedEvents) -> f64 {
    let mut pen = 0.0;
    for (k, &obs) in events.first_move.iter().enumerate() {
        let ti = k + 1; // trajectory index (cue-first)
        if ti >= sim.trajectories.len() {
            continue;
        }
        let tr = &sim.trajectories[ti];
        let p0 = tr.state_at(0.0).pos;
        let mut sim_move: Option<f64> = None;
        let mut t = 0.0;
        let end = sim.settled_time();
        while t <= end {
            if (tr.state_at(t).pos - p0).length() > 0.02 {
                sim_move = Some(t);
                break;
            }
            t += 1.0 / 30.0;
        }
        match (obs, sim_move) {
            // A contact the data rules out must be effectively vetoed — a
            // slightly rougher faithful path always beats an invented physical
            // event (the RMS scale is ~0.1-0.5 m, so 1.5 m is decisive).
            (FirstMove::Still, Some(_)) => pen += 1.50,   // fabricated contact
            (FirstMove::At(_), None) => pen += 1.50,      // missing contact
            (FirstMove::At(to), Some(ts)) => pen += 0.5 * (to - ts).abs().min(0.6),
            (FirstMove::Still, None) => {}
            // A corrupted track (identity swap) testifies to nothing.
            (FirstMove::Unreliable, _) => {}
        }
    }

    // Cushion skeleton: the rails each ball verifiably bounced off, in time.
    // A candidate whose path merely *looks* similar while telling a different
    // physical story — a different rail, a bounce that never happened, an
    // observed bounce it skips — is a wrong reconstruction; each such element
    // error costs more than the RMS a wrong-family path typically saves.
    const CUSHION_PEN: f64 = 0.70; // per missing/fabricated bounce (decisive vs ~0.1–0.5 m RMS)
    const MATCH_W: f64 = 1.0; // obs↔sim bounces pair up within this (s) — physics drift slack
    // "Observed clearly away from that rail" — generous, because near-rail
    // positions carry the worst calibration/parallax bias: a ball measured
    // 20 cm out may well have touched. Only unambiguous absence convicts.
    const EXCLUDE_D: f64 = 0.25;
    let n = events.cushions.len().min(sim.trajectories.len());
    let sim_b = sim_bounces(sim, n, events.bounds);
    for k in 0..n {
        // ORDER-PRESERVING alignment (longest common subsequence over the two
        // bounce lists; a pair matches on same rail within MATCH_W). Greedy
        // per-event pairing was order-blind: a fabricated early bounce could
        // hide behind a real later bounce off the same rail, letting a
        // rail-first path impersonate a ball-first one. Sequence order IS the
        // physical story — an out-of-order match is no match.
        let obs = &events.cushions[k];
        let simk = &sim_b[k];
        let (no, ns) = (obs.len(), simk.len());
        let pair = |i: usize, j: usize| -> bool {
            obs[i].1 == simk[j].1 && (obs[i].0 - simk[j].0).abs() <= MATCH_W
        };
        let mut dp = vec![vec![0usize; ns + 1]; no + 1];
        for i in (0..no).rev() {
            for j in (0..ns).rev() {
                dp[i][j] = dp[i + 1][j].max(dp[i][j + 1]);
                if pair(i, j) {
                    dp[i][j] = dp[i][j].max(dp[i + 1][j + 1] + 1);
                }
            }
        }
        let mut used = vec![false; ns];
        let (mut i, mut j) = (0, 0);
        while i < no && j < ns {
            if pair(i, j) && dp[i][j] == dp[i + 1][j + 1] + 1 {
                used[j] = true;
                i += 1;
                j += 1;
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                i += 1;
            } else {
                j += 1;
            }
        }
        // Matched bounces cost nothing (their timing error is already in the
        // RMS); every observed bounce the sim skipped is a missing element.
        pen += CUSHION_PEN * (no - dp[0][0]) as f64;
        for (i, &(ts, rail)) in simk.iter().enumerate() {
            if used[i] {
                continue;
            }
            // A sim-only bounce is fabricated only when the track POSITIVELY
            // contradicts it: the ball was observed around that time and stayed
            // far from that rail. No samples there (gap, track ended) ⇒ free —
            // absence of evidence must not veto the physics.
            let mut seen = 0usize;
            let contradicted = events.tracks[k]
                .iter()
                .filter(|(t, _)| (*t - ts).abs() <= MATCH_W)
                .all(|&(_, p)| {
                    seen += 1;
                    rail_dist(p, rail, events.bounds) > EXCLUDE_D
                });
            if seen >= 5 && contradicted {
                pen += CUSHION_PEN;
            }
        }
    }
    pen
}

/// RMS position error between a simulation and the observed tracks.
fn total_error(sim: &Simulation, observed: &[ObservedTrack]) -> f64 {
    let mut sse = 0.0;
    let mut n = 0usize;
    for (i, track) in observed.iter().enumerate() {
        if i >= sim.trajectories.len() {
            break;
        }
        for &(t, p) in track {
            sse += (sim.trajectories[i].state_at(t).pos - p).length_squared();
            n += 1;
        }
    }
    if n == 0 { f64::INFINITY } else { (sse / n as f64).sqrt() }
}

fn eval(
    scene: &Scene,
    p: [f64; 4],
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    observed: &[ObservedTrack],
    events: &ObservedEvents,
) -> f64 {
    let action = CueAction::from_tip_offset(p[0], p[1], p[2], p[3], ball.radius);
    let sim = simulate(&scene.ball_states(&action), table, ball, phys);
    total_error(&sim, observed) + event_penalty(&sim, events)
}

/// The cue ball's launch — (heading radians, speed m/s) — read directly off its
/// initial free flight. A level shot travels straight on the cloth regardless of
/// spin, so the free-flight heading *is* the aim; the speed over the same window
/// is the launch speed. Both are observed and physics-independent, so they anchor
/// the fit instead of floating free.
///
/// The window is bounded by **geometry and sufficiency, not a fixed distance**:
/// a fixed cutoff (an earlier version used 18 cm) leaves a fast ball with a
/// single blurred strike-frame inside it — exactly the degenerate estimate it
/// was meant to avoid — while a pure time cap lets a slow, swerving ball curve
/// for a meter and drag the chord off the launch heading. Samples accumulate
/// until there are *enough* (≥3 spanning ≥20 cm — blur averaged, curvature not
/// yet significant), stopping early if the cue (a) nears an object ball's rest
/// (raw first-detection or corrected — contact imminent), (b) approaches a rail
/// it didn't start beside, or (c) bends off the running chord (a bounce). The
/// heading is the chord start→last free sample; the speed is the least-squares
/// slope of displacement over time through the rest point, so the blurred first
/// frame is one vote among many rather than the whole answer.
/// Returns `(heading, speed, heading_spread)`: the spread is the largest angular
/// deviation of any single free-flight sample's chord from the final heading — a
/// direct measure of how well the launch direction was actually observed. One
/// blurred sample or a curving path yields a large spread; a long clean baseline
/// a small one. The fit widens its aim window to this spread, so shots with a
/// genuinely ambiguous launch aren't clamped to a false certainty.
fn launch_estimate(
    cue: &[(f64, DVec3)],
    obstacles: &[DVec3],
    t_stop: f64,
    bounds: [f64; 4],
    radius: f64,
) -> (f64, f64, f64) {
    const MIN_MOVE: f64 = 0.015; // ignore sub-1.5 cm wobble as the ball leaves rest
    const MAX_T: f64 = 0.45; // longest usable launch window (very slow shots)
    const ENOUGH_N: usize = 3; // stop once the chord has this many samples…
    const ENOUGH_D: f64 = 0.20; // …spanning at least this much travel (m)
    const RAIL_GUARD: f64 = 0.035; // stop short of an approaching rail (m)
    let Some(&(t0, p0)) = cue.first() else { return (0.0, 2.0, 0.12) };
    // Proximity cut-off for an object contact: generous (2R + 2.5 cm), because a
    // noisy rest can sit ~2-3 cm off and the collision must still terminate the
    // window. `t_stop` (first observed object motion) backs this up exactly.
    let guard = 2.0 * radius + 0.025;
    let [min_x, max_x, min_y, max_y] = bounds;
    // Rails the ball starts beside don't end the window (it moves away or along).
    let start_near = [
        p0.x - min_x < RAIL_GUARD,
        max_x - p0.x < RAIL_GUARD,
        p0.y - min_y < RAIL_GUARD,
        max_y - p0.y < RAIL_GUARD,
    ];

    let mut t_rest = t0; // last sample still within jitter of the start
    let mut free: Vec<(f64, DVec3)> = Vec::new();
    for &(t, p) in cue {
        let d = p - p0;
        let dl = d.length();
        if dl < MIN_MOVE {
            t_rest = t;
            continue;
        }
        if t - t0 > MAX_T {
            break;
        }
        if t >= t_stop - 0.017 {
            // An object ball has started moving by here, so a collision already
            // happened — this sample is post-impact, whatever the (noisy) rest
            // positions claim. Half a frame of margin covers frame quantization.
            break;
        }
        if obstacles.iter().any(|o| (p - *o).length() < guard) {
            break; // at/inside an object contact — no longer free flight
        }
        let near = [
            p.x - min_x < RAIL_GUARD,
            max_x - p.x < RAIL_GUARD,
            p.y - min_y < RAIL_GUARD,
            max_y - p.y < RAIL_GUARD,
        ];
        if (0..4).any(|k| near[k] && !start_near[k]) {
            break; // approaching a rail — a bounce is imminent
        }
        if let Some(&(_, last)) = free.last() {
            // Bend test against the running chord: a cross-track jump beyond ~18°
            // of the accumulated displacement is a bounce between samples. Kept
            // tolerant — real swerve curls ~10° across the first samples, and the
            // rail/object guards already catch geometric contacts; this is only a
            // backstop for a bounce the geometry checks somehow missed.
            let chord = (last - p0).normalize();
            let cross = (d - chord * d.dot(chord)).length();
            if cross > (0.31 * dl).max(0.025) {
                break;
            }
        }
        free.push((t, p));
    }

    if free.is_empty() {
        // Nothing usable (e.g. the ball is instantly at an obstacle): fall back
        // to the first sample past jitter so we still return *a* heading — with a
        // wide spread, because one strike-blurred frame barely observes it.
        let fallback = cue.iter().find(|(_, p)| (*p - p0).length() >= MIN_MOVE);
        return match fallback {
            Some(&(t, p)) => {
                let dv = p - p0;
                (dv.y.atan2(dv.x), (dv.length() / (t - t_rest).max(1e-6)).clamp(0.1, 8.0), 0.12)
            }
            None => (0.0, 2.0, 0.12),
        };
    }
    // Heading: chord to the first sample that makes the baseline *sufficient*
    // (≥ ENOUGH_N samples spanning ≥ ENOUGH_D) — long enough to average strike
    // blur, short enough that swerve hasn't bent the path off the launch heading.
    // If the flight ends before sufficiency, use everything there is.
    let mut end = free.len() - 1;
    for (i, &(_, p)) in free.iter().enumerate() {
        if i + 1 >= ENOUGH_N && (p - p0).length() >= ENOUGH_D {
            end = i;
            break;
        }
    }
    let dv = free[end].1 - p0;
    let heading = dv.y.atan2(dv.x);
    // How well is that heading actually pinned down? Largest angular deviation of
    // any window sample's own chord from the final heading (plus a floor when the
    // window is a single sample, which observes almost nothing).
    let mut spread = if end == 0 { 0.10_f64 } else { 0.0_f64 };
    for &(_, p) in &free[..=end] {
        let d = p - p0;
        let a = (d.y.atan2(d.x) - heading).sin().abs().asin(); // |wrapped| angle
        spread = spread.max(a);
    }
    // Launch speed: least-squares slope through the origin (the rest moment) of
    // displacement vs time — over the *first* ~0.17 s of free flight, regardless
    // of the heading span. The heading wants the longest clean baseline; the
    // speed wants the shortest that clears the blur, because a follow ball
    // accelerates to natural roll and a sliding ball decelerates — the launch
    // value exists only near the strike. (Always keep ≥2 samples for a slope.)
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &(t, p)) in free.iter().enumerate() {
        let dt = t - t_rest;
        if dt > 0.17 && i >= 2 {
            break;
        }
        num += dt * (p - p0).length();
        den += dt * dt;
    }
    let speed = if den > 1e-9 { (num / den).clamp(0.1, 8.0) } else { 2.0 };
    (heading, speed, spread)
}

/// Wrap an angle difference to (-π, π].
fn wrap_angle(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// A struck ball's departure direction: the chord from its rest position out to
/// ~20 cm of its early travel (skipping samples still at the rest). This is the
/// line of centers of the collision (± throw), observed with a far longer clean
/// baseline than the cue's own pre-contact flight on an immediate collision.
fn departure_dir(track: &[(f64, DVec3)], rest: DVec3) -> Option<DVec3> {
    let mut best: Option<DVec3> = None;
    for &(_, p) in track {
        let dl = (p - rest).length();
        if dl > 0.05 {
            best = Some(p);
        }
        if dl > 0.20 {
            break;
        }
    }
    let d = best? - rest;
    (d.length() > 0.05).then(|| d / d.length())
}

/// Reconstruct the cue action from observed tracks (index 0 = cue ball).
///
/// Two-stage: the strict pass pins aim/speed to the directly-observed launch
/// and spin to the miscue disk. When that pass CANNOT explain the data (high
/// residual), the pins' inputs are the prime suspects — launch estimates come
/// from a few blur-prone frames — so a relaxed pass (wider aim/speed windows,
/// effective spin box beyond nominal miscue) runs and wins if it fits the
/// tracks materially better. Clean shots never trigger it.
pub fn fit_action(
    scene: &Scene,
    observed: &[ObservedTrack],
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &FitConfig,
) -> FitResult {
    // Judge each pass on the CUE's own error: object tracks can carry constant
    // corruption (phantom detections) that no action changes, which would mask
    // both the trigger and the comparison.
    let cue_rms = |r: &FitResult| {
        let sim = simulate(&scene.ball_states(&r.action), table, ball, phys);
        total_error(&sim, &observed[..1])
    };
    let strict = fit_action_pass(scene, observed, table, ball, phys, cfg);
    let strict_cue = cue_rms(&strict);
    if strict_cue <= 0.30 {
        return strict;
    }
    let relaxed_cfg = FitConfig {
        aim_window: (cfg.aim_window * 4.0).max(0.10),
        speed_window: (cfg.speed_window * 1.6).min(0.5),
        // effective spin can exceed nominal miscue: real strokes extract more
        // spin per offset than the simple tip map (elevation, acceleration)
        miscue_limit: 0.72,
        ..*cfg
    };
    let relaxed = fit_action_pass(scene, observed, table, ball, phys, &relaxed_cfg);
    if cue_rms(&relaxed) < strict_cue * 0.8 {
        relaxed
    } else {
        strict
    }
}

fn fit_action_pass(
    scene: &Scene,
    observed: &[ObservedTrack],
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    cfg: &FitConfig,
) -> FitResult {
    // Tip-offset grid points inside the miscue disk.
    let lim = cfg.miscue_limit;
    let mut offsets = Vec::new();
    for hi in 0..cfg.offset_steps {
        let h = lerp(-lim, lim, hi, cfg.offset_steps);
        for vi in 0..cfg.offset_steps {
            let v = lerp(-lim, lim, vi, cfg.offset_steps);
            if h * h + v * v <= lim * lim + 1e-9 {
                offsets.push((h, v));
            }
        }
    }

    // The cue's pre-collision heading IS the aim and its free-flight speed IS the
    // launch speed — both directly observed and physics-independent. Search only a
    // narrow window around each; letting them float lets the fit absorb physics
    // error into a wrong aim/speed (a reconstruction that visibly disagrees with
    // the tracked shot's direction and how hard it was hit).
    //
    // The launch window is bounded by where a contact could occur: both the scene
    // rests (possibly back-extrapolated) and the raw first detections count as
    // obstacles, since either may be closer to the true contact point. The moment
    // any object ball first *moves* bounds the window exactly (position-noise
    // free): the cue's free flight is over by then.
    let mut events = ObservedEvents::from_tracks(observed, table.center_bounds(ball.radius));
    if !cfg.skeleton {
        // Legacy scoring: no observed bounces to demand, no tracks to
        // contradict a simulated one — the cushion terms all go silent.
        events.cushions.iter_mut().for_each(Vec::clear);
        events.tracks.iter_mut().for_each(Vec::clear);
    }
    let mut obstacles: Vec<DVec3> = scene.objects.clone();
    obstacles.extend(observed.iter().skip(1).filter_map(|t| t.first().map(|s| s.1)));
    let moves: Vec<(usize, f64)> = observed
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(k, tr)| {
            let &(_, p0) = tr.first()?;
            tr.iter().find(|(_, p)| (*p - p0).length() > 0.02).map(|&(t, _)| (k, t))
        })
        .collect();
    let t_stop = moves.iter().map(|&(_, t)| t).fold(f64::INFINITY, f64::min);
    let (mut aim0, speed0, spread) =
        launch_estimate(&observed[0], &obstacles, t_stop, table.center_bounds(ball.radius), ball.radius);
    // The aim window is the *observed* heading uncertainty: at least the caller's
    // window, wider when the free flight was blurred/curved/short — capped so the
    // fit can never wander far from the launch direction that was actually seen.
    let mut aim_window = cfg.aim_window.max(spread.min(0.12));

    // Immediate collision (the cue reaches a ball within a frame or two): the
    // cue's own 1-3 blurred samples barely observe the aim, but the STRUCK
    // ball's departure observes it well — an object leaves along the line of
    // centers, so its departure direction + rest position locate the cue's
    // center at the moment of contact, and cue-start → contact-center IS the
    // aim (± throw, which the window absorbs). Used only when the cue chord is
    // genuinely uncertain and the two estimates roughly agree (a wild
    // disagreement means the wrong ball or a rail-first path — trust the direct
    // observation then).
    if spread > 0.05 {
        let t0 = observed[0].first().map_or(0.0, |s| s.0);
        let first = moves
            .iter()
            .filter(|&&(_, t)| t - t0 < 0.8)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some(&(k, _)) = first {
            if let Some(&rest) = scene.objects.get(k - 1) {
                // A ball resting near a cushion may bank off it immediately, so
                // its observed departure already contains the bounce and points
                // to the wrong contact side — the geometry only holds in the open.
                let [min_x, max_x, min_y, max_y] = table.center_bounds(ball.radius);
                let clear = (rest.x - min_x).min(max_x - rest.x).min(rest.y - min_y).min(max_y - rest.y) > 0.10;
                if let Some(dir) = departure_dir(&observed[k], rest).filter(|_| clear) {
                    let contact = rest - dir * (2.0 * ball.radius);
                    let to = contact - scene.cue;
                    if to.length() > 0.08 {
                        let aim_geo = to.y.atan2(to.x);
                        let dd = wrap_angle(aim_geo - aim0);
                        if dd.abs() < 0.35 {
                            // Center between the two estimates with BOTH inside
                            // the window: the fit itself (whole-trajectory RMS)
                            // arbitrates which observation to trust.
                            aim0 = wrap_angle(aim0 + dd / 2.0);
                            aim_window = cfg.aim_window.max((dd.abs() / 2.0 + 0.04).min(0.20));
                        }
                    }
                }
            }
        }
    }
    // The launch chord is a handful of blur-prone samples; a credible early
    // object motion is causal certainty (nothing else was moving — the cue
    // struck THAT ball). If the aim window cannot even GRAZE the first-struck
    // ball, the chord is the corrupted measurement (strike blur, the player
    // occluding the corner) — so long as no cue cushion is observed before the
    // contact (a rail-first path reaches balls the chord can't). Don't replace
    // the window; ENLARGE it to span both hypotheses and let the whole-track
    // error + event penalty arbitrate.
    let mut unpin_speed = false;
    {
        let t0 = observed[0].first().map_or(0.0, |s| s.0);
        let first_struck = events
            .first_move
            .iter()
            .enumerate()
            .filter_map(|(k, fm)| match fm {
                FirstMove::At(t) => Some((k, *t)),
                _ => None,
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|&(_, t)| t - t0 < 0.8);
        if let Some((k, t_hit)) = first_struck {
            let cue_banked_first = events.cushions[0].iter().any(|&(tb, _)| tb < t_hit);
            if let Some(&rest) = scene.objects.get(k).filter(|_| !cue_banked_first) {
                let to = rest - scene.cue;
                if to.length() > 0.08 {
                    let aim_ball = to.y.atan2(to.x);
                    // half-angle subtended by a grazing contact
                    let graze = (2.0 * ball.radius / to.length()).min(1.0).asin();
                    let dd = wrap_angle(aim_ball - aim0);
                    // A ball that verifiably left its rest took a real hit, not
                    // an edge-of-cone kiss. If the window can't even reach well
                    // into the contact cone, the chord AND its speed slope are
                    // blur-corrupt — open the speed to the full range; the
                    // event skeleton and whole-track error keep it honest.
                    if dd.abs() > aim_window + 0.7 * graze {
                        unpin_speed = true;
                    }
                    // Same verdict from energy: an equal-mass collision cannot
                    // send the struck ball out faster than the cue arrived. If
                    // its departure speed crowds the pinned launch ceiling
                    // (which the cue only reaches BEFORE cloth friction), the
                    // measured launch speed is blur-suppressed.
                    let dep_speed = {
                        let tr = &observed[k + 1];
                        let rest = tr.first().map_or(rest, |s| s.1);
                        let mut mv = tr.iter().filter(|&&(_, p)| (p - rest).length() > 0.025);
                        match (mv.next(), mv.next()) {
                            (Some(&(t1, p1)), Some(&(t2, p2))) if t2 > t1 => {
                                ((p2 - rest).length() - (p1 - rest).length()) / (t2 - t1)
                            }
                            _ => 0.0,
                        }
                    };
                    if dep_speed > 0.8 * (speed0 * (1.0 + cfg.speed_window)) {
                        unpin_speed = true;
                    }
                    // Either way, the fit must be free to choose the contact
                    // THICKNESS: a chord biased thin by a few cm otherwise
                    // forbids the (thick) hit that actually happened. Enlarge
                    // the window to the union of the observed-heading window
                    // and the verified ball's full contact cone.
                    let lo = (-aim_window).min(dd - graze - 0.03);
                    let hi = aim_window.max(dd + graze + 0.03);
                    aim0 = wrap_angle(aim0 + (lo + hi) / 2.0);
                    aim_window = (hi - lo) / 2.0;
                }
            }
        }
    }
    let speed_lo = if unpin_speed { cfg.speed_min } else { (speed0 * (1.0 - cfg.speed_window)).max(cfg.speed_min) };
    let speed_hi = if unpin_speed { cfg.speed_max } else { (speed0 * (1.0 + cfg.speed_window)).min(cfg.speed_max) };

    // 1. Grid, parallel over aim (order preserved for determinism).
    let grid: Vec<([f64; 4], f64)> = (0..cfg.aim_steps)
        .into_par_iter()
        .flat_map_iter(|ai| {
            let aim = aim0 - aim_window
                + 2.0 * aim_window * ai as f64 / (cfg.aim_steps.max(2) - 1) as f64;
            let mut local = Vec::with_capacity(cfg.speed_steps * offsets.len());
            for si in 0..cfg.speed_steps {
                let speed = lerp(speed_lo, speed_hi, si, cfg.speed_steps);
                for &(h, v) in &offsets {
                    let p = [aim, speed, h, v];
                    local.push((p, eval(scene, p, table, ball, phys, observed, &events)));
                }
            }
            local
        })
        .collect();

    // 2. Multi-start coordinate-descent refinement from the best grid candidates.
    // The aim is directly observed (it's the cue's pre-collision heading), so keep
    // refinement inside the same window around `aim0`: otherwise it walks the aim
    // off the real heading to absorb physics-model error, giving a reconstruction
    // whose *starting direction* visibly disagrees with the tracked shot.
    let aim_bounds = (aim0 - aim_window, aim0 + aim_window);
    let speed_bounds = (speed_lo, speed_hi);
    let mut grid = grid;
    grid.sort_by(|a, b| a.1.total_cmp(&b.1));
    let k = cfg.multistart.min(grid.len());
    let refined: Vec<([f64; 4], f64)> = grid[..k]
        .par_iter()
        .map(|&(p0, e0)| refine(scene, p0, e0, aim_bounds, speed_bounds, table, ball, phys, observed, &events, cfg))
        .collect();
    let (p, _err) = refined
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty candidates");

    // Rank by RMS+event-penalty (causal faithfulness), but REPORT the pure
    // position error — "fit error" must mean what it says.
    let action = CueAction::from_tip_offset(p[0], p[1], p[2], p[3], ball.radius);
    let sim = simulate(&scene.ball_states(&action), table, ball, phys);
    FitResult { action, rms_m: total_error(&sim, observed) }
}

/// Coordinate descent with shrinking steps from a single start point. Aim is
/// kept within `aim_bounds` (the observed-heading window) so the fit can't trade
/// a wrong starting direction for a lower downstream error.
fn refine(
    scene: &Scene,
    mut p: [f64; 4],
    mut err: f64,
    aim_bounds: (f64, f64),
    speed_bounds: (f64, f64),
    table: &TableSpec,
    ball: &BallSpec,
    phys: &PhysicsParams,
    observed: &[ObservedTrack],
    events: &ObservedEvents,
    cfg: &FitConfig,
) -> ([f64; 4], f64) {
    let mut steps = [0.04, 0.2, 0.06, 0.06]; // aim(rad), speed(m/s), h, v
    for _ in 0..cfg.refine_iters {
        let mut improved = false;
        for k in 0..4 {
            for dir in [1.0, -1.0] {
                let mut cand = p;
                cand[k] += dir * steps[k];
                if k == 0 && (cand[0] < aim_bounds.0 || cand[0] > aim_bounds.1) {
                    continue; // don't let the aim leave the observed-heading window
                }
                if k == 1 && (cand[1] < speed_bounds.0 || cand[1] > speed_bounds.1) {
                    continue; // keep the speed within the observed launch-speed window
                }
                if (cand[2] * cand[2] + cand[3] * cand[3]).sqrt() > cfg.miscue_limit {
                    continue;
                }
                let e = eval(scene, cand, table, ball, phys, observed, &events);
                if e < err {
                    p = cand;
                    err = e;
                    improved = true;
                }
            }
        }
        if !improved {
            for s in &mut steps {
                *s *= 0.5;
            }
        }
    }
    (p, err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (BallSpec, PhysicsParams, TableSpec, Scene) {
        let ball = BallSpec::carom();
        let r = ball.radius;
        let scene = Scene::new(
            DVec3::new(-1.0, -0.3, r),
            vec![DVec3::new(0.5, 0.15, r), DVec3::new(0.9, -0.25, r)],
        );
        (ball, PhysicsParams::default(), TableSpec::carom_match(), scene)
    }

    /// Every bounce read from sampled tracks corresponds to a real sim bounce
    /// (same ball, same rail, close in time), the cue's early bounces are all
    /// found, and the truth's own event penalty is ~zero.
    #[test]
    fn skeleton_matches_simulation() {
        let (ball, phys, table, scene) = setup();
        let truth = CueAction::from_tip_offset(0.29, 6.0, 0.20, 0.15, ball.radius);
        let sim = simulate(&scene.ball_states(&truth), &table, &ball, &phys);
        let bounds = table.center_bounds(ball.radius);
        let observed = sample_tracks(&sim, 30.0);
        let events = ObservedEvents::from_tracks(&observed, bounds);

        let truth_b = sim_bounces(&sim, observed.len(), bounds);
        assert!(
            truth_b[0].len() >= 2,
            "test shot must bank (got {} cue bounces)",
            truth_b[0].len()
        );
        for (k, obs) in events.cushions.iter().enumerate() {
            for &(to, rail) in obs {
                assert!(
                    truth_b[k].iter().any(|&(ts, r)| r == rail && (ts - to).abs() < 0.2),
                    "extracted bounce (ball {k}, t {to:.2}, rail {rail}) has no sim counterpart"
                );
            }
        }
        // Firm bounces (real perpendicular impact speed) must be seen; slow
        // grazes may honestly go unclaimed — the detector is conservative.
        let found = &events.cushions[0];
        for &(ts, rail) in &truth_b[0] {
            let v = sim.trajectories[0].state_at(ts - 1e-4).vel;
            let v_perp = if rail < 2 { v.x.abs() } else { v.y.abs() };
            if v_perp > 0.8 {
                assert!(
                    found.iter().any(|&(to, r)| r == rail && (ts - to).abs() < 0.2),
                    "missed the cue's rail-{rail} bounce at t {ts:.2} (v⊥ {v_perp:.2})"
                );
            }
        }
        assert!(event_penalty(&sim, &events) < 0.1, "truth should not be penalized");
    }

    /// An action whose path skips the observed bounces / contact is decisively
    /// penalized, and one that fabricates a bounce the tracks contradict is too.
    #[test]
    fn wrong_skeleton_is_penalized() {
        let (ball, phys, table, scene) = setup();
        let truth = CueAction::from_tip_offset(0.29, 3.5, 0.20, 0.15, ball.radius);
        let sim = simulate(&scene.ball_states(&truth), &table, &ball, &phys);
        let bounds = table.center_bounds(ball.radius);
        let events = ObservedEvents::from_tracks(&sample_tracks(&sim, 30.0), bounds);

        // Too soft: never reaches the rails the real shot bounced off.
        let soft = CueAction::from_tip_offset(0.29, 1.0, 0.0, 0.0, ball.radius);
        let soft_sim = simulate(&scene.ball_states(&soft), &table, &ball, &phys);
        assert!(
            event_penalty(&soft_sim, &events) > 0.6,
            "missing observed bounces must be decisive"
        );

        // Fabricated: the real shot rolls gently and stops in the open; a hard
        // bank crosses rails the tracks show the cue was never near.
        let gentle = CueAction::from_tip_offset(0.0, 1.2, 0.0, 0.0, ball.radius);
        let gentle_sim = simulate(&scene.ball_states(&gentle), &table, &ball, &phys);
        let gentle_ev = ObservedEvents::from_tracks(&sample_tracks(&gentle_sim, 30.0), bounds);
        let bank = CueAction::from_tip_offset(std::f64::consts::PI / 2.0, 4.5, 0.0, 0.0, ball.radius);
        let bank_sim = simulate(&scene.ball_states(&bank), &table, &ball, &phys);
        assert!(
            event_penalty(&bank_sim, &gentle_ev) > 0.6,
            "a bounce the tracks contradict must be decisive"
        );
    }
}

/// Sample every ball's trajectory at `fps` into observed tracks — the shape the
/// tracker produces, and what the synthetic fit test feeds back in.
pub fn sample_tracks(sim: &Simulation, fps: f64) -> Vec<ObservedTrack> {
    sim.trajectories
        .iter()
        .map(|traj| {
            let total = traj.time_to_rest();
            let n = (total * fps).ceil() as usize;
            (0..=n)
                .map(|k| {
                    let t = (k as f64 / fps).min(total);
                    (t, traj.state_at(t).pos)
                })
                .collect()
        })
        .collect()
}
