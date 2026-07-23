//! Track-measurement utilities shared by the event-local calibration tools
//! (`cushion_events`, `collision_events`).
//!
//! The one non-obvious contract: these estimators define the **measurement
//! operator** that turns true ball motion into "what a 30 fps tracker reports".
//! Event-local fits must push the *model* through the same operator (simulate
//! the post-event motion, sample it at the real frame times, run the same
//! estimator) rather than un-bias the measurement — post-event sliding decays
//! ~20× faster than rolling, and how much of it the window sees depends on the
//! very parameters being fit.

use billiards_core::math::DVec3;

/// Least-squares linear velocity over `pts` (time, position), plus the window
/// mid time and the cross-track residual RMS — deviation perpendicular to the
/// fitted direction. Deceleration is along-track and doesn't inflate the
/// cross-track term; a kink (an unnoticed contact) or a curve does, so gate on
/// it. Requires ≥3 samples spanning ≥0.10 s with no internal gap > 0.12 s.
pub fn window_velocity(pts: &[(f64, DVec3)]) -> Option<(DVec3, f64, f64)> {
    if pts.len() < 3 || pts.last()?.0 - pts.first()?.0 < 0.10 {
        return None;
    }
    if pts.windows(2).any(|w| w[1].0 - w[0].0 > 0.12) {
        return None;
    }
    let n = pts.len() as f64;
    let tm = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let pm = pts.iter().fold(DVec3::ZERO, |a, p| a + p.1) / n;
    let (mut num, mut den) = (DVec3::ZERO, 0.0);
    for &(t, p) in pts {
        num += (p - pm) * (t - tm);
        den += (t - tm) * (t - tm);
    }
    if den <= 1e-9 {
        return None;
    }
    let v = num / den;
    let speed = v.length();
    if speed < 1e-6 {
        return None;
    }
    let u = v / speed;
    let cross_sse: f64 = pts
        .iter()
        .map(|&(t, p)| {
            let r = p - (pm + v * (t - tm));
            (r - u * r.dot(u)).length_squared()
        })
        .sum();
    Some((v, tm, (cross_sse / n).sqrt()))
}

/// The line fit behind [`window_velocity`], exposed for geometry: mean time,
/// mean position, and velocity — `p(t) = pm + v·(t − tm)`.
pub fn window_line(pts: &[(f64, DVec3)]) -> Option<(f64, DVec3, DVec3)> {
    let (v, tm, _) = window_velocity(pts)?;
    let n = pts.len() as f64;
    let pm = pts.iter().fold(DVec3::ZERO, |a, p| a + p.1) / n;
    Some((tm, pm, v))
}
