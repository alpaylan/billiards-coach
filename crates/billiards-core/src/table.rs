//! Table geometry, ball specification, and tunable physics parameters.
//!
//! Defaults describe a regulation international three-cushion (carom) setup.
//! The friction coefficients are literature-typical *placeholders*: Phase 4
//! (system identification) will fit them per-table/per-cloth against real
//! reconstructed shots, so treat them as an initial calibration point rather
//! than ground truth.

/// Physical constants of a ball. All three carom balls are identical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallSpec {
    /// Ball radius (m).
    pub radius: f64,
    /// Ball mass (kg).
    pub mass: f64,
}

impl BallSpec {
    /// Regulation carom ball: 61.5 mm diameter, ~210 g.
    pub fn carom() -> Self {
        Self { radius: 0.030_75, mass: 0.210 }
    }

    /// Moment of inertia of a solid sphere, `I = (2/5)·m·r²`.
    pub fn moment_of_inertia(&self) -> f64 {
        0.4 * self.mass * self.radius * self.radius
    }
}

/// Playing-surface geometry, measured *inside* the cushions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableSpec {
    /// Long dimension of the playing surface (m).
    pub length: f64,
    /// Short dimension of the playing surface (m).
    pub width: f64,
    /// Height of the cushion nose above the cloth (m). Not used until the
    /// cushion-rebound milestone, but part of the geometry contract now.
    pub cushion_height: f64,
}

impl TableSpec {
    /// Regulation international match table: 2.84 m × 1.42 m playing area,
    /// measured between the cushion noses. Cushion contact height ~1.2·radius.
    pub fn carom_match() -> Self {
        Self { length: 2.84, width: 1.42, cushion_height: 0.036_9 }
    }

    /// Horizontal distance from a ball's center to the cushion nose line at the
    /// instant of contact. Because the nose sits *above* the ball equator (at
    /// height `cushion_height`), the center contacts short of the nose by
    /// `√(R² − (h − R)²)`.
    pub fn contact_inset(&self, radius: f64) -> f64 {
        let dh = self.cushion_height - radius;
        (radius * radius - dh * dh).max(0.0).sqrt()
    }

    /// Axis-aligned bounds `[min_x, max_x, min_y, max_y]` that a ball *center*
    /// travels within — the nose rectangle shrunk by [`contact_inset`].
    pub fn center_bounds(&self, radius: f64) -> [f64; 4] {
        let inset = self.contact_inset(radius);
        let hx = self.length / 2.0 - inset;
        let hy = self.width / 2.0 - inset;
        [-hx, hx, -hy, hy]
    }
}

/// Tunable parameters of the physics model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsParams {
    /// Gravitational acceleration (m/s²).
    pub g: f64,
    /// Coefficient of sliding (kinetic) friction between ball and cloth.
    pub mu_slide: f64,
    /// Coefficient of rolling resistance between ball and cloth.
    pub mu_roll: f64,
    /// Coefficient of spinning ("boring") friction that decays vertical english
    /// at `5·μ_spin·g/(2R)`. Default 0: on the heated match tables of the
    /// verification corpus any added cloth decay (even pooltool's ~0.014)
    /// measurably worsened reconstruction — the calibrated Han cushion friction
    /// already accounts for the english lost per rebound.
    pub mu_spin: f64,
    /// Coefficient of restitution of the cushion (Han 2005 `e_c`). ~0.85 per
    /// van Balen; a calibration target for Phase 4.
    pub cushion_restitution: f64,
    /// Speed dependence of the cushion restitution: effective
    /// `e_c + slope·(|v_n| − 1)` (clamped to [0.3, 0.99]), so the calibrated
    /// `cushion_restitution` keeps its meaning at 1 m/s. `0` = constant.
    pub cushion_restitution_slope: f64,
    /// Tangential-speed dependence of the cushion restitution: effective
    /// `e += slope_t·|v_t|` where `v_t` is the tangential (along-rail) speed
    /// at impact. `0` = none (default). Exists to express the fast/oblique
    /// regime the event-local harvest measures as more elastic than the slow
    /// perpendicular one; calibrate against cushion_events before use.
    pub cushion_restitution_slope_t: f64,
    /// Chain-restitution delta: a ball arriving at a cushion still disturbed
    /// from its PREVIOUS cushion contact (< ~0.35 s ago) rebounds effectively
    /// hotter than a settled one — measured directly on two-bounce chains
    /// (chain-fit e_c ≈ 0.89 vs isolated ≈ 0.84, masa4_day2). Effective
    /// `e += chain·(1 − Δt/0.35)` where Δt is time since that ball's last
    /// cushion event. `0` = off (default): the flat single-e model.
    pub cushion_restitution_chain: f64,
    /// Coefficient of ball–cushion friction (Han 2005 `f_c`).
    pub cushion_friction: f64,
    /// Ball–cushion resolution: `0` uses the closed-form Han 2005 impulse,
    /// `> 0` integrates the impact Mathavan 2010-style with that step budget —
    /// dual contact (cushion nose + the table under the ball), rotating slip
    /// at both, and work-based compression/restitution phases.
    pub cushion_contact_steps: u32,
    /// Coefficient of restitution between two balls (`e_b`).
    pub ball_restitution: f64,
    /// Ball–ball friction floor `a` in the speed-dependent Alciatore curve
    /// `u_b(v) = a + b·exp(−c·v)`, where `v` is the relative surface speed at
    /// the contact point (m/s). Governs collision- and spin-induced throw.
    /// With `ball_friction_b = 0` this reduces to a constant coefficient.
    pub ball_friction: f64,
    /// Amplitude `b` of the speed-dependent part of the ball–ball friction
    /// curve: slow, grazing contacts are much grippier than fast ones.
    pub ball_friction_b: f64,
    /// Decay rate `c` (s/m) of the speed-dependent ball–ball friction curve.
    pub ball_friction_c: f64,
    /// Ball–ball contact resolution: `0` uses the closed-form instantaneous
    /// impulse (english-only friction), `> 0` integrates the contact through
    /// that many impulse steps Mathavan-style — the slip direction rotates
    /// during contact and the full 3-D surface velocity (follow/draw as well
    /// as english) drives friction, so spin transfer between balls emerges.
    pub ball_contact_steps: u32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            g: 9.81,
            mu_slide: 0.2,
            mu_roll: 0.01,
            mu_spin: 0.0,
            cushion_restitution: 0.85,
            cushion_restitution_slope: 0.0,
            cushion_restitution_slope_t: 0.0,
            cushion_restitution_chain: 0.0,
            cushion_friction: 0.2,
            cushion_contact_steps: 0,
            ball_restitution: 0.95,
            // Constant coefficient (b = 0): Alciatore's pool-measured curve
            // (a 9.951e-3, b 0.108, c 1.088) and softer/faster variants were
            // all neutral-to-worse on the 607-shot labeled corpus — polished
            // carom balls don't show the pool balls' low-speed grip, and the
            // constant 0.06 sits mid-curve anyway. The curve stays available
            // as a calibration target.
            ball_friction: 0.06,
            ball_friction_b: 0.0,
            ball_friction_c: 1.088,
            ball_contact_steps: 0,
        }
    }
}

impl PhysicsParams {
    /// Parameters calibrated against real tracked three-cushion shots (the 2026
    /// MASA 4 match, `calibrate_shots`) with BOTH the aim and the launch speed
    /// pinned to their directly-observed values — so the fit can't hide physics
    /// error in a rotated/inflated action and the residual is genuine physics.
    ///
    /// Fit on the **1080p** dataset (crisp inset, ~100% detection): with clean
    /// tracks, cushion restitution settles at a physical 0.91 instead of straining
    /// against the search bound as it did on the noisier 720p tracks — evidence the
    /// earlier near-elastic value was a tracking-noise artifact, not real physics.
    ///
    /// Refit on the clock-segmented, **stroke-aligned** shots (t=0 moved to the
    /// stroke so the launch speed the fit reads is the real one, not a pre-stroke
    /// average): `e_c`/`f_c` reproduce almost exactly (0.914/0.297) while the cloth
    /// terms sharpen now that the early slide-to-roll phase is timed correctly.
    ///
    /// NOTE: these are the conservative *fallback* values. Each processed game
    /// carries its own fitted `calibration.json` (see `calibrate_match`), which
    /// the editor and verifier prefer; refits with the corrected loader push
    /// `e_c` to the search bound (~0.97) — best-reproduction values that likely
    /// compensate per-rail energy the cushion model bleeds, so they belong in
    /// per-game files, not baked in as physics.
    pub fn carom_calibrated() -> Self {
        Self {
            cushion_restitution: 0.914,
            cushion_friction: 0.297,
            mu_slide: 0.160,
            mu_roll: 0.0061,
            ..Self::default()
        }
    }
}
