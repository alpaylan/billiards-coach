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
    /// Coefficient of spinning ("boring") friction that decays vertical english.
    pub mu_spin: f64,
    /// Coefficient of restitution of the cushion (Han 2005 `e_c`). ~0.85 per
    /// van Balen; a calibration target for Phase 4.
    pub cushion_restitution: f64,
    /// Coefficient of ball–cushion friction (Han 2005 `f_c`).
    pub cushion_friction: f64,
    /// Coefficient of restitution between two balls (`e_b`).
    pub ball_restitution: f64,
    /// Coefficient of ball–ball friction (`u_b`) — governs collision-induced
    /// and spin-induced throw.
    pub ball_friction: f64,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            g: 9.81,
            mu_slide: 0.2,
            mu_roll: 0.01,
            mu_spin: 0.044,
            cushion_restitution: 0.85,
            cushion_friction: 0.2,
            ball_restitution: 0.95,
            ball_friction: 0.06,
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
