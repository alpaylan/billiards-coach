//! Table-space ↔ screen-space mapping. Kept free of egui types so the geometry
//! is unit-testable on its own.
//!
//! Table space is meters with the table center at the origin, `+x` right and
//! `+y` up. Screen space is pixels with `+y` *down*, so the vertical axis flips.

use billiards_core::TableSpec;

#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Screen pixel at table-space origin.
    pub cx: f32,
    pub cy: f32,
    /// Pixels per meter.
    pub scale: f32,
}

impl View {
    /// Fit the table's playing area into a screen rect (given by center and
    /// size), leaving `margin` pixels for the rail frame.
    pub fn fit(center: (f32, f32), size: (f32, f32), table: &TableSpec, margin: f32) -> Self {
        let sx = (size.0 - 2.0 * margin) / table.length as f32;
        let sy = (size.1 - 2.0 * margin) / table.width as f32;
        View { cx: center.0, cy: center.1, scale: sx.min(sy).max(1.0) }
    }

    pub fn to_screen(&self, wx: f64, wy: f64) -> (f32, f32) {
        (self.cx + wx as f32 * self.scale, self.cy - wy as f32 * self.scale)
    }

    pub fn to_world(&self, sx: f32, sy: f32) -> (f64, f64) {
        (((sx - self.cx) / self.scale) as f64, ((self.cy - sy) / self.scale) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_world_roundtrip() {
        let table = TableSpec::carom_match();
        let view = View::fit((500.0, 300.0), (1000.0, 600.0), &table, 40.0);
        for &(x, y) in &[(0.0, 0.0), (1.2, -0.5), (-1.0, 0.6)] {
            let (sx, sy) = view.to_screen(x, y);
            let (wx, wy) = view.to_world(sx, sy);
            assert!((wx - x).abs() < 1e-4 && (wy - y).abs() < 1e-4, "roundtrip failed for ({x},{y})");
        }
    }

    #[test]
    fn y_axis_flips() {
        let table = TableSpec::carom_match();
        let view = View::fit((500.0, 300.0), (1000.0, 600.0), &table, 40.0);
        let (_, top) = view.to_screen(0.0, 0.5); // higher in table space
        let (_, bottom) = view.to_screen(0.0, -0.5);
        assert!(top < bottom, "positive table-y should map to a smaller screen-y");
    }
}
