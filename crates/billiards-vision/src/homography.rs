//! Planar homography — the geometric heart of "2D image → table plane".
//!
//! Because the table is a known-dimension plane, a single view relates image
//! pixels to table coordinates by a `3×3` homography (no full 3D reconstruction
//! needed). Four point correspondences determine it exactly; we solve the
//! resulting `8×8` linear system directly (no SVD) via Gauss–Jordan elimination.

/// A 2D projective transform mapping `(x, y)` in one plane to `(X, Y)` in
/// another: `[X Y 1]ᵀ ∝ M · [x y 1]ᵀ`.
#[derive(Clone, Copy, Debug)]
pub struct Homography {
    m: [[f64; 3]; 3],
}

impl Homography {
    /// Fit the homography mapping each `src[i]` to `dst[i]`, from exactly four
    /// correspondences. Returns `None` if the points are degenerate (collinear).
    pub fn from_correspondences(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<Self> {
        // Each correspondence contributes two rows; unknowns are the eight free
        // entries of M with m22 fixed to 1.
        let mut a = [[0.0f64; 8]; 8];
        let mut b = [0.0f64; 8];
        for i in 0..4 {
            let (x, y) = src[i];
            let (xp, yp) = dst[i];
            let r = 2 * i;
            a[r] = [x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y];
            b[r] = xp;
            a[r + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y];
            b[r + 1] = yp;
        }
        let h = solve8(a, b)?;
        Some(Homography {
            m: [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        })
    }

    /// Project a point through the homography.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.m;
        let w = m[2][0] * x + m[2][1] * y + m[2][2];
        ((m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w)
    }
}

/// Solve `A·x = b` for an 8×8 system by Gauss–Jordan elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut pivot = col;
        for r in (col + 1)..8 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            for c in col..8 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; 8];
    for i in 0..8 {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A perspective-ish image quad and the table nose corners it maps to.
    fn corners() -> ([(f64, f64); 4], [(f64, f64); 4]) {
        let image = [(250.0, 80.0), (550.0, 80.0), (720.0, 400.0), (80.0, 400.0)];
        let table = [(-1.42, 0.71), (1.42, 0.71), (1.42, -0.71), (-1.42, -0.71)];
        (image, table)
    }

    #[test]
    fn maps_corners_exactly() {
        let (image, table) = corners();
        let h = Homography::from_correspondences(image, table).unwrap();
        for i in 0..4 {
            let (x, y) = h.apply(image[i].0, image[i].1);
            assert!((x - table[i].0).abs() < 1e-9 && (y - table[i].1).abs() < 1e-9);
        }
    }

    #[test]
    fn inverse_round_trips() {
        let (image, table) = corners();
        let fwd = Homography::from_correspondences(image, table).unwrap();
        let inv = Homography::from_correspondences(table, image).unwrap();
        // An arbitrary interior image point survives image→table→image.
        let (px, py) = (400.0, 250.0);
        let (tx, ty) = fwd.apply(px, py);
        let (rx, ry) = inv.apply(tx, ty);
        assert!((rx - px).abs() < 1e-6 && (ry - py).abs() < 1e-6, "round-trip ({rx},{ry})");
    }
}
