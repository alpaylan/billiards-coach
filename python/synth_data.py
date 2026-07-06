#!/usr/bin/env python3
"""Domain-randomized synthetic training data for the ball detector.

Our renderer's advantage: unlimited *labeled* frames for free. Each sample is a
top-down-ish table with the three balls at known positions, plus heavy
randomization — cloth/ball colors, perspective, lighting, motion blur, occlusion,
and (crucially) red/white distractor clutter at the rails — so the network learns
*ball appearance*, not color alone (which is exactly what fooled classical CV: a
fixed red rail object read as the red ball).

Samples are generated on the fly (no dataset on disk). Labels are per-ball image
centers + visibility. Run directly to write a montage of samples for review.
"""

import cv2
import numpy as np

BALL_CLASSES = ["white", "yellow", "red"]


def _cloth_bgr(rng):
    # Saturated blue (most common in three-cushion) or green tournament cloth.
    hue = rng.choice([rng.uniform(106, 118), rng.uniform(55, 68)])
    sat = rng.uniform(180, 255)
    val = rng.uniform(140, 215)
    return cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _ball_bgr(rng, cls):
    j = lambda c, s: int(np.clip(c + rng.uniform(-s, s), 0, 255))
    if cls == "white":
        b = rng.uniform(200, 245)
        return (j(b, 12), j(b + 8, 12), j(b + 12, 12))
    if cls == "yellow":
        return (j(60, 40), j(200, 30), j(235, 20))
    return (j(50, 30), j(50, 30), j(220, 30))  # red


def _persp_quad(rng, w, h):
    """Four table corners with random keystone/margins (clockwise from TL)."""
    mx, my = rng.uniform(0.04, 0.13) * w, rng.uniform(0.04, 0.13) * h
    k = rng.uniform(-0.10, 0.10) * w  # keystone
    j = lambda s: rng.uniform(-0.02, 0.02) * s
    return np.float32([
        [mx + max(0, k) + j(w), my + j(h)],
        [w - mx - max(0, -k) + j(w), my + j(h)],
        [w - mx + min(0, k) + j(w), h - my + j(h)],
        [mx + min(0, -k) + j(w), h - my + j(h)],
    ])


def render_sample(rng, w=288, h=192, base_r_range=(4.0, 9.0)):
    """Return (bgr uint8 image, labels) where labels = [(cls, x, y, visible, r)].
    `base_r_range` controls ball size — shrink it to mimic the small balls of the
    low-res overhead inset."""
    img = np.full((h, w, 3), 0, np.uint8)
    img[:] = rng.integers(60, 130)  # background gray
    # Random background clutter (crowd/venue).
    for _ in range(rng.integers(0, 8)):
        c = tuple(int(v) for v in rng.integers(30, 200, 3))
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, p, int(rng.integers(4, 20)), c, -1)

    quad = _persp_quad(rng, w, h)
    # Light tan wood rail underneath, then the cloth inset on top.
    wood = tuple(int(np.clip(c + rng.uniform(-20, 20), 0, 255)) for c in (120, 160, 195))
    cv2.fillConvexPoly(img, quad.astype(np.int32), wood)
    inner = (quad - quad.mean(axis=0)) * rng.uniform(0.82, 0.9) + quad.mean(axis=0)
    cv2.fillConvexPoly(img, inner.astype(np.int32), tuple(int(v) for v in _cloth_bgr(rng)))
    # Diamond sights on the rail band (and a subtle inner cushion line).
    for t in np.linspace(0.1, 0.9, 7):
        for a, b in [(0, 1), (3, 2), (0, 3), (1, 2)]:
            p = quad[a] * (1 - t) + quad[b] * t
            m = inner[a] * (1 - t) + inner[b] * t
            q = p * 0.5 + m * 0.5
            cv2.circle(img, (int(q[0]), int(q[1])), 1, (60, 60, 60), -1)
    cv2.polylines(img, [inner.astype(np.int32)], True, (110, 90, 40), 1)

    # Homography from unit table [0,1]^2 to the cloth (inner) quad.
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), inner.astype(np.float32))

    def to_img(u, v):
        p = H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    base_r = rng.uniform(*base_r_range)
    labels = []
    for cls in BALL_CLASSES:
        u, v = rng.uniform(0.06, 0.94), rng.uniform(0.06, 0.94)
        x, y = to_img(u, v)
        r = base_r * (0.8 + 0.5 * v)  # nearer (larger v) => bigger
        color = _ball_bgr(rng, cls)
        # Shaded sphere: darker rim, lit core, small specular highlight.
        cv2.circle(img, (int(x), int(y)), int(round(r)), tuple(int(c * 0.65) for c in color), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), max(1, int(round(r * 0.82))), color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x - r * 0.3), int(y - r * 0.3)), max(1, int(r * 0.28)), (248, 248, 248), -1, cv2.LINE_AA)
        # Occasional motion-blur streak.
        if rng.random() < 0.25:
            ang = rng.uniform(0, 2 * np.pi)
            L = rng.uniform(r, 5 * r)
            x2, y2 = x + L * np.cos(ang), y + L * np.sin(ang)
            cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), color, int(round(2 * r)), cv2.LINE_AA)
        labels.append([cls, float(x), float(y), 1, float(r)])  # cls, cx, cy, visible, radius

    # Distractors that fooled classical CV: red/white/yellow rail objects & cue stick.
    for _ in range(rng.integers(0, 4)):
        dc = _ball_bgr(rng, rng.choice(BALL_CLASSES))
        # near an edge of the table quad
        e = quad[rng.integers(0, 4)]
        p = (int(e[0] + rng.uniform(-14, 14)), int(e[1] + rng.uniform(-14, 14)))
        if rng.random() < 0.5:
            cv2.rectangle(img, (p[0] - 3, p[1] - 2), (p[0] + 3, p[1] + 2), dc, -1)
        else:
            cv2.circle(img, p, int(rng.integers(2, 5)), dc, -1)
    if rng.random() < 0.5:  # cue stick
        a = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        b = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, a, b, (150, 190, 220), int(rng.integers(2, 5)), cv2.LINE_AA)

    # Occlusion (hand/arm): a dark blob that may hide a ball.
    if rng.random() < 0.4:
        oc = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        orad = int(rng.integers(15, 45))
        cv2.circle(img, oc, orad, tuple(int(v) for v in rng.integers(20, 90, 3)), -1)
        for lab in labels:
            if np.hypot(lab[1] - oc[0], lab[2] - oc[1]) < orad * 0.8:
                lab[3] = 0  # occluded

    # Lighting: brightness gradient + vignette.
    gx = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0), w)
    gy = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), h)
    gain = np.outer(gy, gx)[:, :, None]
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img, labels


def main():
    rng = np.random.default_rng(0)
    tiles = [render_sample(rng)[0] for _ in range(12)]
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
    cv2.imwrite("synth_samples.png", grid)
    print("wrote synth_samples.png (12 domain-randomized samples)")


if __name__ == "__main__":
    main()
