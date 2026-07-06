#!/usr/bin/env python3
"""Multi-frame ball tracking + shot segmentation for three-cushion.

Three-cushion's three *distinctly colored* balls (white / yellow / red) make
tracking easy: detect each color per frame and the per-color centroid sequence
IS that ball's track — no data-association ambiguity (unlike same-colored pool
balls). A "turn"/shot is then the interval where a ball is moving.

Pipeline: frames -> per-frame color detection -> lift to table coords via the
calibration homography -> per-ball trajectories -> motion-based shot segments.

Validated against a synthetic clip with known ground-truth paths:

    python3 track.py --synthetic

Run on real frames (e.g. the overhead inset), giving the table's 4 corners in
the image (clockwise from top-left) and the table's long-axis orientation:

    python3 track.py --frames real_frames --corners "18,15 187,13 188,310 17,312" \
        --orient vertical --out traj.png
"""

import argparse
import glob
import os

import cv2
import numpy as np

TABLE_L, TABLE_W = 2.84, 1.42
FPS = 30.0

COLORS = ["white", "yellow", "red"]
HSV_RANGES = {
    "red":    [((0, 120, 90), (10, 255, 255)), ((170, 120, 90), (179, 255, 255))],
    "yellow": [((18, 70, 130), (35, 255, 255))],
    "white":  [((0, 0, 180), (179, 60, 255))],
}
DRAW_BGR = {"white": (240, 240, 240), "yellow": (0, 210, 240), "red": (40, 40, 220)}
MIN_AREA, MAX_AREA = 12, 3000


# --- calibration -----------------------------------------------------------

def table_targets(orient):
    """The 4 table-corner coords (meters) matching image corners given clockwise
    from top-left, respecting which image edge is the table's long (2.84 m) axis."""
    hl, hw = TABLE_L / 2, TABLE_W / 2
    if orient == "horizontal":  # long axis runs left-right in the image
        return np.float32([[-hl, hw], [hl, hw], [hl, -hw], [-hl, -hw]])
    else:  # vertical: top edge is a short rail
        return np.float32([[-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]])


def calibrate(image_corners, orient):
    H = cv2.getPerspectiveTransform(np.float32(image_corners), table_targets(orient))
    return H




def to_table(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# --- detection -------------------------------------------------------------

def detect_frame(bgr, table_mask):
    """{color: (u, v) or None} for the largest ball-sized blob of each color."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {}
    for color in COLORS:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.bitwise_and(mask, table_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        best, best_area = None, MIN_AREA
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            elongation = max(w_, h_) / max(1, min(w_, h_))
            # A ball is compact and round; reject elongated blobs (the cue stick)
            # and sparse ones.
            if a <= MIN_AREA or a > MAX_AREA or elongation > 2.2 or a < 0.35 * w_ * h_:
                continue
            if a > best_area:
                best, best_area = cent[i], a
        out[color] = (float(best[0]), float(best[1])) if best is not None else None
    return out


# --- tracking + segmentation ----------------------------------------------

def fill_gaps(track, max_gap=5):
    """Linear-interpolate only SHORT missing runs (up to `max_gap` frames) — motion
    blur or a brief occlusion, where a straight line between the neighbours is safe.
    Longer dropouts are left as None: the ball went somewhere we didn't see (e.g.
    into a corner and back), and inventing a straight path there would be a lie.
    Leading/trailing gaps are also left None rather than frozen at the last fix.
    Downstream code treats None as 'unknown' and simply doesn't use those frames."""
    n = len(track)
    known = [i for i, p in enumerate(track) if p is not None]
    if not known:
        return track
    out = list(track)
    for i in range(n):
        if out[i] is not None:
            continue
        lo = max((k for k in known if k < i), default=None)
        hi = min((k for k in known if k > i), default=None)
        if lo is None or hi is None:
            continue  # leading/trailing gap — don't fabricate or freeze
        if hi - lo - 1 > max_gap:
            continue  # too long to trust a straight line through it
        t = (i - lo) / (hi - lo)
        (x0, y0), (x1, y1) = track[lo], track[hi]
        out[i] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out


def _classical_ball(bgr, table_mask, color):
    """The most BALL-LIKE blob of `color`: compact, round, and nearest the
    expected ball area for this frame scale — not merely the largest."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.bitwise_and(mask, table_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    # ball diameter ≈ 3.7% of the inset frame width (61.5mm on a 1.42m short rail)
    exp_area = 3.1416 * (0.0185 * bgr.shape[1]) ** 2
    best, best_ratio = None, 4.0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a < 0.25 * exp_area or a > 4.0 * exp_area:
            continue
        if max(w_, h_) / max(1, min(w_, h_)) > 1.8 or a < 0.5 * w_ * h_:
            continue  # elongated or sparse — an arm or cue, not a ball
        ratio = max(a, exp_area) / max(1.0, min(a, exp_area))
        if ratio < best_ratio:
            best, best_ratio = cent[i], ratio
    return (float(best[0]), float(best[1])) if best is not None else None


def make_learned_detector(weights="finetuned_detector.pt", thr=0.22):
    """Detector callable backed by the fine-tuned CNN (finetune_detector.py).
    Same interface as detect_frame: (bgr, table_mask) -> {color: (u,v) or None}.
    Detections outside the table mask (e.g. rail false-positives) are rejected.
    Runs on the Apple-Silicon GPU (MPS) when available — a full broadcast day is
    hundreds of shots, and the CPU path is ~an order of magnitude slower."""
    import torch
    from finetune_detector import ID_CLS, build_model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model()
    model.load_state_dict(torch.load(weights))
    model.eval()
    model.to(device)

    def detect(bgr, table_mask):
        t = torch.tensor(np.transpose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255
        with torch.no_grad():
            out = model([t.to(device)])[0]
            out = {k: v.cpu() for k, v in out.items()}
        # Collect BLOBS (positions) with per-colour scores, then assign colours
        # to blobs JOINTLY. Independent per-colour maxima lose a ball whenever
        # the CNN misjudges its colour (e.g. a rail-shadowed red scoring
        # white 0.43 / red 0.08): exclusivity recovers it, because the other
        # two blobs are strongly claimed by their true colours.
        blobs = []  # [(cx, cy), {color: score}]
        frame_w = bgr.shape[1]
        for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
            c = ID_CLS.get(int(l))
            if not c or s < 0.04:
                continue
            # a ball is small; a player's arm/shirt is not (boxes are in px)
            if max(float(b[2] - b[0]), float(b[3] - b[1])) > 0.09 * frame_w:
                continue
            cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
            iy, ix = int(cy), int(cx)
            if not (0 <= iy < table_mask.shape[0] and 0 <= ix < table_mask.shape[1] and table_mask[iy, ix]):
                continue  # off the table (rail/background)
            for pos, scores in blobs:
                if np.hypot(pos[0] - cx, pos[1] - cy) < 9.0:
                    scores[c] = max(scores.get(c, 0.0), float(s))
                    break
            else:
                blobs.append(((cx, cy), {c: float(s)}))
        blobs = blobs[:8]  # plenty; keeps the assignment tiny

        best_assign, best_total = None, -1.0
        from itertools import permutations
        n = len(blobs)
        idxs = list(range(n)) + [None] * 3  # None = colour unassigned
        for pick in permutations(idxs, 3):
            # distinct blob indices only (Nones may repeat)
            used = [p for p in pick if p is not None]
            if len(used) != len(set(used)):
                continue
            total, ok = 0.0, True
            for c, p in zip(COLORS, pick):
                if p is None:
                    continue
                sc = blobs[p][1].get(c, 0.0)
                if sc <= 0.0:
                    ok = False
                    break
                total += sc
            if ok and total > best_total:
                best_total, best_assign = total, pick
        result = {c: None for c in COLORS}
        if best_assign is not None:
            for c, p in zip(COLORS, best_assign):
                # exclusivity may hand a colour a weakly-scored blob, but never a
                # junk one: 0.12 floor (an arm scoring red 0.08 stays unclaimed)
                if p is not None and blobs[p][1].get(c, 0.0) >= 0.12:
                    result[c] = blobs[p][0]
        # Hybrid fallback: colours the CNN missed entirely (corner-tucked balls
        # are outside its training set) go to a BALL-SHAPED classical pick — the
        # blob nearest the expected ball size, compact and round. (Choosing the
        # LARGEST colour blob marked a forearm as red: skin passes the hue range
        # and out-areas the ball.)
        if any(v is None for v in result.values()):
            taken = [v for v in result.values() if v is not None]
            for c in COLORS:
                if result[c] is None:
                    p = _classical_ball(bgr, table_mask, c)
                    if p and all(np.hypot(p[0] - q[0], p[1] - q[1]) > 9.0 for q in taken):
                        result[c] = p
                        taken.append(p)
        return result

    return detect


def track_clip(frames, shape, image_corners, orient, detect_fn=detect_frame, mask_pad=-5):
    """`frames` is any iterable of BGR arrays (streamed, so frame count is
    unbounded); `shape` is (H, W) of a frame. `detect_fn(bgr, mask)` returns
    per-color image centroids (classical color or the learned detector).
    `mask_pad`: negative erodes the table mask (classical, to avoid the tan
    cushion), positive dilates it (learned detector, to keep edge balls)."""
    H = calibrate(image_corners, orient)
    table_mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(table_mask, np.int32(image_corners), 255)
    k = abs(mask_pad)
    if k:
        kern = np.ones((k, k), np.uint8)
        table_mask = cv2.erode(table_mask, kern) if mask_pad < 0 else cv2.dilate(table_mask, kern)

    # A ball can't teleport, but it *does* keep moving while it's undetected. The
    # jump allowance therefore grows with the gap since the last good fix: a ball
    # re-acquired after N missed frames may legitimately be N steps away, so gating
    # on a single-frame distance (anchored on a now-stale position) wrongly rejects
    # every re-acquisition and freezes the ball. Capped so a true false blob across
    # the table is still rejected.
    max_jump = 0.30  # meters/frame — the fastest a real carom ball moves (~9 m/s
                     # at 30 fps); a re-acquisition must imply no more than this
                     # *average* speed over the gap, which rejects false blobs.
    gap_cap = 5      # cap the reach at ~1.5 m so a teleport across the table fails
    raw = {c: [] for c in COLORS}
    prev = {c: None for c in COLORS}     # last accepted table position
    gap = {c: 0 for c in COLORS}         # frames since this ball's last accepted fix

    def d(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    for f in frames:
        det = detect_fn(f, table_mask)
        cand = {c: (to_table(H, *det[c]) if det[c] else None) for c in COLORS}

        # White and yellow are the two visually confusable balls, and a low-
        # confidence frame can flip the detector's labels wholesale — each
        # candidate lands squarely on the OTHER ball, then flips back a few
        # frames later (the "zigzag"). Colour can't arbitrate its own mislabel;
        # continuity can: if exchanging the two labels turns two large jumps
        # into two small ones, the labels were swapped. Checked before gating
        # so the teleport gates judge the corrected identities.
        a, b = cand["white"], cand["yellow"]
        pa, pb = prev["white"], prev["yellow"]
        if a and b and pa and pb:
            keep = d(a, pa) + d(b, pb)
            swap = d(b, pa) + d(a, pb)
            if keep > 0.16 and swap < 0.4 * keep:
                cand["white"], cand["yellow"] = b, a

        for c in COLORS:
            p = cand[c]
            if p and prev[c] and d(p, prev[c]) > max_jump * min(gap[c] + 1, gap_cap):
                cand[c] = None  # implausible jump even accounting for the gap

        # Identity resolution: two colours can't be the same blob. If two candidates
        # coincide (closer than a ball diameter — real balls only touch at ~6 cm),
        # it's a colour mislabel; keep the one consistent with its own motion (the
        # smaller jump from where that ball was) and drop the impostor. This stops a
        # ball's tracker from hopping onto a stationary neighbour.
        cols = [c for c in COLORS if cand[c]]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if cand[a] and cand[b] and d(cand[a], cand[b]) < 0.055:
                    ja = d(cand[a], prev[a]) if prev[a] else 1e9
                    jb = d(cand[b], prev[b]) if prev[b] else 1e9
                    cand[a if ja > jb else b] = None

        for c in COLORS:
            p = cand[c]
            raw[c].append(p)
            if p:
                prev[c] = p
                gap[c] = 0
            else:
                gap[c] += 1
    return {c: fill_gaps(raw[c]) for c in COLORS}


def segment_shots(tracks, fps=FPS, v_move=0.12, min_len=0.25, bridge=0.3):
    """Intervals (start_frame, end_frame) where any ball is moving."""
    n = len(next(iter(tracks.values())))
    speed = np.zeros(n)
    for c in COLORS:
        p = tracks[c]
        for i in range(1, n):
            if p[i] and p[i - 1]:
                d = np.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]) * fps
                speed[i] = max(speed[i], d)

    moving = speed > v_move
    shots, i = [], 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < n and (moving[j + 1] or gap < bridge * fps):
            j += 1
            gap = 0 if moving[j] else gap + 1
        # trim trailing bridged stillness
        while j > i and not moving[j]:
            j -= 1
        if (j - i) / fps >= min_len:
            shots.append((i, j))
        i = j + 1
    return shots


# --- synthetic clip (ground truth) ----------------------------------------

def _interp(keys, f):
    if f <= keys[0][0]:
        return keys[0][1], keys[0][2]
    if f >= keys[-1][0]:
        return keys[-1][1], keys[-1][2]
    for (f0, x0, y0), (f1, x1, y1) in zip(keys, keys[1:]):
        if f0 <= f <= f1:
            t = (f - f0) / (f1 - f0)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return keys[-1][1], keys[-1][2]


def make_synthetic(n=90):
    """Landscape blue table, 3 balls on scripted paths. Returns frames, ground
    truth (table coords per frame), image corners, orientation."""
    scale, margin = 200, 30
    tw, th = int(TABLE_L * scale), int(TABLE_W * scale)
    W, H = tw + 2 * margin, th + 2 * margin
    cx, cy = W / 2, H / 2
    corners = [(margin, margin), (W - margin, margin), (W - margin, H - margin), (margin, H - margin)]

    # Scripted paths (frame, x, y) in table meters. Cue strikes yellow ~f30.
    paths = {
        "white":  [(0, -1.1, -0.3), (14, -1.1, -0.3), (30, 0.28, 0.22), (52, -0.5, 0.55), (72, -0.5, 0.55)],
        "yellow": [(0, 0.35, 0.25), (30, 0.35, 0.25), (46, 0.78, 0.05), (89, 0.78, 0.05)],
        "red":    [(0, 0.85, -0.15), (89, 0.85, -0.15)],
    }

    def t2i(x, y):
        return int(cx + x * scale), int(cy - y * scale)

    frames, gt = [], {c: [] for c in COLORS}
    for f in range(n):
        img = np.full((H, W, 3), (70, 70, 70), np.uint8)
        cv2.rectangle(img, (margin, margin), (W - margin, H - margin), (200, 110, 30), -1)
        for c in COLORS:
            x, y = _interp(paths[c], f)
            gt[c].append((x, y))
            cv2.circle(img, t2i(x, y), 6, DRAW_BGR[c], -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (3, 3), 0)  # mild sensor blur
        noise = np.random.normal(0, 4, img.shape)  # signed, clipped (no uint8 wraparound)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(img)
    return frames, gt, corners, "horizontal"


# --- visualization ---------------------------------------------------------

def draw_trajectories(tracks, shots, out_path):
    scale, margin = 190, 24
    tw, th = int(TABLE_L * scale), int(TABLE_W * scale)
    W, H = tw + 2 * margin, th + 2 * margin
    cx, cy = W / 2, H / 2
    canvas = np.full((H, W, 3), (60, 60, 60), np.uint8)
    cv2.rectangle(canvas, (margin, margin), (W - margin, H - margin), (150, 90, 30), -1)

    def t2i(x, y):
        return int(cx + x * scale), int(cy - y * scale)

    # Only draw the moving span of each ball, per detected shot.
    span = range(shots[0][0], shots[-1][1] + 1) if shots else range(len(next(iter(tracks.values()))))
    for c in COLORS:
        pts = [t2i(*tracks[c][i]) for i in span if tracks[c][i] is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, DRAW_BGR[c], 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[0], 6, DRAW_BGR[c], -1)
            cv2.rectangle(canvas, (pts[-1][0] - 5, pts[-1][1] - 5), (pts[-1][0] + 5, pts[-1][1] + 5), DRAW_BGR[c], -1)
    cv2.imwrite(out_path, canvas)


# --- entry points ----------------------------------------------------------

def run_synthetic():
    frames, gt, corners, orient = make_synthetic()
    tracks = track_clip(frames, frames[0].shape[:2], corners, orient)
    shots = segment_shots(tracks)

    # Position accuracy over frames where the ball is detectable.
    errs = []
    for c in COLORS:
        for i in range(len(frames)):
            if tracks[c][i] is not None:
                gx, gy = gt[c][i]
                errs.append(np.hypot(tracks[c][i][0] - gx, tracks[c][i][1] - gy))
    mean_err, max_err = float(np.mean(errs)), float(np.max(errs))

    print("synthetic validation")
    print(f"  frames: {len(frames)}  balls: {', '.join(COLORS)}")
    print(f"  tracking error: mean {mean_err*1000:.1f} mm, max {max_err*1000:.1f} mm")
    print(f"  detected shots (frames): {shots}  (scripted motion ~14-52)")
    draw_trajectories(tracks, shots, "synthetic_traj.png")
    print("  wrote synthetic_traj.png")

    assert mean_err < 0.02, f"tracking error too high: {mean_err}"
    assert len(shots) == 1 and shots[0][0] <= 18 and shots[0][1] >= 48, f"bad segmentation: {shots}"
    print("  OK")


def _rest_bounds(tracks, a, b, fps, v_rest=0.04, hold=0.3):
    """Extend a motion interval [a,b] outward to the surrounding ball-rest frames:
    back to the last still frame before the stroke (so t=0 is the true starting
    layout), and forward to the moment the balls actually settle — using a *median*
    of ball speed over the next `hold` seconds, so a single-frame detection spike
    (a hand or the next player entering the inset) doesn't look like motion and
    isn't mistaken for the ball still rolling."""
    n = len(next(iter(tracks.values())))
    spd = np.zeros(n)
    for c in COLORS:
        p = tracks[c]
        for i in range(1, n):
            if p[i] and p[i - 1]:
                spd[i] = max(spd[i], np.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]) * fps)

    w = max(1, int(hold * fps))
    # onset: step back frame-by-frame from the first moving frame to the last still
    # one (per-frame, so we don't average the pre-stroke rest back in and stop early).
    o = a
    while o > 0 and spd[o] > v_rest:
        o -= 1
    # A clean start means the balls were *sustainedly* at rest before the stroke —
    # the median speed over the ~third-second up to the onset is still. A single
    # coincidental slow frame (a ball changing direction mid-maneuver, as when the
    # recording begins mid-shot) doesn't qualify, so those shots are skipped.
    clean = o > 0 and np.median(spd[max(0, o - w):o + 1]) <= v_rest
    # settle: forward to where the balls stay still — median over the next `hold`
    # seconds so a single-frame detection spike doesn't read as the ball still rolling.
    s = b
    while s < n - w and np.median(spd[s:s + w]) >= v_rest:
        s += 1
    return o, min(s + int(0.15 * fps), n - 1), clean  # small tail so the stop is visible


def export_for_fit(tracks, shots, fps, out_path, shot_idx,
                   frames_dir=None, corners_str=None, orient="horizontal", motion=None):
    """Export one shot's per-ball table-coord tracks for the Rust fitter.

    `tracks` holds the *honest* positions (gaps left as None where the ball wasn't
    seen). `motion` is a gap-filled copy used only to locate the shot window / rest
    bounds / cue — because segmentation reads ball speed, and an unfilled gap would
    otherwise look like the ball is at rest. The exported positions come from
    `tracks`, so the written shot still has honest gaps.

    Format: a `cue COLOR` header then `COLOR,t,x,y` data lines — colors are carried
    explicitly so the editor draws each ball correctly. In three-cushion the cue is
    always a player ball (white or yellow), never the shared red object ball; we pick
    whichever of white/yellow travelled most in the shot (the struck ball).

    When the source frames are known we also write a back-link (`frames`/`fps`/
    `start`/`corners`/`orient`) so the editor can play the actual clip beside the
    reconstruction, synced to the same timeline. `corners_str` is in *original*
    frame pixels (the editor loads the raw frames, not the upscaled ones)."""
    if not shots:
        print("no shots to export")
        return False
    if motion is None:
        motion = tracks

    def travel(a, b, c):
        return sum(np.hypot(motion[c][i+1][0]-motion[c][i][0], motion[c][i+1][1]-motion[c][i][1])
                   for i in range(a, b) if motion[c][i] and motion[c][i+1])

    idx = shot_idx if shot_idx is not None else max(
        range(len(shots)), key=lambda k: max(travel(*shots[k], c) for c in COLORS))
    a, b, clean = _rest_bounds(motion, *shots[idx], fps)

    # A valid reconstruction needs the balls at REST at t=0 (the starting layout).
    # If there was no still frame before the stroke — e.g. the recording began
    # mid-shot — skip this shot rather than fit garbage from a moving start.
    if not clean:
        print(f"skip shot {idx}: no at-rest start (recording began mid-shot)")
        return False

    # Cue = the player ball (white/yellow) that was STRUCK — i.e. the one that
    # starts moving first. Object balls only move once the cue reaches them, so
    # first-motion identifies the cue unambiguously (more robust than "moved most",
    # which mis-picks when an object ball ends up travelling further).
    def first_move(c):
        for i in range(a, b):
            if motion[c][i] and motion[c][i + 1]:
                d = np.hypot(motion[c][i + 1][0] - motion[c][i][0],
                             motion[c][i + 1][1] - motion[c][i][1]) * fps
                if d > 0.15:
                    return i
        return b + 1  # never moved
    cue = min(("white", "yellow"), key=first_move)
    order = [cue] + [c for c in COLORS if c != cue]  # cue first, keep true colors
    with open(out_path, "w") as f:
        f.write(f"cue {cue}\n")
        if frames_dir:
            # Back-link to the source clip so the editor shows the real video.
            f.write(f"frames {os.path.abspath(frames_dir)}\n")
            f.write(f"fps {fps:g}\n")
            f.write(f"start {a}\n")
            if corners_str:
                f.write(f"corners {corners_str}\n")
            f.write(f"orient {orient}\n")
        f.write("color,t,x,y\n")
        for c in order:
            for i in range(a, b + 1):
                if tracks[c][i]:
                    f.write(f"{c},{(i-a)/fps:.4f},{tracks[c][i][0]:.4f},{tracks[c][i][1]:.4f}\n")
    print(f"exported shot {idx} ({a/fps:.2f}-{b/fps:.2f}s), cue={cue} -> {out_path}")
    return True


def run_real(frames_dir, corners_str, orient, out, scale, fit_out=None, shot_idx=None, detector="color"):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise SystemExit(f"no frames in {frames_dir}")
    corners = [(float(a) * scale, float(b) * scale)
               for a, b in (p.split(",") for p in corners_str.split())]

    def load(p):
        img = cv2.imread(p)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else img

    detect_fn = make_learned_detector() if detector == "learned" else detect_frame
    mask_pad = 12 if detector == "learned" else -5
    print(f"detector: {detector}")
    shape = load(paths[0]).shape[:2]
    frames = (load(p) for p in paths)  # streamed
    tracks = track_clip(frames, shape, corners, orient, detect_fn, mask_pad)
    shots = segment_shots(tracks)

    print(f"tracked {len(paths)} frames from {frames_dir}")
    for c in COLORS:
        seen = sum(p is not None for p in tracks[c])
        print(f"  {c:6}: detected in {seen}/{len(paths)} frames")
    print(f"  shots (start->end s): {[(round(a/FPS,2), round(b/FPS,2)) for a,b in shots]}")
    for a, b in shots:
        length = sum(
            np.hypot(tracks[c][i+1][0]-tracks[c][i][0], tracks[c][i+1][1]-tracks[c][i][1])
            for c in COLORS for i in range(a, b) if tracks[c][i] and tracks[c][i+1]
        )
        print(f"    shot {a/FPS:.2f}-{b/FPS:.2f}s: total ball travel {length:.2f} m")
    draw_trajectories(tracks, shots, out)
    print(f"  wrote {out}")
    if fit_out:
        export_for_fit(tracks, shots, FPS, fit_out, shot_idx,
                       frames_dir=frames_dir, corners_str=corners_str, orient=orient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--frames")
    ap.add_argument("--corners")
    ap.add_argument("--orient", choices=["horizontal", "vertical"], default="vertical")
    ap.add_argument("--out", default="traj.png")
    ap.add_argument("--scale", type=float, default=1.0, help="upscale factor for small insets")
    ap.add_argument("--fit-out", help="export a shot's tracks as CSV for the Rust fitter")
    ap.add_argument("--shot", type=int, help="shot index to export (default: most ball travel)")
    ap.add_argument("--detector", choices=["color", "learned"], default="color")
    args = ap.parse_args()
    if args.synthetic:
        run_synthetic()
    elif args.frames and args.corners:
        run_real(args.frames, args.corners, args.orient, args.out, args.scale, args.fit_out, args.shot, args.detector)
    else:
        ap.error("pass --synthetic or (--frames and --corners)")


if __name__ == "__main__":
    main()
