#!/usr/bin/env python3
"""Align tracked .shot files so t=0 is the STROKE, not the clip start.

The extractor pads a video lead-in before each stroke (PAD_PRE) so you can see the
player address the ball. The tracker now detects the cue ball sitting *at rest*
through that lead-in — but the physics reconstruction launches the ball at t=0, so
during the pre-stroke stillness the simulation races ahead of the real (stationary)
ball and the whole reconstruction is time-shifted. Fix it by moving t=0 to the
stroke onset: drop the pre-stroke rows, re-zero the timestamps, and bump the
`start` frame by the same amount so the editor's video sync is unchanged
(frame = start + t*fps stays constant for every real moment).

    python3 align_shots.py ../data/masa4_1080_match/*.shot
"""

import glob
import math
import sys

ONSET_MOVE = 0.025  # cue displacement (m) from its rest that marks the stroke
LEAD = 1            # keep this many frames of pre-stroke as the t=0 rest state


def stroke_onset(cue_rows):
    """Index into cue_rows of the last at-rest frame before the stroke."""
    if not cue_rows:
        return 0
    x0, y0 = cue_rows[0][1], cue_rows[0][2]
    for i, (_t, x, y) in enumerate(cue_rows):
        if math.hypot(x - x0, y - y0) > ONSET_MOVE:
            return max(0, i - LEAD)
    return 0


def align(path):
    header, rows = [], []
    cue = "white"
    fps, start = 30.0, 0
    with open(path) as f:
        for ln in f:
            s = ln.rstrip("\n")
            p = s.split(",")
            if len(p) == 4 and p[0] in ("white", "yellow", "red"):
                rows.append((p[0], float(p[1]), float(p[2]), float(p[3])))
            else:
                header.append(s)
                if s.startswith("cue "):
                    cue = s[4:].strip()
                elif s.startswith("fps "):
                    fps = float(s.split()[1])
                elif s.startswith("start "):
                    start = int(s.split()[1])

    cue_rows = [(t, x, y) for c, t, x, y in rows if c == cue]
    i_s = stroke_onset(cue_rows)
    if i_s == 0:
        return 0.0  # already stroke-aligned (or launches immediately)
    t_s = cue_rows[i_s][0]
    shift_frames = round(t_s * fps)

    out = []
    for h in header:
        if h.startswith("start "):
            out.append(f"start {start + shift_frames}")
        else:
            out.append(h)
    for c, t, x, y in rows:
        if t + 1e-9 < t_s:
            continue  # pre-stroke — drop
        out.append(f"{c},{t - t_s:.4f},{x:.4f},{y:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return t_s


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(a)
    for p in sorted(paths):
        t_s = align(p)
        print(f"{p.rsplit('/', 1)[-1]:<16} " +
              (f"trimmed {t_s:.2f}s pre-stroke -> t=0 at stroke" if t_s else "already aligned"))


if __name__ == "__main__":
    main()
