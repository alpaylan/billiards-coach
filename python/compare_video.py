#!/usr/bin/env python3
"""Side-by-side viewer: actual video vs the engine's reconstruction.

Left  — the real broadcast frames with the tracked balls marked (perception).
Right — a top-down table animating the *tracked* path ("what happened") against
        the *reconstructed* path (the fitted cue action re-simulated).

    python3 compare_video.py FRAMES_DIR TRACKED.csv RECON.csv \
        --corners "85,32 493,22 577,168 20,188" --start-frame 122 --fps 15 \
        --out compare.mp4
"""

import argparse
import os

import cv2
import numpy as np

TABLE_L, TABLE_W = 2.84, 1.42
COLORS = ["white", "yellow", "red"]
BGR = {"white": (240, 240, 240), "yellow": (0, 210, 240), "red": (40, 40, 220)}


def load_tracks(path):
    """Read a color-labeled shot/recon file: header lines (`cue`, `frames`, `fps`,
    `start`, `corners`, `orient`, and the `color,t,x,y` header) then `COLOR,t,x,y`
    data lines — colors are carried explicitly, so there is no index mapping."""
    tracks = {c: {} for c in COLORS}
    for line in open(path).read().splitlines():
        parts = line.strip().split(",")
        if len(parts) != 4 or parts[0] not in tracks:
            continue  # header or blank line
        c, t, x, y = parts
        tracks[c][round(float(t), 4)] = (float(x), float(y))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("tracked")
    ap.add_argument("recon")
    ap.add_argument("--corners", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--orient", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--out", default="compare.mp4")
    args = ap.parse_args()

    tracked = load_tracks(args.tracked)
    recon = load_tracks(args.recon)
    times = sorted(set().union(*(tracked[c].keys() for c in COLORS)))

    hl, hw = TABLE_L / 2, TABLE_W / 2
    img_corners = np.float32([[float(a) for a in p.split(",")] for p in args.corners.split()])
    if args.orient == "horizontal":
        tab_corners = np.float32([[-hl, hw], [hl, hw], [hl, -hw], [-hl, -hw]])
    else:  # portrait inset: top edge is a short rail
        tab_corners = np.float32([[-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]])
    tab2img = cv2.getPerspectiveTransform(tab_corners, img_corners)

    def to_img(x, y):
        p = tab2img @ np.array([x, y, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    # Top-down canvas geometry.
    scale, margin = 180, 22
    TW, TH = int(TABLE_L * scale), int(TABLE_W * scale)
    CW, CH = TW + 2 * margin, TH + 2 * margin

    def to_canvas(x, y):
        return int(CW / 2 + x * scale), int(CH / 2 - y * scale)

    frame0 = cv2.imread(os.path.join(args.frames_dir, f"f_{args.start_frame + 1:04d}.png"))
    fh, fw = frame0.shape[:2]
    out_h = max(fh, CH)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw + CW, out_h))
    montage = []

    for k, t in enumerate(times):
        # LEFT: actual frame with tracked ball markers.
        fpath = os.path.join(args.frames_dir, f"f_{args.start_frame + 1 + round(t * args.fps):04d}.png")
        left = cv2.imread(fpath)
        if left is None:
            left = frame0.copy()
        for c in COLORS:
            if t in tracked[c]:
                cv2.circle(left, to_img(*tracked[c][t]), 6, BGR[c], -1)
                cv2.circle(left, to_img(*tracked[c][t]), 7, (20, 20, 20), 1)
        cv2.putText(left, "ACTUAL (tracked)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # RIGHT: top-down, tracked (solid) vs reconstructed (dotted).
        right = np.full((CH, CW, 3), (60, 60, 60), np.uint8)
        cv2.rectangle(right, (margin, margin), (CW - margin, CH - margin), (150, 90, 30), -1)
        past = [tt for tt in times if tt <= t]
        for c in COLORS:
            solid = [to_canvas(*tracked[c][tt]) for tt in past if tt in tracked[c]]
            for a, b in zip(solid, solid[1:]):
                cv2.line(right, a, b, BGR[c], 2, cv2.LINE_AA)
            dots = [to_canvas(*recon[c][tt]) for tt in past if tt in recon[c]]
            for i, p in enumerate(dots):
                if i % 2 == 0:
                    cv2.circle(right, p, 1, BGR[c], -1)  # dotted reconstruction
            if solid:
                cv2.circle(right, solid[-1], 6, BGR[c], -1)
            if dots:
                cv2.circle(right, dots[-1], 7, BGR[c], 2)  # ring = reconstructed
        cv2.putText(right, "top-down: -- tracked   o reconstructed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Compose (pad to same height).
        comp = np.zeros((out_h, fw + CW, 3), np.uint8)
        comp[:fh, :fw] = left
        comp[:CH, fw:] = right
        writer.write(comp)
        if k in (0, len(times) // 3, 2 * len(times) // 3, len(times) - 1):
            montage.append(cv2.resize(comp, (fw + CW, out_h)))

    writer.release()
    if montage:
        cv2.imwrite(args.out.rsplit(".", 1)[0] + "_montage.png", np.vstack(montage))
    print(f"wrote {args.out} ({len(times)} frames) and _montage.png")


if __name__ == "__main__":
    main()
