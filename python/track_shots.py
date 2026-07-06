#!/usr/bin/env python3
"""Track each extracted shot (from detect_shots.py --extract) into a .shot file.

Loads the learned inset detector ONCE and runs it over every shot_NN/ directory,
segmenting the stroke and exporting a color-labeled shot that links back to the
real inset frames (so the editor plays the literal clip). Prints a per-shot
summary (cue ball, samples, whether a stroke segmented).

    python3 track_shots.py ../data/masa4_shots --out-dir ../data/shots \
        --corners "7,12 172,10 174,327 9,327" --orient vertical --scale 4
"""

import argparse
import glob
import os

import cv2
import numpy as np

import track as tk


def track_one(shotdir, out_path, corners_str, orient, scale, detect_fn):
    paths = sorted(glob.glob(os.path.join(shotdir, "*.png")))
    if not paths:
        return None
    corners = [(float(a) * scale, float(b) * scale)
               for a, b in (p.split(",") for p in corners_str.split())]

    def load(p):
        img = cv2.imread(p)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else img

    shape = load(paths[0]).shape[:2]
    frames = (load(p) for p in paths)
    tracks = tk.track_clip(frames, shape, corners, orient, detect_fn, mask_pad=12)
    # Gap-filled copy for segmentation/rest-bounds (an unfilled gap reads as rest);
    # the exported positions still come from `tracks` (honest gaps preserved).
    motion = {c: tk.fill_gaps(tracks[c], max_gap=10 ** 9) for c in tk.COLORS}
    shots = tk.segment_shots(motion)
    if not shots:
        return dict(ok=False, reason="no stroke segmented")
    ok = tk.export_for_fit(tracks, shots, tk.FPS, out_path, None, motion=motion,
                           frames_dir=shotdir, corners_str=corners_str, orient=orient)
    if not ok and os.path.exists(out_path):
        os.remove(out_path)  # don't leave a stale .shot for a skipped shot
    return dict(ok=ok, reason=None if ok else "no clean at-rest start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots_root", help="dir containing shot_NN/ subdirs")
    ap.add_argument("--out-dir", default="../data/shots")
    ap.add_argument("--corners", required=True)
    ap.add_argument("--orient", default="vertical")
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--only", type=int, nargs="*", help="only these shot indices")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(args.shots_root, "shot_*")))
    if args.only is not None:
        dirs = [d for d in dirs if int(os.path.basename(d).split("_")[1]) in args.only]

    print(f"loading learned detector once, tracking {len(dirs)} shots…")
    detect_fn = tk.make_learned_detector()
    for d in dirs:
        name = os.path.basename(d)
        out = os.path.join(args.out_dir, f"{name}.shot")
        try:
            r = track_one(d, out, args.corners, args.orient, args.scale, detect_fn)
            print(f"{name}: {'ok -> ' + out if r and r['ok'] else 'FAILED (' + (r['reason'] if r else 'no frames') + ')'}")
        except Exception as e:
            print(f"{name}: ERROR {e}")


if __name__ == "__main__":
    main()
