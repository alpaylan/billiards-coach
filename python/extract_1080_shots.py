#!/usr/bin/env python3
"""Find and extract shots from a 1080p MASA 4 clip into per-shot inset frame dirs.

Uses inset ball-motion (frame differencing) to locate shots — the 1080p inset is
crisp enough to track directly, so we don't need the scoreboard/shot-clock OCR
here. Each detected shot's overhead inset is dumped to `<out>/shot_NN/`, ready for
`track_shots.py` (learned detector, scale 3, the 1080p corners).

    python3 extract_1080_shots.py CLIP.mp4 --out ../data/masa4_1080_frames
"""

import argparse
import os
import shutil

import cv2
import numpy as np

INSET = (15, 405, 290, 515)          # x, y, w, h crop of the 1080p inset (+ rail)
INTERIOR = (25, 265, 30, 490)        # x0, x1, y0, y1 of blue interior for motion


def find_shots(video, scan_hz=10.0):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / scan_hz)))
    ix0, iy, iw, ih = INSET
    x0, x1, y0, y1 = INTERIOR
    prev, i, mot, t = None, 0, [], []
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                sub = f[iy + y0:iy + y1, ix0 + x0:ix0 + x1]
                g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.int16)
                mot.append(0 if prev is None else int(np.count_nonzero(np.abs(g - prev) > 28)))
                prev = g
                t.append(i / fps)
        i += 1
    cap.release()
    mot, t = np.array(mot), np.array(t)
    active = mot > 120
    shots, k = [], 0
    while k < len(active):
        if active[k]:
            j = k
            gap = 0
            while j + 1 < len(active) and (active[j + 1] or gap < scan_hz * 0.8):
                j += 1
                gap = 0 if active[j] else gap + 1
            while j > k and not active[j]:
                j -= 1
            a, b, peak = t[k], t[j], int(mot[k:j + 1].max())
            if b - a >= 1.0 and peak >= 400:      # a real shot, not a flicker
                shots.append((a, b, peak))
            k = j + 1
        else:
            k += 1
    return fps, shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="../data/masa4_1080_frames")
    args = ap.parse_args()

    fps, shots = find_shots(args.video)
    print(f"found {len(shots)} shots:")
    shutil.rmtree(args.out, ignore_errors=True)
    ix, iy, iw, ih = INSET
    for k, (a, b, peak) in enumerate(shots):
        d = os.path.join(args.out, f"shot_{k:02d}")
        os.makedirs(d)
        ss = max(0.0, a - 0.7)
        dur = (b + 1.2) - ss
        os.system(f'ffmpeg -nostdin -loglevel error -ss {ss:.2f} -t {dur:.2f} -i "{args.video}" '
                  f'-vf "crop={iw}:{ih}:{ix}:{iy},fps=30" -y "{d}/f_%04d.png"')
        n = len(os.listdir(d))
        print(f"  shot_{k:02d}: {a:.0f}-{b:.0f}s (peak {peak}) -> {n} frames")
    print(f"extracted to {args.out}/ — now: track_shots.py {args.out} "
          f'--corners "19,19 264,15 270,492 23,492" --orient vertical --scale 3')


if __name__ == "__main__":
    main()
