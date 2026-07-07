#!/usr/bin/env python3
"""Build digit templates (0-9) for the banner reader — self-labeled.

For every annotated game we KNOW each side's cumulative score before every
shot (results.json make/miss sequence), and each shot's absolute video time
(`video_t0` in its .shot header). Sampling the banner just before each stroke
therefore yields score-box glyphs with known digit labels — no manual labeling.
Templates are deduplicated per digit and saved to digit_templates.npz.

    python3 harvest_digits.py VIDEO GAME_DIR [GAME_DIR…]
"""

import glob
import json
import os
import re
import sys

import cv2
import numpy as np

import scoreboard as sb


def shot_times(game_dir):
    """[(video_t0, shot_name)] sorted by time."""
    out = []
    for p in sorted(glob.glob(os.path.join(game_dir, "shot_*.shot"))):
        m = re.search(r"video_t0 ([\d.]+)", open(p).read())
        if m:
            out.append((float(m.group(1)), os.path.basename(p)))
    return sorted(out)


def main():
    video, dirs = sys.argv[1], sys.argv[2:]
    cap = cv2.VideoCapture(video)
    bank = {d: [] for d in range(10)}
    n_frames = 0
    for gd in dirs:
        results = {s["shot"]: s for s in json.load(open(os.path.join(gd, "results.json")))["shots"]}
        cum = {"left": 0, "right": 0}
        for t0, name in shot_times(gd):
            r = results.get(name)
            if r is None:
                continue
            # banner state BEFORE this shot reflects all previous shots
            cap.set(cv2.CAP_PROP_POS_MSEC, (t0 - 1.0) * 1000)
            ok, f = cap.read()
            if ok:
                n_frames += 1
                for side, box in (("left", sb.SCORE_L), ("right", sb.SCORE_R)):
                    expect = str(cum[side])
                    gl = sb._glyphs(sb._box(f, box))
                    if len(gl) == len(expect):
                        for ch, g in zip(expect, gl):
                            bank[int(ch)].append(g)
            if r["result"] == "make" and r["player"] in cum:
                cum[r["player"]] += 1
    cap.release()

    out = {}
    for d, gs in bank.items():
        uniq = []
        for g in gs:
            if not any((g == u).mean() > 0.95 for u in uniq):
                uniq.append(g)
        if uniq:
            out[str(d)] = np.stack(uniq[:12])
        print(f"digit {d}: {len(gs)} samples -> {len(uniq)} templates")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_templates.npz"), **out)
    print(f"harvested from {n_frames} frames -> digit_templates.npz")


if __name__ == "__main__":
    main()
