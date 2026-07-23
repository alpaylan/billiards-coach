#!/usr/bin/env python3
"""Build web-bundle games straight from the source video — no frames dirs.

For labeling re-tracks (--no-frames) the per-shot PNG dirs don't exist, but
each shot's clip window is known (its `video_t0` + row span), so the compact
per-shot MP4s can be encoded directly from the match video with one ffmpeg
call each (crop = the preset inset). Writes shots.json (+ copies calibration)
exactly like publish_bundle.publish_game.

    python3 make_bundle_direct.py VIDEO GAME_DIR OUT_DIR [--preset 1080p]
"""

import glob
import json
import os
import re
import subprocess
import sys

INSETS = {"1080p": (15, 405, 290, 515), "720p": (15, 270, 185, 333)}


def main():
    video, gd, out = sys.argv[1], sys.argv[2], sys.argv[3]
    preset = "1080p" if "--preset" not in sys.argv else sys.argv[sys.argv.index("--preset") + 1]
    x, y, w, h = INSETS[preset]
    os.makedirs(out, exist_ok=True)

    results = {}
    rp = os.path.join(gd, "results.json")
    if os.path.exists(rp):
        for s in json.load(open(rp))["shots"]:
            results[s["shot"]] = s

    shots = []
    for p in sorted(glob.glob(os.path.join(gd, "shot_*.shot"))):
        name = os.path.basename(p)
        txt = open(p).read()
        r = results.get(name, {})
        if r.get("result") == "spurious":
            continue  # not a real shot — solver-verified junk
        import shutil
        shutil.copy(p, os.path.join(out, name))
        entry = {"file": name, "player": r.get("player"), "result": r.get("result")}
        v = re.search(r"video_t0 ([\d.]+)", txt)
        # clip duration = last row time + the start offset
        ts = [float(l.split(",")[1]) for l in txt.splitlines() if l.count(",") == 3 and not l.startswith("color")]
        st = re.search(r"start (\d+)", txt)
        if v and ts:
            t0 = float(v.group(1))
            dur = (int(st.group(1)) if st else 0) / 30.0 + max(ts) + 1.0
            mp4 = name.replace(".shot", ".mp4")
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", video,
                   "-t", f"{dur:.2f}", "-vf", f"crop={w}:{h}:{x}:{y}",
                   "-c:v", "libx264", "-crf", "26", "-g", "15", "-pix_fmt", "yuv420p", "-an",
                   "-y", os.path.join(out, mp4)]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                entry["mp4"] = mp4
        shots.append(entry)
    json.dump(shots, open(os.path.join(out, "shots.json"), "w"), indent=1)
    cal = os.path.join(gd, "calibration.json")
    if os.path.exists(cal):
        import shutil
        shutil.copy(cal, out)
    made = sum(1 for s in shots if s.get("result") == "make")
    print(f"{out}: {len(shots)} shots, {made} made")


if __name__ == "__main__":
    main()
