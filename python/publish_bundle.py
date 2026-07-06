#!/usr/bin/env python3
"""Publish a processed match as a static web bundle for the browser viewer.

Takes a match root (the editor-browsable directory with `match.json` +
`game_XX/` dirs) and emits a self-contained `bundle/` the viewer fetches over
HTTP:

    bundle/match.json
    bundle/game_00/{shots.json, calibration.json, shot_XXX.shot, shot_XXX.mp4}

Frames become one small H.264 MP4 per shot (Cloudflare Pages caps deployments
at 20k files, so shipping PNG frame dirs is out of the question; ~700 KB/shot
instead of ~8 MB).

    python3 publish_bundle.py ../data/masa4 --out ../web/dist/bundle
"""

import argparse
import glob
import json
import os
import shutil
import subprocess


def encode_mp4(frames_dir, out_mp4):
    """Frames dir (f_%04d.png) -> a compact even-dimensioned H.264 MP4."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-framerate", "30",
         "-i", os.path.join(frames_dir, "f_%04d.png"),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         # short GOP (-g 15): the viewer scrubs by seeking, and seeks land on
         # keyframes — a default 250-frame GOP would make scrubbing jump ~8s
         "-c:v", "libx264", "-crf", "26", "-g", "15", "-pix_fmt", "yuv420p", "-y", out_mp4],
        capture_output=True,
    )
    return r.returncode == 0


def publish_game(match_root, gdir_name, out_dir):
    src = os.path.join(match_root, gdir_name)
    os.makedirs(out_dir, exist_ok=True)
    # results.json -> shots.json (ordered index the viewer walks)
    results = {}
    rp = os.path.join(src, "results.json")
    if os.path.exists(rp):
        for s in json.load(open(rp))["shots"]:
            results[s["shot"]] = s
    shots = []
    for p in sorted(glob.glob(os.path.join(src, "shot_*.shot"))):
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(out_dir, name))
        r = results.get(name, {})
        entry = {"file": name, "player": r.get("player"), "result": r.get("result")}
        # footage: the .shot header points at its frames dir
        frames = None
        for ln in open(p):
            if ln.startswith("frames "):
                frames = ln.split(None, 1)[1].strip()
                break
        if frames and os.path.isdir(frames):
            mp4 = name.replace(".shot", ".mp4")
            if encode_mp4(frames, os.path.join(out_dir, mp4)):
                entry["mp4"] = mp4
        shots.append(entry)
    with open(os.path.join(out_dir, "shots.json"), "w") as fh:
        json.dump(shots, fh, indent=1)
    for extra in ("calibration.json", "verify.json"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copy(p, out_dir)
    return len(shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_root")
    ap.add_argument("--out", default="../web/dist/bundle")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.match_root, "match.json")))
    os.makedirs(args.out, exist_ok=True)
    for g in manifest["games"]:
        n = publish_game(args.match_root, g["dir"], os.path.join(args.out, g["dir"]))
        # refresh counts from the (re-annotated) results rather than trusting
        # whatever the manifest recorded at build time
        sj = os.path.join(args.out, g["dir"], "shots.json")
        shots = json.load(open(sj))
        g["n_shots"] = len(shots)
        g["n_made"] = sum(1 for s in shots if s.get("result") == "make")
        print(f"  {g['dir']}: {n} shots, {g['n_made']} made ({g['left']} vs {g['right']})")
    # the viewer only needs the games list
    slim = {"games": manifest["games"]}
    with open(os.path.join(args.out, "match.json"), "w") as fh:
        json.dump(slim, fh, indent=1)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs)
    print(f"bundle -> {args.out} ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
