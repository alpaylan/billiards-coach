#!/usr/bin/env python3
"""End-to-end: a YouTube link (or a local match video) -> browsable games.

One command turns a broadcast into a set of games the editor can open, each with
its shots tracked, reconstructed, calibrated to that table, and marked make/miss —
no manual steps in between:

    download (yt-dlp)            the video, if given a URL
      -> segment_games          split into games by the players on the scoreboard
        -> build_match          per game: shot-clock segment, track, stroke-align,
                                 mark make/miss (uses the game's time span)
          -> calibrate_match     per game: fit + store this table's physics
            -> match.json        the manifest the editor's game picker reads

    python3 pipeline.py https://www.youtube.com/watch?v=... --name masa4 --preset 1080p
    python3 pipeline.py ../data/masa4_1080_live.mp4 --name masa4 --preset 1080p
    cargo run -p billiards-ui --release -- data/masa4      # browse the games
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import segment_games as seg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOD_FORMAT = {"1080p": "137", "720p": "136"}  # yt-dlp DASH video-only formats


def download(url, dest, preset):
    if os.path.exists(dest):
        print(f"using cached download {dest}")
        return
    fmt = VOD_FORMAT.get(preset, "137")
    print(f"downloading {url} (format {fmt}) -> {dest} …")
    subprocess.run(["yt-dlp", "-f", f"{fmt}/bestvideo/best", "-o", dest, url], check=True)


def run(cmd, cwd=None):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="YouTube URL or local video file")
    ap.add_argument("--name", default="match")
    ap.add_argument("--preset", default="1080p", choices=["720p", "1080p"])
    ap.add_argument("--out-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--games", default="all", help="'all' or a comma list of game indices")
    ap.add_argument("--games-json", help="reuse an existing segment_games scan instead of re-scanning")
    args = ap.parse_args()

    # 1. get the video
    if args.source.startswith("http"):
        video = os.path.join(args.out_root, f"{args.name}_source.mp4")
        download(args.source, video, args.preset)
    else:
        video = os.path.abspath(args.source)

    match_root = os.path.join(args.out_root, args.name)
    shutil.rmtree(match_root, ignore_errors=True)
    os.makedirs(match_root, exist_ok=True)

    # 2. segment into games (or reuse a prior scan — the full-day scan takes ~20 min)
    if args.games_json:
        with open(args.games_json) as fh:
            prior = json.load(fh)
        fps = prior["fps"]
        games = [{"left": g["left"], "right": g["right"], "t0": g["t0"], "t1": g["t1"],
                  "kl": "", "kr": "", "n": 0} for g in prior["games"]]
        print(f"\n=== reusing scan {args.games_json}: {len(games)} game(s) ===")
    else:
        print(f"\n=== segmenting games from {video} ===")
        fps, games = seg.find_games(video)
        print(f"{len(games)} game(s) found")
    want = range(len(games)) if args.games == "all" else [int(x) for x in args.games.split(",")]

    manifest = {"video": video, "fps": fps, "name": args.name, "games": []}
    for gi in want:
        if gi >= len(games):
            continue
        g = games[gi]
        gname = f"game_{gi:02d}"
        print(f"\n=== {gname}: {g['left']!r} vs {g['right']!r}  ({g['t0']:.0f}-{g['t1']:.0f}s) ===")
        gdir = os.path.join(match_root, gname)

        # 3. build the game's shots (track + align + make/miss), scoped to its span
        # (padded: the first shot's clock reset can precede the first clean
        # scoreboard read by a few seconds, and the last balls can outroll it)
        run([sys.executable, os.path.join(REPO, "python", "build_match.py"), video,
             "--name", os.path.join(args.name, gname), "--preset", args.preset,
             "--t0", max(0.0, g["t0"] - 20.0), "--t1", g["t1"] + 30.0, "--out-root", args.out_root])

        # 4. calibrate this game's table physics + store it
        run(["cargo", "run", "-q", "-p", "billiards-solver", "--example", "calibrate_match",
             "--release", "--", os.path.relpath(gdir, REPO)], cwd=REPO)

        # 5. verify the reconstruction against the tracked original (guards fidelity)
        run(["cargo", "run", "-q", "-p", "billiards-solver", "--example", "verify_match",
             "--release", "--", os.path.relpath(gdir, REPO)], cwd=REPO)

        # per-game name/scoreboard crops
        seg._save_crops(video, fps, g, gdir)
        results = os.path.join(gdir, "results.json")
        made = None
        if os.path.exists(results):
            made = sum(s["result"] == "make" for s in json.load(open(results))["shots"])
        manifest["games"].append({
            "id": gi, "dir": gname, "left": g["left"], "right": g["right"],
            "t0": round(g["t0"], 1), "t1": round(g["t1"], 1),
            "n_shots": len(glob.glob(os.path.join(gdir, "shot_*.shot"))), "n_made": made,
        })

    with open(os.path.join(match_root, "match.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n=== done: {len(manifest['games'])} game(s) -> {match_root}/match.json ===")
    print(f"browse: cargo run -p billiards-ui --release -- {os.path.relpath(match_root, REPO)}")


if __name__ == "__main__":
    main()
