#!/usr/bin/env python3
"""End-to-end: a recorded match video -> a browsable set of tracked shots.

Chains the pipeline so one command turns a match `.mp4` into `.shot` files the
editor can browse as a whole match:

  1. detect_shots  — segment the game into shots via the shot clock + ball motion
  2. extract       — dump each shot's overhead-inset frames
  3. track_shots   — learned-detector tracking -> a color-labeled .shot per shot
                     (linked to its real frames, so the editor plays the footage)

    python3 build_match.py ../data/masa4_live/masa4_match_XXXX.mp4 --name masa4_match
    cargo run -p billiards-ui --release -- data/masa4_match      # browse it

Defaults are the MASA 4 / bilardo inset geometry; override for another source.
"""

import argparse
import glob
import os

import detect_shots as ds
import track as tk
import track_shots as ts

# MASA 4 / bilardo geometry per stream resolution. The overhead inset and the
# scoreboard both scale with frame height, so the shot-clock detection is
# resolution-independent (scoreboard.py handles that); only the inset crop
# geometry differs. `inset` = extraction crop (x,y,w,h); `interior` = the blue
# playing surface in *full-frame* coords for motion (x0,x1,y0,y1); `corners` are
# in the extracted-inset crop's coords.
PRESETS = {
    "720p": dict(inset=(15, 270, 185, 333), interior=(25, 190, 285, 595),
                 corners="7,12 172,10 174,327 9,327", scale=4.0),
    "1080p": dict(inset=(15, 405, 290, 515), interior=(50, 270, 440, 880),
                  corners="19,19 264,15 270,492 23,492", scale=3.0),
    # bilardo.com.tr Gölbaşı productions (Hakan Keleş 2024 matches; 2026 daily
    # tournament streams). Same banner/clock family as masa4; the overhead
    # inset sits bottom-RIGHT in the 2024 videos and mid-LEFT in the 2026
    # tournament streams. Corners are DRAFT (eyeballed from frames) — refine
    # against recon quality on first use.
    "bilardo_r": dict(inset=(1585, 395, 320, 520), interior=(390, 1780, 210, 890),
                      corners="27,23 300,18 305,497 20,503", scale=3.0),
    "bilardo_l": dict(inset=(10, 405, 295, 530), interior=(420, 1800, 170, 860),
                      corners="28,37 262,33 266,500 25,503", scale=3.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default="match")
    ap.add_argument("--out-root", default="../data")
    ap.add_argument("--preset", choices=list(PRESETS), default="720p")
    ap.add_argument("--orient", default="vertical")
    ap.add_argument("--t0", type=float, default=0.0, help="only build shots after this video time (s)")
    ap.add_argument("--t1", type=float, default=float("inf"), help="only build shots before this video time (s)")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    ds.INSET, ds.INTERIOR = p["inset"], p["interior"]  # shot-clock detection reads these
    args.corners, args.scale = p["corners"], p["scale"]

    frames_root = os.path.join(args.out_root, f"{args.name}_frames")
    shots_out = os.path.join(args.out_root, args.name)
    os.makedirs(shots_out, exist_ok=True)

    print(f"[1/3] segmenting shots from {args.video} …")
    # Scan only the requested span (a game of a broadcast day), then shift the
    # analyzer's span-relative times back to absolute video time.
    fps, t, frac, act, motion, zero, fine = ds.analyze(args.video, t0=args.t0, t1=args.t1)
    shots = ds.find_shots(fps, t, frac, fine)
    for s in shots:
        for k in ("win_start", "win_end", "onset", "settle", "stroke_t"):
            s[k] += args.t0
    starts = [float(t[k]) for k in range(len(zero)) if zero[k]]
    print(f"      {len(shots)} shots; clock active {act.mean()*100:.0f}%"
          + (f"; game-start (0-0) at ~{starts[0]:.0f}s" if starts else "; began mid-game"))
    ds.montage(args.video, shots, os.path.join(args.out_root, f"{args.name}_shots.png"), fps)
    ds.extract_inset(args.video, shots, frames_root, fps)

    print(f"[2/3] loading learned detector, tracking {len(shots)} shots …")
    detect_fn = tk.make_learned_detector()
    ok = 0
    for d in sorted(glob.glob(os.path.join(frames_root, "shot_*"))):
        name = os.path.basename(d)
        out = os.path.join(shots_out, f"{name}.shot")
        try:
            r = ts.track_one(d, out, args.corners, args.orient, args.scale, detect_fn)
            good = bool(r and r.get("ok"))
            ok += good
            if good:
                # record where this shot sits in the ORIGINAL video (extracted frame 0
                # = onset-PAD_PRE) so downstream steps (score reading, editor seek) can
                # map shot-local time back to the match timeline.
                nn = int(name.split("_")[1])
                if nn < len(shots):
                    v0 = max(0.0, shots[nn]["onset"] - ds.PAD_PRE)
                    with open(out, "a") as fh:
                        fh.write(f"video_t0 {v0:.3f}\n")
            print(f"      {name}: {'ok' if good else 'no stroke'}")
        except Exception as e:
            print(f"      {name}: ERROR {e}")

    # Move t=0 to the stroke so the reconstruction launches when the ball does,
    # not at the (padded) clip start — otherwise the sim races ahead of the still
    # pre-stroke ball and the whole reconstruction is time-shifted.
    import align_shots
    n_aligned = sum(bool(align_shots.align(p)) for p in glob.glob(os.path.join(shots_out, "shot_*.shot")))
    print(f"      stroke-aligned {n_aligned} shots (trimmed pre-stroke lead-in)")

    # Repair white↔yellow identity swaps + fabricated swap-bridge samples
    # (belt-and-braces: track.py now arbitrates by continuity, this cleans any
    # flip that still slips through).
    import fix_swaps
    for p in glob.glob(os.path.join(shots_out, "shot_*.shot")):
        fix_swaps.fix(p)

    # Mark make/miss + attribute each shot to a player from the scoreboard.
    print("      reading scoreboard for make/miss + player …")
    import annotate_results
    try:
        annotate_results.annotate(args.video, shots_out)
    except Exception as e:
        print(f"      (result annotation skipped: {e})")

    print(f"[3/3] done — {ok}/{len(shots)} shots tracked -> {shots_out}/")
    print(f"      browse: cargo run -p billiards-ui --release -- {shots_out}")


if __name__ == "__main__":
    main()
