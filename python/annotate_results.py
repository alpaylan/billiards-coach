#!/usr/bin/env python3
"""Mark each tracked shot make/miss and attribute it to a player.

Two robust signals off the broadcast itself:
  * make/miss  = did the shooting side's score box change across the shot? (compare
                 the score box just before the stroke vs a few seconds after it
                 settles — ink-normalized so a single-digit change is caught).
  * shooter    = the cue ball's COLOR. In three-cushion each player owns one cue
                 ball (white or yellow) for the whole match, so cue colour is a
                 fixed player id. We learn colour->side once by correlating a made
                 shot's cue with the side whose score went up, then every shot is
                 attributed with no fragile turn-order reconstruction.

Writes `result make|miss` and `player left|right` into each `.shot` header and a
`results.json` summary in the match dir.

    python3 annotate_results.py MATCH.mp4 ../data/<match> [--offset SECONDS]
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

import scoreboard as sb


def _sbit(frame, box):
    crop = sb._box(frame, box)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if g.mean() < 110:
        g = 255 - g
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.resize(b, (48, 32)) > 127


def _inkdiff(a, b):
    if a is None or b is None:
        return 0.0
    u = (a | b).sum()
    return float(((a != b) & (a | b)).sum()) / max(u, 1)


def _score_bits(cap, fps, t, nsamp=5, spread=1.0):
    """Median score-box bitmaps in a small window around t (robust to a flicker
    or a mid-transition frame). Returns {'L':bitmap,'R':bitmap}."""
    acc = {"L": [], "R": []}
    for dt in np.linspace(-spread, spread, nsamp):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, t + dt) * fps))
        ok, f = cap.read()
        if not ok:
            continue
        for side, box in (("L", sb.SCORE_L), ("R", sb.SCORE_R)):
            acc[side].append(_sbit(f, box))
    out = {}
    for side in ("L", "R"):
        out[side] = (np.stack(acc[side]).mean(0) > 0.5) if acc[side] else None
    return out


def _load_shot(path):
    start, fps, last_t, cue, video_t0 = 0, 30.0, 0.0, "white", None
    with open(path) as fh:
        for ln in fh:
            p = ln.split(",")
            if len(p) == 4 and p[0] in ("white", "yellow", "red"):
                last_t = max(last_t, float(p[1]))
            elif ln.startswith("cue "):
                cue = ln[4:].strip()
            elif ln.startswith("start "):
                start = int(ln.split()[1])
            elif ln.startswith("fps "):
                fps = float(ln.split()[1])
            elif ln.startswith("video_t0 "):
                video_t0 = float(ln.split()[1])
    return {"start": start, "fps": fps, "last_t": last_t, "cue": cue, "video_t0": video_t0}


def _write_header(path, result, player):
    with open(path) as fh:
        lines = fh.read().splitlines()
    lines = [l for l in lines if not l.startswith(("result ", "player "))]
    # insert the annotations right after the cue header line
    out, done = [], False
    for l in lines:
        out.append(l)
        if l.startswith("cue ") and not done:
            out += [f"result {result}", f"player {player}"]
            done = True
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def annotate(video, match_dir, offset=0.0):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n_frames / fps if n_frames > 0 else float("inf")
    # Collect every shot's stroke time first: a shot's scoring window runs from
    # just before ITS stroke to just before the NEXT shot's stroke. The scorer
    # often enters a point 10-20 s after the balls settle (during the opponent's
    # address), so sampling at settle+6s missed late entries entirely; interval
    # attribution assigns every scoreboard change to exactly one shot.
    metas = []
    for path in sorted(glob.glob(os.path.join(match_dir, "shot_*.shot"))):
        s = _load_shot(path)
        base = s["video_t0"] if s["video_t0"] is not None else offset
        ts = base + s["start"] / s["fps"]
        metas.append((path, s, ts))
    metas.sort(key=lambda m: m[2])

    # PRIMARY make/miss signal: the game's own turn structure. In three-cushion
    # a player keeps shooting until they miss, so a shot is a MAKE iff the NEXT
    # shot is taken by the same cue ball. (The broadcast scoreboard can't be a
    # per-shot signal here — the operator enters points in batches, sometimes
    # 70+ s late — so it only serves as a fallback across gaps.) The rule needs
    # the next shot to be the immediately following segmented window; when the
    # window in between failed to track, fall back to the scoreboard interval.
    frames_root = match_dir.rstrip("/") + "_frames"
    all_windows = sorted(
        int(d.split("_")[-1]) for d in os.listdir(frames_root)
        if d.startswith("shot_") and os.path.isdir(os.path.join(frames_root, d))
    ) if os.path.isdir(frames_root) else []
    idx_of = {}
    for path, _s, _ts in metas:
        name = os.path.basename(path)
        idx_of[path] = int(name.split("_")[1].split(".")[0])
    tracked_by_idx = {idx_of[p]: (p, s, ts) for p, s, ts in metas}

    # Score bits at every shot boundary (just before each stroke, plus one after
    # the last shot) — used both for the scoreboard fallback and to learn which
    # cue colour belongs to which side of the scoreboard.
    bounds = [_score_bits(cap, fps, max(ts - 1.2, 1.0)) for _p, _s, ts in metas]
    last_te = metas[-1][2] + metas[-1][1]["last_t"] if metas else 0.0
    bounds.append(_score_bits(cap, fps, min(last_te + 25.0, dur - 1.5)))

    shots = []
    for k, (path, s, ts) in enumerate(metas):
        i = idx_of[path]
        nxt = next((j for j in all_windows if j > i), None)
        made, scored_side, source = None, None, "sequence"
        # Untracked windows between this shot and the next tracked one are
        # usually phantoms — duplicate clock resets fired while this shot's
        # balls were still rolling (a real shot starts from rest, which is why
        # they failed to track). Treat them as transparent when the next tracked
        # shot follows within a normal turnaround; only a long gap means a real
        # shot may be missing, where the sequence says nothing.
        if nxt is not None and nxt not in tracked_by_idx and k + 1 < len(metas):
            if metas[k + 1][2] - ts < 75.0:
                nxt = idx_of[metas[k + 1][0]]
        if nxt is not None and nxt in tracked_by_idx:
            nxt_cue = tracked_by_idx[nxt][1]["cue"]
            made = nxt_cue == s["cue"]
        else:
            # last shot, or the next window didn't track: scoreboard interval
            source = "scoreboard"
            chg = {side: _inkdiff(bounds[k][side], bounds[k + 1][side]) > 0.22
                   for side in ("L", "R")}
            made = chg["L"] or chg["R"]
            scored_side = "L" if chg["L"] else ("R" if chg["R"] else None)
        shots.append({"path": path, "cue": s["cue"], "made": bool(made),
                      "scored_side": scored_side, "ts": round(ts, 1), "source": source})
    cap.release()

    # learn cue-colour -> side: each scoreboard change (they arrive in batches,
    # often late) is voted to the shooter of the interval it landed in.
    vote = {}
    for k, sh in enumerate(shots):
        for side in ("L", "R"):
            if _inkdiff(bounds[k][side], bounds[k + 1][side]) > 0.22:
                vote[(sh["cue"], side)] = vote.get((sh["cue"], side), 0) + 1
    color_side = {}
    for (cue, side), n in sorted(vote.items(), key=lambda kv: -kv[1]):
        if cue not in color_side and side not in color_side.values():
            color_side[cue] = side
    # the other colour maps to the other side
    sides = {"L", "R"}
    for cue in ("white", "yellow"):
        if cue not in color_side:
            taken = set(color_side.values())
            color_side[cue] = (sides - taken).pop() if len(taken) == 1 else "L"

    side_name = {"L": "left", "R": "right"}
    results = []
    for sh in shots:
        player = side_name[color_side.get(sh["cue"], "L")]
        result = "make" if sh["made"] else "miss"
        _write_header(sh["path"], result, player)
        results.append({"shot": os.path.basename(sh["path"]), "player": player,
                        "cue": sh["cue"], "result": result, "t": sh["ts"]})
        print(f"  {os.path.basename(sh['path']):<14} {player:<5} ({sh['cue']:<6}) {result}")

    with open(os.path.join(match_dir, "results.json"), "w") as fh:
        json.dump({"color_side": color_side, "shots": results}, fh, indent=2)
    made = sum(r["result"] == "make" for r in results)
    print(f"{made}/{len(results)} made; cue map {color_side} -> {match_dir}/results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("match_dir")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds to add to each shot's .shot start-frame time to hit the video")
    args = ap.parse_args()
    annotate(args.video, args.match_dir, args.offset)


if __name__ == "__main__":
    main()
