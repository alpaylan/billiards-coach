#!/usr/bin/env python3
"""Segment a match video into GAMES, each identified by its two players.

A game is a maximal span of clock-active play with a stable matchup — the pair of
name plates on the scoreboard. Games break where the matchup changes (a new pair of
players sits down). Within one matchup, several games may run back-to-back; those
are split later by score resets (make/miss pass), but the matchup split already
gives the user the "pick a game by who's playing" view.

Output: `<out>/games.json` (each game's time range + players) plus per-game
name-plate crops and a scoreboard thumbnail for the visual picker.

    python3 segment_games.py MATCH.mp4 --out ../data/<name>_games
"""

import argparse
import json
import os
import re
import shutil

import cv2

import scoreboard as sb


def _key(name):
    """OCR string -> a normalized identity key (letters/digits only)."""
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def _close(a, b):
    """Two matchup keys refer to the same player, tolerating OCR noise/clipping."""
    if not a or not b:
        return False
    if a == b:
        return True
    s, l = sorted((a, b), key=len)
    if len(s) >= 4 and s in l:
        return True
    common = sum(1 for c in set(s) if c in l)
    return len(set(s)) > 0 and common / len(set(s)) > 0.78


def find_games(video, scan_dt=2.0, ocr_every=15.0, min_dur=120.0, min_reads=3,
               merge_gap=12 * 60.0):
    """Scan the scoreboard and return (fps, [game,...]). Each game dict carries the
    matchup keys, raw OCR names, and [t0,t1] span.

    Built for a full broadcast day (many hours, several games):
      * seek-based sampling every `scan_dt` s (a sequential decode of 11+ hours
        would take longer than the games themselves);
      * a game is a maximal clock-active run of ONE matchup, merged across gaps up
        to `merge_gap` — the mid-game "MOLA" timeout (~5 min, card on the table,
        clock off) must NOT split a game;
      * a gap is a real boundary when the matchup changes, when both score boxes
        come back at 0-0 having been non-zero (back-to-back games of the same
        pair), or when it simply exceeds `merge_gap`;
      * practice/warm-up runs untimed (no clock), so it is skipped automatically.

    Names are OCRed only every `ocr_every` s of active play (tesseract is the
    slow part); the clock/score reads are cheap numpy on the sampled frame."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    zt = _zero_template()

    games = []
    last_ocr = -1e9
    names = ("", "")
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if not ok:
            t += scan_dt
            continue
        active, _ = sb.clock(f)
        if active:
            if t - last_ocr >= ocr_every or not (names[0] and names[1]):
                rl, rr = sb.names(f)
                if _key(rl) and _key(rr):
                    names = (rl, rr)
                    last_ocr = t
            kl, kr = _key(names[0]), _key(names[1])
            zero = bool(zt is not None and sb.scores_zero(f, zt))
            if kl and kr:
                g = games[-1] if games else None
                # Back-to-back games of the SAME pair: scores return to 0-0 after
                # a gap, when this game had already moved past 0-0.
                fresh = bool(zero and g and g.get("nonzero_seen") and t - g["t1"] > 60.0)
                if (g and _close(g["kl"], kl) and _close(g["kr"], kr)
                        and t - g["t1"] < merge_gap and not fresh):
                    g["t1"], g["n"] = t, g["n"] + 1
                    g["nonzero_seen"] = g.get("nonzero_seen", False) or not zero
                else:
                    games.append({"kl": kl, "kr": kr, "left": names[0], "right": names[1],
                                  "t0": t, "t1": t, "n": 1, "nonzero_seen": not zero})
        t += scan_dt
    cap.release()
    return fps, [g for g in games if g["t1"] - g["t0"] >= min_dur and g["n"] >= min_reads]


def _zero_template():
    """The '0' digit template detect_shots uses for fresh-game (0-0) detection."""
    import numpy as np
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_template.npy")
    return np.load(p) if os.path.exists(p) else None


def _save_crops(video, fps, game, gdir):
    """Grab a representative mid-game frame; save name plates + scoreboard strip."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((game["t0"] + game["t1"]) / 2 * fps))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return
    lc, rc = sb.name_crops(f)
    cv2.imwrite(os.path.join(gdir, "player_left.png"), lc)
    cv2.imwrite(os.path.join(gdir, "player_right.png"), rc)
    h = f.shape[0]
    s = h / 720.0
    band = f[int(600 * s):int(665 * s), int(160 * s):int(1130 * s)]
    cv2.imwrite(os.path.join(gdir, "scoreboard.png"), band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="../data/games")
    ap.add_argument("--scan-dt", type=float, default=2.0)
    args = ap.parse_args()

    print(f"scanning {args.video} for games …")
    fps, games = find_games(args.video, scan_dt=args.scan_dt)
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for k, g in enumerate(games):
        gdir = os.path.join(args.out, f"game_{k:02d}")
        os.makedirs(gdir, exist_ok=True)
        _save_crops(args.video, fps, g, gdir)
        rec = {"id": k, "left": g["left"], "right": g["right"],
               "t0": round(g["t0"], 1), "t1": round(g["t1"], 1),
               "dir": f"game_{k:02d}"}
        manifest.append(rec)
        print(f"  game_{k:02d}: {g['left']!r} vs {g['right']!r}  "
              f"{g['t0']:.0f}-{g['t1']:.0f}s ({g['t1']-g['t0']:.0f}s, {g['n']} reads)")

    with open(os.path.join(args.out, "games.json"), "w") as fh:
        json.dump({"video": os.path.abspath(args.video), "fps": fps, "games": manifest}, fh, indent=2)
    print(f"{len(games)} game(s) -> {args.out}/games.json")


if __name__ == "__main__":
    main()
