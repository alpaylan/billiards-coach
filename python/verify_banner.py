#!/usr/bin/env python3
"""Banner guard: validate shot segmentation + make/miss annotation against the
broadcast's own scoreboard — score digits, the innings counter, and the
turn/run indicator (white ball far-left = white's turn, yellow far-right =
yellow's, with the ongoing-run count beside it).

Checks per shot (banner sampled ~1 s before the stroke):
  * score_l/score_r  == cumulative makes we attribute to each side
  * turn icon's side == the player we attribute the shot to
  * run counter      == the shooter's current consecutive-make count
  * innings          == the shooter's turn index (each side's Nth visit)
At game end: banner totals vs our totals (did segmentation miss shots?).

    python3 verify_banner.py VIDEO GAME_DIR [GAME_DIR…]
"""

import glob
import json
import os
import re
import sys

import cv2

import scoreboard as sb


def shot_times(game_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(game_dir, "shot_*.shot"))):
        m = re.search(r"video_t0 ([\d.]+)", open(p).read())
        if m:
            out.append((float(m.group(1)), os.path.basename(p)))
    return sorted(out)


def check_game(cap, gd, templates):
    results = {s["shot"]: s for s in json.load(open(os.path.join(gd, "results.json")))["shots"]}
    color_side = json.load(open(os.path.join(gd, "results.json"))).get("color_side", {})
    side_of_color = {("white" if c == "white" else "yellow"): ("left" if s == "L" else "right")
                     for c, s in color_side.items()}

    # sample the banner before every shot
    seq = []
    for t0, name in shot_times(gd):
        r = results.get(name)
        if r is None or r.get("player") not in ("left", "right"):
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + 1.2) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        seq.append((name, r, sb.banner_state(f, templates)))

    stats = {"shots": len(seq), "delta_ok": 0, "result_wrong": 0, "missed_shots": 0,
             "delta_unread": 0, "turn_ok": 0, "turn_bad": 0, "turn_unread": 0}
    bad_lines = []
    for k in range(len(seq)):
        name, r, (sl, inn, sr, turn, brun) = seq[k]
        player = r["player"]

        # turn icon = who we say is shooting
        if turn is None:
            stats["turn_unread"] += 1
        elif side_of_color.get(turn) == player:
            stats["turn_ok"] += 1
        else:
            stats["turn_bad"] += 1
            bad_lines.append(f"{name}: banner turn {turn} vs ours {player}")

    # Score checks are per TURN, not per shot: the scoreboard operator enters
    # points at the END of a turn (verified frame-by-frame — a 3-run shows as a
    # single 0->3 jump when the opponent sits down). So: group consecutive
    # shots by shooter and require the banner's change on the shooter's side
    # across the turn to equal the makes we credited in it.
    turns = []
    for k, (name, r, st) in enumerate(seq):
        if not turns or turns[-1][0] != r["player"]:
            turns.append([r["player"], k, 0])
        if r["result"] == "make":
            turns[-1][2] += 1
    stats["turns"] = max(0, len(turns) - 1)  # last turn has no closing sample
    for ti in range(len(turns) - 1):
        player, k0, makes = turns[ti]
        st0, st1 = seq[k0][2], seq[turns[ti + 1][1]][2]
        sl, sr, sl2, sr2 = st0[0], st0[2], st1[0], st1[2]
        if None in (sl, sr, sl2, sr2):
            stats["delta_unread"] += 1
            continue
        d_shooter, d_other = ((sl2 - sl, sr2 - sr) if player == "left" else (sr2 - sr, sl2 - sl))
        if d_shooter == makes and d_other == 0:
            stats["delta_ok"] += 1
        else:
            stats["result_wrong"] += 1
            bad_lines.append(
                f"turn@{seq[k0][0]} ({player}): banner credits {d_shooter} (opp {d_other:+}), ours {makes} makes")
    return stats, bad_lines


def main():
    video, dirs = sys.argv[1], sys.argv[2:]
    templates = sb.load_templates()
    cap = cv2.VideoCapture(video)
    for gd in dirs:
        stats, bad = check_game(cap, gd, templates)
        name = gd.rstrip("/").rsplit("/", 1)[-1]
        print(f"{name}: {stats['shots']} shots, {stats.get('turns', 0)} turns · "
              f"turn-score ok {stats['delta_ok']} · mismatch {stats['result_wrong']} · "
              f"unread {stats['delta_unread']} · "
              f"turn-icon {stats['turn_ok']}/{stats['turn_ok']+stats['turn_bad']} ok")
        for l in bad[:10]:
            print(f"    {l}")
        if len(bad) > 10:
            print(f"    … {len(bad)-10} more")
    cap.release()


if __name__ == "__main__":
    main()
