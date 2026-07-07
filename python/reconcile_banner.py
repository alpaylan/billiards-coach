#!/usr/bin/env python3
"""Reconcile shot annotations against the banner — and derive the TRUE labels.

Authorities, in order:
  * the TURN ICON (white ball far-left / yellow far-right) says who is
    shooting — it defines turn grouping;
  * SCORE DELTAS across turn boundaries say how many points each turn earned
    (the scoreboard operator enters points at end of turn);
  * the CONTINUATION RULE (a player keeps shooting only after a make) then
    fully determines per-shot results inside a turn of N shots earning P
    points: the first N-1 shots are makes, the last is a make iff P == N.
    P outside {N-1, N} flags the turn as structurally inconsistent
    (mis-grouped or mis-segmented) instead of silently guessing.

Reports disagreements with the current results.json / .shot headers;
`--fix` rewrites both (originals backed up as results.json.bak).

    python3 reconcile_banner.py VIDEO GAME_DIR… [--fix]
"""

import glob
import json
import os
import re
import shutil
import sys

import cv2

import scoreboard as sb


def shot_rows(gd):
    out = []
    for p in sorted(glob.glob(os.path.join(gd, "shot_*.shot"))):
        m = re.search(r"video_t0 ([\d.]+)", open(p).read())
        if m:
            out.append((float(m.group(1)), os.path.basename(p), p))
    return sorted(out)


def sample(cap, t, templates):
    cap.set(cv2.CAP_PROP_POS_MSEC, (t + 1.2) * 1000)
    ok, f = cap.read()
    return sb.banner_state(f, templates) if ok else (None,) * 5


def reconcile_game(cap, gd, templates, fix):
    res_path = os.path.join(gd, "results.json")
    if os.path.exists(res_path):
        doc = json.load(open(res_path))
    else:
        # fresh track (no annotate_results pass): derive everything from banner
        doc = {"color_side": {}, "shots": [{"shot": os.path.basename(p)}
               for p in sorted(glob.glob(os.path.join(gd, "shot_*.shot")))]}
    results = {s["shot"]: s for s in doc["shots"]}
    color_side = doc.get("color_side", {})  # cue color -> 'L'/'R'
    side_of = {c: ("left" if s == "L" else "right") for c, s in color_side.items()}

    rows = shot_rows(gd)
    states = [sample(cap, t, templates) for t, _, _ in rows]

    # shooter per shot: the icon, holes filled by same-colored neighbors,
    # falling back to the existing annotation
    shooters = []
    for k, (t, name, _) in enumerate(rows):
        icon = states[k][3]
        if icon is None:
            prev = next((states[j][3] for j in range(k - 1, -1, -1) if states[j][3]), None)
            nxt = next((states[j][3] for j in range(k + 1, len(rows)) if states[j][3]), None)
            icon = prev if prev == nxt else None
        shooters.append(icon)
    for k, (t, name, _) in enumerate(rows):
        if shooters[k] is None:
            r = results.get(name, {})
            # annotation side -> cue color
            inv = {v: c for c, v in side_of.items()}
            shooters[k] = inv.get(r.get("player"))

    # turns by consecutive shooter
    turns = []
    for k, s in enumerate(shooters):
        if not turns or turns[-1]["cue"] != s:
            turns.append({"cue": s, "shots": []})
        turns[-1]["shots"].append(k)

    for t in turns:
        t["delta_k"] = t["shots"][0]  # pre-adjustment boundary, for score deltas

    # NOTE (measured, not yet exploited): the run counter shows the operator
    # flips the icon ~one shot late at ~40% of boundaries ("first shot of a
    # turn already shows run=1"). A greedy boundary shift by that count makes
    # things WORSE (27/48 vs 41/48 consistent turns on the pilot) — the +1 has
    # other causes at some boundaries. Correcting this properly is a joint
    # solve over (boundaries, labels) against icons + runs + deltas +
    # continuation rule; until then, plain icon grouping stands.

    # color<->side mapping straight from the banner when absent: whichever
    # side's score moves across a white turn's boundary is white's side.
    if not side_of:
        votes = {"white": {"left": 0, "right": 0}, "yellow": {"left": 0, "right": 0}}
        for ti in range(len(turns) - 1):
            t = turns[ti]
            st0, st1 = states[t["shots"][0]], states[turns[ti + 1]["shots"][0]]
            if t["cue"] in votes and None not in (st0[0], st0[2], st1[0], st1[2]):
                if st1[0] > st0[0] and st1[2] == st0[2]:
                    votes[t["cue"]]["left"] += 1
                elif st1[2] > st0[2] and st1[0] == st0[0]:
                    votes[t["cue"]]["right"] += 1
        for c, v in votes.items():
            if v["left"] != v["right"]:
                side_of[c] = "left" if v["left"] > v["right"] else "right"
        doc["color_side"] = {c: ("L" if s == "left" else "R") for c, s in side_of.items()}

    stats = {"turns": len(turns), "consistent": 0, "inconsistent": 0, "unread": 0,
             "relabel_result": 0, "relabel_player": 0}
    lines = []
    derived = {}  # shot name -> (cue_color, result)
    for ti in range(len(turns) - 1):
        t = turns[ti]
        k0 = t["shots"][0]
        st0, st1 = states[t["delta_k"]], states[turns[ti + 1]["delta_k"]]
        if t["cue"] not in ("white", "yellow") or None in (st0[0], st0[2], st1[0], st1[2]):
            stats["unread"] += 1
            continue
        side = side_of.get(t["cue"], "left")
        p = (st1[0] - st0[0]) if side == "left" else (st1[2] - st0[2])
        n = len(t["shots"])
        if p == n or p == n - 1:
            stats["consistent"] += 1
            for j, k in enumerate(t["shots"]):
                make = j < n - 1 or p == n
                derived[rows[k][1]] = (t["cue"], "make" if make else "miss")
        else:
            stats["inconsistent"] += 1
            lines.append(f"  INCONSISTENT turn@{rows[k0][1]} ({t['cue']}): {n} shots but banner credits {p}")

    # diff derived vs current annotations
    for name, (cue, result) in derived.items():
        r = results.get(name)
        if r is None:
            continue
        want_player = side_of.get(cue)
        if r.get("result") != result:
            stats["relabel_result"] += 1
            lines.append(f"  {name}: result {r.get('result')} -> {result}")
        if want_player and r.get("player") != want_player:
            stats["relabel_player"] += 1
            lines.append(f"  {name}: player {r.get('player')} -> {want_player} (cue {cue})")
        if fix:
            r["result"] = result
            if want_player:
                r["player"] = want_player
            r["cue"] = cue

    if fix:
        shutil.copy(res_path, res_path + ".bak")
        json.dump(doc, open(res_path, "w"), indent=1)
        # update .shot headers too (result/player lines)
        for _, name, path in rows:
            if name not in derived:
                continue
            cue, result = derived[name]
            want_player = side_of.get(cue, "")
            txt = open(path).read().splitlines()
            txt = [l for l in txt if not (l.startswith("result ") or l.startswith("player "))]
            txt.insert(1, f"player {want_player}")
            txt.insert(1, f"result {result}")
            open(path, "w").write("\n".join(txt) + "\n")
    return stats, lines


def main():
    args = [a for a in sys.argv[1:] if a != "--fix"]
    fix = "--fix" in sys.argv
    video, dirs = args[0], args[1:]
    templates = sb.load_templates()
    cap = cv2.VideoCapture(video)
    for gd in dirs:
        stats, lines = reconcile_game(cap, gd, templates, fix)
        name = gd.rstrip("/").rsplit("/", 1)[-1]
        print(f"{name}: {stats['turns']} turns · consistent {stats['consistent']} · "
              f"INCONSISTENT {stats['inconsistent']} · unread {stats['unread']} · "
              f"relabels: result {stats['relabel_result']}, player {stats['relabel_player']}"
              f"{'  [FIXED]' if fix else ''}")
        for l in lines[:14]:
            print(l)
        if len(lines) > 14:
            print(f"  … {len(lines)-14} more")
    cap.release()


if __name__ == "__main__":
    main()
