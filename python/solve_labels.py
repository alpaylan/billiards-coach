#!/usr/bin/env python3
"""Joint label solver: choose per-shot outcomes that best explain the banner.

Instead of trusting any single signal (the icon lags ~a shot at ~40% of turn
boundaries; the run counter is live but only readable sometimes; the score
digits are batched per turn), solve for the maximum-agreement assignment by
dynamic programming over the shot sequence.

Model per shot k (state BEFORE the shot): shooter color c, run r = makes so
far this turn, cumulative points per color. The shot is then one of:
  make      -> (c, r+1, cum[c]+1)
  miss      -> turn ends; next state (other(c), 0, cum)
  spurious  -> not a real shot (junk/duplicate that survived the gates);
               state unchanged, fixed penalty.
Emissions: icon match (cheap to violate at r==0 — the operator's late flip),
run-counter match, score-digit match (soft, batched). Exact DP, then
backtrack to labels.

    python3 solve_labels.py VIDEO GAME_DIR [--fix]
"""

import glob
import json
import os
import sys

import cv2

import scoreboard as sb
import reconcile_banner as rb

COST_SPURIOUS = 1.6  # was 3.0: a mola (timeout) after a missed shot tracks as a
#                      junk candidate; at 3.0 the DP preferred relabeling the
#                      real miss as a make over marking the junk spurious
#                      (verified against 5 hand-labeled turns on YouTube)
COST_ICON_MIDTURN = 2.5
COST_ICON_TURNSTART = 0.4
COST_RUN = 2.0
COST_SCORE_UNIT = 1.0
SCORE_CAP = 2.5
MAX_RUN = 7


def objects_moved(shot_path):
    """How many NON-CUE balls left their rest (>2.5 cm) in this shot's track.
    A three-cushion point requires contacting BOTH object balls — a shot where
    fewer than two moved cannot be a make. This is the physics vote that
    settles 'banner paid nothing' turns: the ball tracks already know the shot
    hit nothing."""
    cue, tracks = None, {}
    for line in open(shot_path):
        if line.startswith("cue "):
            cue = line.split()[1].strip()
        f = line.strip().split(",")
        if len(f) == 4 and f[0] in ("white", "yellow", "red"):
            tracks.setdefault(f[0], []).append((float(f[2]), float(f[3])))
    n = 0
    for c, pts in tracks.items():
        if c == cue or not pts:
            continue
        x0, y0 = pts[0]
        if max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in pts) > 0.025:
            n += 1
    return n


COST_MAKE_IMPOSSIBLE = 6.0  # near-veto: contacting both object balls is REQUIRED to score


def solve(states, side_of, final_score=None, moved=None):
    """states: per shot (sl, inn, sr, icon, run). `final_score` = (sl, sr) read
    off the POST-GAME board — it anchors the last turn, whose points otherwise
    have no closing sample (and which may legally end on the winning make).
    Returns list of ('make'|'miss'|'spurious', cue) per shot, plus total cost."""
    n = len(states)
    colors = ("white", "yellow")

    def emit(k, c, r, cw, cy):
        sl, _inn, sr, icon, brun = states[k]
        cost = 0.0
        if icon is not None and icon != c:
            cost += COST_ICON_TURNSTART if r == 0 else COST_ICON_MIDTURN
        if brun is not None and brun != r:
            cost += COST_RUN
        if sl is not None and sr is not None and side_of:
            # The operator enters points at END of turn: the shooter's r
            # in-progress makes are NOT on the board yet.
            cum = {"white": cw, "yellow": cy}
            cum[c] -= r
            exp_l = cum[next(cc for cc, s in side_of.items() if s == "left")]
            exp_r = cum[next(cc for cc, s in side_of.items() if s == "right")]
            cost += min(SCORE_CAP, COST_SCORE_UNIT * abs(sl - exp_l))
            cost += min(SCORE_CAP, COST_SCORE_UNIT * abs(sr - exp_r))
        return cost

    # Scores can start non-zero (capture truncation): offset by first readable.
    off_w = off_y = 0
    for st in states:
        if st[0] is not None and st[2] is not None and side_of:
            by_side = {"left": st[0], "right": st[2]}
            off = {c: by_side[s] for c, s in side_of.items()}
            off_w, off_y = off.get("white", 0), off.get("yellow", 0)
            break

    # DP: dict state -> (cost, parent, action)
    start = {}
    for c in colors:
        start[(c, 0, off_w, off_y)] = (0.0, None, None)
    layers = [start]
    for k in range(n):
        cur = layers[-1]
        nxt = {}

        def push(state, cost, parent, action):
            if state not in nxt or cost < nxt[state][0]:
                nxt[state] = (cost, parent, action)

        for (c, r, cw, cy), (cost, _, _) in cur.items():
            e = emit(k, c, r, cw, cy)
            other = "yellow" if c == "white" else "white"
            # make (a track that moved <2 object balls cannot score)
            if r + 1 <= MAX_RUN:
                phys = COST_MAKE_IMPOSSIBLE if moved is not None and moved[k] < 2 else 0.0
                cw2, cy2 = (cw + 1, cy) if c == "white" else (cw, cy + 1)
                push((c, r + 1, cw2, cy2), cost + e + phys, (c, r, cw, cy), ("make", c))
            # miss -> turn ends
            push((other, 0, cw, cy), cost + e, (c, r, cw, cy), ("miss", c))
            # spurious
            push((c, r, cw, cy), cost + e + COST_SPURIOUS, (c, r, cw, cy), ("spurious", c))
        layers.append(nxt)

    def terminal(state):
        if not final_score or not side_of:
            return 0.0
        c, r, cw, cy = state
        cum = {"white": cw, "yellow": cy}
        exp_l = cum[next(cc for cc, s in side_of.items() if s == "left")]
        exp_r = cum[next(cc for cc, s in side_of.items() if s == "right")]
        fl, fr = final_score
        cost = 0.0
        if fl is not None:
            cost += min(4.0, abs(fl - exp_l))
        if fr is not None:
            cost += min(4.0, abs(fr - exp_r))
        return cost

    end_state, (total, _, _) = min(
        ((st, v) for st, v in layers[-1].items()),
        key=lambda kv: kv[1][0] + terminal(kv[0]),
    )
    total += terminal(end_state)
    # backtrack
    actions = []
    state = end_state
    for k in range(n, 0, -1):
        cost, parent, action = layers[k][state]
        actions.append(action)
        state = parent
    actions.reverse()
    return actions, total


def run_game(cap, gd, templates, fix):
    rows = rb.shot_rows(gd)
    states = [rb.sample(cap, t, templates) for t, _, _ in rows]

    # color->side from score-delta votes at icon boundaries (reuse reconcile's idea)
    votes = {"white": {"left": 0, "right": 0}, "yellow": {"left": 0, "right": 0}}
    prev_icon, seg_start = None, 0
    marks = [k for k in range(len(states)) if states[k][3]]
    for a, b in zip(marks, marks[1:]):
        c0, c1 = states[a][3], states[b][3]
        if c0 != c1 and None not in (states[a][0], states[a][2], states[b][0], states[b][2]):
            if states[b][0] > states[a][0] and states[b][2] == states[a][2]:
                votes[c0]["left"] += 1
            elif states[b][2] > states[a][2] and states[b][0] == states[a][0]:
                votes[c0]["right"] += 1
    side_of = {}
    for c, v in votes.items():
        if v["left"] != v["right"]:
            side_of[c] = "left" if v["left"] > v["right"] else "right"
    if len(side_of) == 1:
        (c0, s0), = side_of.items()
        side_of["yellow" if c0 == "white" else "white"] = "right" if s0 == "left" else "left"

    # post-game board: sample ~25 s after the last shot for the final anchor
    final_score = None
    if rows:
        cap.set(cv2.CAP_PROP_POS_MSEC, (rows[-1][0] + 25.0) * 1000)
        ok, f = cap.read()
        if ok:
            fs = sb.banner_state(f, templates)
            final_score = (fs[0], fs[2])
    moved = [objects_moved(path) for _, _, path in rows]
    actions, total = solve(states, side_of, final_score=final_score, moved=moved)

    # agreement report — re-simulate the chosen path's state exactly
    agree = {"icon_ok": 0, "icon_n": 0, "run_ok": 0, "run_n": 0}
    rr = 0
    for k, (act, cue) in enumerate(actions):
        icon, brun = states[k][3], states[k][4]
        if icon is not None:
            agree["icon_n"] += 1
            agree["icon_ok"] += int(icon == cue)
        if brun is not None:
            agree["run_n"] += 1
            agree["run_ok"] += int(brun == rr)
        if act == "make":
            rr += 1
        elif act == "miss":
            rr = 0

    name = gd.rstrip("/").rsplit("/", 1)[-1]
    from collections import Counter
    print(f"{name}: {len(rows)} shots · cost {total:.1f} · {dict(Counter(a for a, _ in actions))} · "
          f"icon {agree['icon_ok']}/{agree['icon_n']} · run {agree['run_ok']}/{agree['run_n']} · side_of {side_of}")

    if fix:
        doc = {"color_side": {c: ("L" if s == "left" else "R") for c, s in side_of.items()}, "shots": []}
        for (t0, name_, path), (act, cue) in zip(rows, actions):
            doc["shots"].append({"shot": name_, "player": side_of.get(cue), "cue": cue,
                                 "result": act if act != "spurious" else "spurious", "t": t0})
            txt = [l for l in open(path).read().splitlines()
                   if not (l.startswith("result ") or l.startswith("player "))]
            txt.insert(1, f"player {side_of.get(cue, '')}")
            txt.insert(1, f"result {act}")
            open(path, "w").write("\n".join(txt) + "\n")
        json.dump(doc, open(os.path.join(gd, "results.json"), "w"), indent=1)
        print(f"  wrote results.json + headers")
    return actions


def main():
    args = [a for a in sys.argv[1:] if a != "--fix"]
    fix = "--fix" in sys.argv
    video, dirs = args[0], args[1:]
    templates = sb.load_templates()
    cap = cv2.VideoCapture(video)
    for gd in dirs:
        run_game(cap, gd, templates, fix)
    cap.release()


if __name__ == "__main__":
    main()
