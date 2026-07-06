#!/usr/bin/env python3
"""Segment a recorded MASA 4 match into individual shots, using the shot clock.

Signals (one pass over the video):
  - **shot clock** (green depleting ring, `scoreboard.clock`): resets once per
    shot, so a reset marks a turn boundary — no OCR of the number needed.
  - **inset ball motion** (per-frame differencing in the overhead inset): the
    stroke is the biggest motion burst inside each clock window. We bound it
    *rest-to-rest* — from the still frame just before the stroke to when the
    balls settle — so t=0 is the true starting layout and nothing is clipped.
  - **game start** (`scoreboard.scores_zero`): both score boxes read 0.

Outputs a shot manifest (CSV) and a contact-sheet montage. With --extract it
dumps each shot's inset frames (a bit wider than rest-to-rest; `track.py` then
trims to the exact ball-rest span).

    python3 detect_shots.py ../data/masa4_live/masa4_match_XXXX.mp4 \
        --manifest ../data/masa4_shots.csv --montage ../data/masa4_shots.png
"""

import argparse
import os
import shutil

import cv2
import numpy as np

import scoreboard as sb

INSET = (15, 270, 185, 333)              # inset crop x, y, w, h (full frame)
INTERIOR = (25, 190, 285, 595)           # x0, x1, y0, y1 for motion (skip the rail)
HERE = os.path.dirname(os.path.abspath(__file__))

# Rest-to-rest hysteresis on the per-frame inset motion (pixel-change count).
MOTION_HI = 55      # a clear stroke / rolling balls
MOTION_LO = 10      # near-still (raw frame diff floor)
REST_S = 0.5        # motion must stay low this long to count as "settled"
PAD_PRE, PAD_POST = 0.5, 0.6   # extra frames kept around the rest-to-rest span


def analyze(path, hz=5.0, t0=0.0, t1=float("inf")):
    """One pass: per-frame inset motion (30 Hz) + the shot-clock/score series
    (sub-sampled at `hz`). Returns fps, sampled (t, frac, act, motion, zero) and
    the full-rate `fine` motion array indexed by frame.

    `t0`/`t1` restrict the scan to one span of a long recording (one game of a
    broadcast day) — the caller must treat all returned times/frames as relative
    to `t0` (build_match offsets them back to video time)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if t0 > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    step = max(1, int(round(fps / hz)))
    ztp = os.path.join(HERE, "zero_template.npy")
    zt = np.load(ztp) if os.path.exists(ztp) else None
    x0, x1, y0, y1 = INTERIOR
    t, frac, act, motion, zero, fine = [], [], [], [], [], []
    prev, i = None, 0
    while i / fps <= t1 - t0:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
        fine.append(0 if prev is None else int(np.count_nonzero(np.abs(g - prev) > 28)))
        prev = g
        if i % step == 0:
            a, fr = sb.clock(f)
            t.append(i / fps); frac.append(fr); act.append(a); motion.append(fine[-1])
            zero.append(bool(zt is not None and sb.scores_zero(f, zt)))
        i += 1
    cap.release()
    A = lambda v: np.array(v)
    return fps, A(t), A(frac), A(act), A(motion), A(zero), A(fine)


def _stroke_span(fine, fps, a, b):
    """Rest-to-rest span (onset, settle) of the main stroke inside frames [a, b),
    from the per-frame motion `fine`. Onset backs up to the last still frame;
    settle waits for REST_S of near-stillness (or the window end)."""
    seg = fine[a:min(b, len(fine))]
    if len(seg) == 0:
        return None
    rest = int(REST_S * fps)
    # split into motion episodes separated by >= `rest` near-still frames
    eps, i = [], 0
    while i < len(seg):
        if seg[i] > MOTION_HI:
            j, gap = i, 0
            while j < len(seg) and gap < rest:
                gap = gap + 1 if seg[j] <= MOTION_LO else 0
                j += 1
            eps.append((i, j - gap))
            i = j
        else:
            i += 1
    if not eps:
        return None
    s, e = max(eps, key=lambda p: seg[p[0]:p[1]].sum())  # the biggest-travel burst
    o = s
    while o > 0 and seg[o] > MOTION_LO:
        o -= 1
    st, run = e, 0
    while st < len(seg) and run < rest:
        run = run + 1 if seg[st] <= MOTION_LO else 0
        st += 1
    return (a + o, a + st)


def find_shots(fps, t, frac, fine):
    """Shot windows from clock resets, each bounded rest-to-rest by `_stroke_span`.
    The settle search runs a little past the next reset, because a slow shot's
    balls can still be rolling when the opponent's clock starts."""
    resets = [int(t[k] * fps) for k in range(1, len(frac))
              if frac[k] - frac[k - 1] > 0.25 and frac[k] > 0.5]
    end = len(fine)
    spill = int(1.5 * fps)  # allow the settle to run a little past the next reset
    shots = []
    for a, b in zip(resets, resets[1:] + [end]):
        span = _stroke_span(fine, fps, a, min(end, b + spill))
        if span is None:
            continue
        onset, settle = span
        stroke = onset + int(np.argmax(fine[onset:max(onset + 1, settle)]))
        shots.append(dict(
            win_start=a / fps, win_end=b / fps,
            onset=onset / fps, settle=settle / fps, stroke_t=stroke / fps,
            peak_motion=int(fine[onset:max(onset + 1, settle)].max()),
        ))
    return shots


def montage(path, shots, out, fps):
    cap = cv2.VideoCapture(path)
    x, y, w, h = INSET
    thumbs = []
    for k, s in enumerate(shots):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s["stroke_t"] * fps))
        ok, f = cap.read()
        if not ok:
            continue
        im = cv2.resize(f[y:y + h, x:x + w], (120, 216), interpolation=cv2.INTER_AREA)
        cv2.putText(im, str(k), (4, 20), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"{s['stroke_t']:.0f}s", (4, 210), 0, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
        thumbs.append(im)
    cap.release()
    per = 10
    rows = []
    for r in range(0, len(thumbs), per):
        row = thumbs[r:r + per]
        while len(row) < per:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(out, np.vstack(rows))


def extract_inset(path, shots, out_dir, fps):
    """Dump each shot's inset frames over [onset-PAD_PRE, settle+PAD_POST] — a bit
    wider than rest-to-rest so `track.py` can trim to the exact ball-rest span."""
    x, y, w, h = INSET
    cap = cv2.VideoCapture(path)
    for k, s in enumerate(shots):
        d = os.path.join(out_dir, f"shot_{k:02d}")
        shutil.rmtree(d, ignore_errors=True)  # clear stale frames from a prior run
        os.makedirs(d)
        a = int(max(0, (s["onset"] - PAD_PRE) * fps))
        # generous end (a three-cushion roll can last ~10 s and outlast the clock
        # reset) so nothing is clipped; track.py trims to the true ball-rest span.
        b = int(max(s["settle"] + PAD_POST, s["onset"] + 11.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for j in range(b - a):
            ok, f = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(d, f"f_{j:04d}.png"), f[y:y + h, x:x + w])
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--manifest", default="shots.csv")
    ap.add_argument("--montage", default="shots.png")
    ap.add_argument("--extract", help="dir to dump per-shot inset frames")
    args = ap.parse_args()

    fps, t, frac, act, motion, zero, fine = analyze(args.video)
    shots = find_shots(fps, t, frac, fine)
    starts = [float(t[k]) for k in range(len(zero)) if zero[k]]
    print(f"{len(shots)} shots over {t[-1]:.0f}s; clock active {act.mean()*100:.0f}% of the time")
    print("game-start (0-0) at ~%.0fs" % starts[0] if starts else "no game-start (0-0) — began mid-game")

    with open(args.manifest, "w") as f:
        f.write("shot,onset_s,stroke_s,settle_s,dur_s,peak_motion\n")
        for k, s in enumerate(shots):
            f.write(f"{k},{s['onset']:.1f},{s['stroke_t']:.1f},{s['settle']:.1f},"
                    f"{s['settle']-s['onset']:.1f},{s['peak_motion']}\n")
    print(f"wrote {args.manifest}")
    montage(args.video, shots, args.montage, fps)
    print(f"wrote {args.montage}")
    if args.extract:
        extract_inset(args.video, shots, args.extract, fps)
        print(f"extracted inset frames per shot -> {args.extract}/shot_NN/")


if __name__ == "__main__":
    main()
