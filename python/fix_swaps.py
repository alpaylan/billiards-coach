#!/usr/bin/env python3
"""Repair white↔yellow identity swaps in already-tracked `.shot` files.

The tracker (before the continuity-arbitration fix in track.py) could flip the
white and yellow labels on a low-confidence frame: each tracker jumps squarely
onto the *other* ball, rides it for a few frames, and flips back — the visible
"zigzag" at the end of a shot. Two artifacts result:

  * swapped stretches — real detections carrying the wrong label; fixed by
    relabeling (honest: positions untouched, identity corrected by continuity);
  * midpoint spikes — the ≤5-frame gap interpolation bridging a swap frame put a
    sample halfway between the two balls, where no ball ever was; those samples
    are fabricated and get dropped.

    python3 fix_swaps.py ../data/<match>/*.shot
"""

import glob
import math
import sys


def _load(path):
    header, rows = [], {}
    order = []
    with open(path) as fh:
        for ln in fh:
            s = ln.rstrip("\n")
            p = s.split(",")
            if len(p) == 4 and p[0] in ("white", "yellow", "red"):
                if p[0] not in rows:
                    rows[p[0]] = []
                    order.append(p[0])
                rows[p[0]].append((float(p[1]), float(p[2]), float(p[3])))
            else:
                header.append(s)
    return header, order, rows


def _dist(a, b):
    return math.hypot(a[1] - b[1], a[2] - b[2])


def unswap(rows):
    """Relabel swapped white/yellow stretches in place; count corrections."""
    if "white" not in rows or "yellow" not in rows:
        return 0
    w = {round(t * 30): (t, x, y) for t, x, y in rows["white"]}
    y = {round(t * 30): (t, x, y) for t, x, y in rows["yellow"]}
    frames = sorted(set(w) | set(y))
    swapped, n = False, 0
    prev_w, prev_y = None, None
    new_w, new_y = [], []
    for f in frames:
        cw, cy = w.get(f), y.get(f)
        if swapped:
            cw, cy = cy, cw
        if cw and cy and _dist(cw, cy) < 0.03:
            # Both labels on one spot: two real balls can't overlap, so one
            # sample is a stolen blob and the other a gap-interpolated bridge of
            # the swap frame. Drop both — and crucially don't let them poison
            # `prev`, or the swap on the next frame becomes undetectable (the
            # midpoint is equidistant from both balls).
            continue
        if cw and cy and prev_w and prev_y:
            keep = _dist(cw, prev_w) + _dist(cy, prev_y)
            swap = _dist(cy, prev_w) + _dist(cw, prev_y)
            # The RATIO carries the evidence (under a swap, crossing the labels
            # restores continuity almost exactly); the absolute floor only has
            # to clear tracking jitter. It was 0.16 — which silently let swaps
            # through whenever the two balls were CLOSE when they exchanged
            # (combined jump ≈ their separation): a real 8 cm-apart exchange
            # measured keep=0.153 and survived. Two resting balls jitter ~1 cm
            # each, so 0.055 is still far above noise.
            if keep > 0.055 and swap < 0.35 * keep:
                swapped = not swapped
                cw, cy = cy, cw
                n += 1
        if cw:
            new_w.append(cw)
            prev_w = cw
        if cy:
            new_y.append(cy)
            prev_y = cy
    rows["white"], rows["yellow"] = new_w, new_y
    return n


def _series(rows, c):
    return {round(t * 30): (t, x, y) for t, x, y in rows.get(c, [])}


def _speed(series, f, n=6):
    """Mean speed (m/s) over the n frames before frame f."""
    pts = [series[g] for g in range(f - n, f + 1) if g in series]
    if len(pts) < 2:
        return None, None
    d = _dist(pts[-1], pts[0])
    dt = pts[-1][0] - pts[0][0]
    return (d / dt if dt > 0 else 0.0), pts[0]


def _dir(a, b):
    dx, dy = b[1] - a[1], b[2] - a[2]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def fix_contact_swaps(rows):
    """Resolve identity exchanges that happen DURING a ball-ball contact: the
    two balls nearly coincide for a few blurred frames and the detector can hand
    each tracker the other ball afterwards — smoothly, so jump tests can't see
    it. Physics can: the previously-STILL ball departs along the line of centers
    (from the incoming ball's contact position through the still ball's rest);
    the incoming ball deflects to the other side. Where the labels violate that,
    exchange them from the contact onward."""
    n_fixed = 0
    for a, b in (("white", "yellow"),):  # the visually confusable pair
        A, B = _series(rows, a), _series(rows, b)
        frames = sorted(set(A) & set(B))
        i = 0
        while i < len(frames):
            f = frames[i]
            if _dist(A[f], B[f]) > 0.075:
                i += 1
                continue
            # a contact: who was still, who was moving, just before it?
            va, _ = _speed(A, f - 2)
            vb, _ = _speed(B, f - 2)
            if va is None or vb is None or (va < 0.15) == (vb < 0.15):
                i += 1
                continue
            still, _mov = (a, b) if va < vb else (b, a)
            S, M = (A, B) if still == a else (B, A)
            # Everything pre-contact only: samples AT the closest frame are
            # blurred and possibly already label-swapped, so the line of centers
            # comes from the still ball's REST and the mover's last clean
            # position before the contact.
            rest_s = next((S[g] for g in range(f - 3, f - 12, -1) if g in S), None)
            m_pre = next((M[g] for g in range(f - 2, f - 8, -1) if g in M), None)
            f_out = next((g for g in frames[i:] if _dist(A[g], B[g]) > 0.09), None)
            if rest_s is None or m_pre is None or f_out is None:
                i += 1
                continue
            # A struck ball resting near a cushion banks immediately, so its
            # observed departure no longer points along the line of centers —
            # the test would be a coin flip. Leave those contacts alone (the
            # editor's report flow handles the rare mislabel with ground truth).
            if abs(rest_s[1]) > 1.386 - 0.10 or abs(rest_s[2]) > 0.673 - 0.10:
                i = frames.index(f_out) if f_out in frames else i + 1
                continue
            # No departure = no contact. Balls can pass within the contact
            # radius for a few blurred frames without touching; the still ball
            # then simply stays put, and the line-of-centers test would be
            # comparing jitter directions — a coin flip that can UNDO a swap
            # `unswap` just repaired. Only judge encounters where the still
            # ball verifiably left its rest.
            post_s = [S[g] for g in range(f_out, f_out + 20) if g in S]
            if not post_s or max(_dist(p, rest_s) for p in post_s) < 0.05:
                i = frames.index(f_out) if f_out in frames else i + 1
                continue
            l_hat = _dir(m_pre, rest_s)
            # Outgoing direction of each LABEL: chord over its own samples once
            # the contact has cleared (origin at its first post-contact sample).
            def out_dir(series):
                pts = [series[g] for g in range(f_out, f_out + 14) if g in series]
                return _dir(pts[0], pts[-1]) if len(pts) >= 2 else None
            out_s, out_m = out_dir(S), out_dir(M)
            if l_hat is None or out_s is None or out_m is None:
                i += 1
                continue
            # The struck (still) ball departs along the line of centers. If the
            # OTHER label satisfies that markedly better, the labels swapped.
            dot_keep = out_s[0] * l_hat[0] + out_s[1] * l_hat[1]
            dot_swap = out_m[0] * l_hat[0] + out_m[1] * l_hat[1]
            if dot_swap > dot_keep + 0.30:
                cut = f
                keep_a = [(t, x, y) for t, x, y in rows[a] if round(t * 30) < cut]
                keep_b = [(t, x, y) for t, x, y in rows[b] if round(t * 30) < cut]
                tail_a = [(t, x, y) for t, x, y in rows[a] if round(t * 30) >= cut]
                tail_b = [(t, x, y) for t, x, y in rows[b] if round(t * 30) >= cut]
                rows[a], rows[b] = keep_a + tail_b, keep_b + tail_a
                A, B = _series(rows, a), _series(rows, b)
                n_fixed += 1
            # skip past this contact either way
            i = frames.index(f_out) if f_out in frames else i + 1
    return n_fixed


def _entered_by_jump(series, f0):
    """Did this track JUMP into frame f0 (vs flowing there)? Compares the entry
    displacement from its last prior sample against plausible ball travel for
    that gap (30 cm/s of undetected drift, floor 8 cm)."""
    pv = max((g for g in series if g < f0), default=None)
    if pv is None:
        return False
    allowed = max(0.08, 0.01 * (f0 - pv))
    return _dist(series[f0], series[pv]) > allowed


def drop_steals(rows):
    """Drop stretches where one tracker rides ANOTHER ball: its samples coincide
    with a second color's concurrent samples (one blob, two labels) after a
    discontinuity in its own track. Those samples are false detections of the
    other ball — dropping them leaves an honest gap."""
    dropped = 0
    for a, b in (("red", "yellow"), ("red", "white"), ("white", "yellow")):
        for victim, host in ((a, b), (b, a)):
            V, H = _series(rows, victim), _series(rows, host)
            both = sorted(set(V) & set(H))
            bad = set()
            run = []
            for f in both:
                if _dist(V[f], H[f]) < 0.035:
                    run.append(f)
                else:
                    if len(run) >= 3 and _entered_by_jump(V, run[0]):
                        bad.update(run)
                    run = []
            if len(run) >= 3 and _entered_by_jump(V, run[0]):
                bad.update(run)
            if bad:
                rows[victim] = [(t, x, y) for t, x, y in rows[victim] if round(t * 30) not in bad]
                dropped += len(bad)
    return dropped


def drop_spikes(rows):
    """Remove isolated samples far from BOTH neighbours while the neighbours sit
    together — the fabricated midpoint of a bridged swap frame."""
    dropped = 0
    for c, tr in rows.items():
        keep = []
        for i, p in enumerate(tr):
            if 0 < i < len(tr) - 1:
                a, b = tr[i - 1], tr[i + 1]
                if (_dist(p, a) > 0.06 and _dist(p, b) > 0.06 and _dist(a, b) < 0.04
                        and p[0] - a[0] < 0.1 and b[0] - p[0] < 0.1):
                    dropped += 1
                    continue
            keep.append(p)
        rows[c] = keep
    return dropped


def fix(path):
    header, order, rows = _load(path)
    n_swap = unswap(rows)
    n_swap += fix_contact_swaps(rows)
    n_spike = drop_spikes(rows)
    n_spike += drop_steals(rows)
    if n_swap or n_spike:
        out = list(header)
        for c in order:
            for t, x, y in rows[c]:
                out.append(f"{c},{t:.4f},{x:.4f},{y:.4f}")
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
    return n_swap, n_spike


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(a)
    for p in sorted(paths):
        n_swap, n_spike = fix(p)
        if n_swap or n_spike:
            print(f"{p.rsplit('/', 1)[-1]:<16} unswapped {n_swap} flips, dropped {n_spike} midpoint spikes")


if __name__ == "__main__":
    main()
