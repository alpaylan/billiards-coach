#!/usr/bin/env python3
"""Read the MASA 4 / bilardo scoreboard + shot clock from a broadcast frame.

The bottom-center overlay (1280x720 frame) reads:
    LEFT_NAME [Lscore]  INNINGS[inning]  [Rscore]  RIGHT_NAME
with a **shot clock** — a green depleting ring with a countdown number — that
appears ABOVE the innings box only while a shot is being timed (real match play,
not warm-up). The green ring's arc length is a direct proxy for the clock value,
so we can find shot boundaries (clock resets to full) without reading the number.

We use this to (1) find a game start (scores 0-0) and (2) segment a game into
individual shots (one shot ~ one clock cycle: appears -> counts down -> resets).
"""

import os
import subprocess

import cv2
import numpy as np

# Fixed overlay geometry (measured on the 1280x720 MASA 4 stream, X-CnEnG5hB4).
SCORE_L = (542, 621, 55, 39)   # x, y, w, h — white box, black digit
INNINGS = (599, 638, 80, 36)  # the DIGIT row; the label sits above at y~621
SCORE_R = (681, 621, 54, 39)
CLOCK_RING = (606, 535, 74, 62)  # green ring bbox above the innings box
GREEN = ((38, 70, 70), (88, 255, 255))  # HSV range for the clock ring
# Player name plates (blue, white text) flanking the scores. The left name is
# right-aligned against SCORE_L and the right name left-aligned against SCORE_R,
# so the crops are generous on the variable side to fit long names.
NAME_L = (170, 622, 372, 36)
NAME_R = (750, 622, 372, 36)

# The coords above are for the 720p base; the broadcast overlay scales linearly
# with frame height, so every box scales by `frame_height / 720` (verified: 1080p
# is an exact 1.5x). Pixel counts (green ring) scale with the area, i.e. by s².
BASE_H = 720
FULL_RING_PX = 1500.0  # green px of a full ring at 720p


def _scale(box, s):
    return tuple(int(round(v * s)) for v in box)


def clock(frame):
    """(active, fraction) of the shot clock. `active` = the green ring is shown;
    `fraction` ~ remaining time (1.0 full / ~40s, 0.0 empty), from arc length.
    Resolution-independent — the overlay geometry scales with the frame height."""
    s = frame.shape[0] / BASE_H
    x, y, w, h = _scale(CLOCK_RING, s)
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    px = int(cv2.countNonZero(cv2.inRange(hsv, np.array(GREEN[0]), np.array(GREEN[1]))))
    return px > 120 * s * s, min(px / (FULL_RING_PX * s * s), 1.0)


def _box(frame, box):
    x, y, w, h = _scale(box, frame.shape[0] / BASE_H)
    return frame[y:y + h, x:x + w]


def scores_zero(frame, zero_template):
    """True if both score boxes read '0' (a fresh game). Uses a single '0'
    template so we don't need a full digit classifier just to spot game start."""
    return (_match_digit(_box(frame, SCORE_L), zero_template) and
            _match_digit(_box(frame, SCORE_R), zero_template))


def _normalize_digit(crop):
    """A score/innings box -> a clean 24x32 binary bitmap of its digit, centered
    with its aspect ratio preserved (so the same glyph normalizes consistently
    regardless of which box it came from)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # score boxes are black-on-white; innings/clock are white-on-black. Make the
    # digit dark on a light field either way, then Otsu.
    if g.mean() < 110:
        g = 255 - g
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # digit=255
    # keep only the largest blob (the digit), dropping stray specks / box edges
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 12:
        return None
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    d = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    # fit into a 24x32 canvas preserving aspect ratio, centered
    W, H = 24, 32
    s = min((W - 4) / w, (H - 4) / h)
    dw, dh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    d = cv2.resize(d, (dw, dh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    ox, oy = (W - dw) // 2, (H - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = d
    return (canvas > 127).astype(np.uint8) * 255


def _match_digit(crop, template):
    n = _normalize_digit(crop)
    if n is None or template is None:
        return False
    return float((n == template).mean()) > 0.85  # >85% pixels agree


def _ocr_line(crop):
    """OCR a single line of the white-on-blue name plate. Pipes the image to
    tesseract over stdin (no temp files), returns an UPPERCASE string. The Turkish
    diacritics don't survive (İ->I, Ş->S, and E sometimes ->F), but the result is
    *stable* for a given plate, so it works as a matchup identity key."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY_INV)  # bright text -> black on white
    b = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    b = cv2.copyMakeBorder(b, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)
    ok, png = cv2.imencode(".png", b)
    if not ok:
        return ""
    try:
        r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7"],
                           input=png.tobytes(), capture_output=True, timeout=15)
        return " ".join(r.stdout.decode("utf-8", "ignore").split()).upper()
    except Exception:
        return ""


def names(frame):
    """(left_player, right_player) read off the scoreboard name plates."""
    return _ocr_line(_box(frame, NAME_L)), _ocr_line(_box(frame, NAME_R))


def name_crops(frame):
    """(left, right) name-plate image crops — the reliable thing to *show* a user
    for picking a game, since OCR mangles the Turkish characters."""
    return _box(frame, NAME_L).copy(), _box(frame, NAME_R).copy()


if __name__ == "__main__":
    import sys, glob
    for p in sys.argv[1:] or sorted(glob.glob("/tmp/fullframes/*.png")):
        f = cv2.imread(p)
        if f is None:
            continue
        active, frac = clock(f)
        print(f"{p}: clock active={active} fraction={frac:.2f}")


# --- full banner reading: score/innings digits + the run indicator ----------
#
# The banner also carries (measured on the 1080p stream, /1.5 to the 720p base):
#   * a BALL ICON at the banner's outer edge showing whose turn it is —
#     white ball far-LEFT for white's turn, yellow ball far-RIGHT for yellow's;
#   * next to it, the ONGOING RUN counter (consecutive points this turn).
# Together with the score digits and the innings box these give an independent
# ground truth to verify shot segmentation + make/miss annotation against.

RUN_L_BALL = (12, 630, 32, 34)
RUN_L_NUM = (42, 622, 58, 46)
RUN_R_NUM = (1162, 622, 58, 46)
RUN_R_BALL = (1223, 630, 32, 35)


def _glyphs(crop):
    """A digit box -> per-digit 24x32 bitmaps, left to right (multi-digit)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if g.mean() < 110:
        g = 255 - g
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    h_box = crop.shape[0]
    comps = []
    for i in range(1, n):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if stats[i, cv2.CC_STAT_AREA] < 25 or h < 0.4 * h_box:
            continue  # specks / box edges — a digit is tall
        comps.append((x, y, w, h, i))
    comps.sort()
    out = []
    for x, y, w, h, i in comps:
        d = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        W, H = 24, 32
        s = min((W - 4) / w, (H - 4) / h)
        dw, dh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        d = cv2.resize(d, (dw, dh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, W), np.uint8)
        canvas[(H - dh) // 2:(H - dh) // 2 + dh, (W - dw) // 2:(W - dw) // 2 + dw] = d
        out.append((canvas > 127).astype(np.uint8))
    return out


def load_templates(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_templates.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {int(k): z[k] for k in z.files}  # digit -> (n, 32, 24) stacks


def _classify(bitmap, templates, min_agree=0.86):
    best, best_a = None, min_agree
    for d, stack in templates.items():
        a = float(np.max((stack == bitmap).mean(axis=(1, 2))))
        if a > best_a:
            best, best_a = d, a
    return best


def read_number(frame, box, templates):
    """The integer shown in `box`, or None if any glyph is unrecognized."""
    gl = _glyphs(_box(frame, box))
    if not gl:
        return None
    digits = [_classify(g, templates) for g in gl]
    if any(d is None for d in digits):
        return None
    return int("".join(str(d) for d in digits))


def banner_state(frame, templates):
    """Everything the banner says: (score_l, innings, score_r, turn, run).
    `turn` is 'white'|'yellow'|None from the outer ball icon; `run` is the
    ongoing-run counter next to it (None when unreadable/absent)."""
    sl = read_number(frame, SCORE_L, templates)
    inn = read_number(frame, INNINGS, templates)
    sr = read_number(frame, SCORE_R, templates)

    def ball(box, want):
        hsv = cv2.cvtColor(_box(frame, box), cv2.COLOR_BGR2HSV)
        if want == "yellow":
            m = cv2.inRange(hsv, np.array((18, 90, 120)), np.array((35, 255, 255)))
        else:  # white: bright, unsaturated
            m = cv2.inRange(hsv, np.array((0, 0, 170)), np.array((179, 70, 255)))
        s = frame.shape[0] / BASE_H
        return int(cv2.countNonZero(m)) > 220 * s * s

    turn, run = None, None
    if ball(RUN_L_BALL, "white"):
        turn, run = "white", read_number(frame, RUN_L_NUM, templates)
    elif ball(RUN_R_BALL, "yellow"):
        turn, run = "yellow", read_number(frame, RUN_R_NUM, templates)
    return sl, inn, sr, turn, run
