#!/usr/bin/env python3
"""Reconstruct a three-cushion scene from a single broadcast frame.

Prototype for the Rust `billiards-vision` pipeline: calibrate a homography from
the four table corners, detect the balls by color within the table region, and
lift them to table coordinates (meters). Writes an annotated image for review
and prints the recovered scene.

The settled algorithm here is what the Rust side (and later an ONNX detector)
targets. Usage:

    python3 reconstruct_frame.py FRAME.png \
        --corners "287,172 988,162 990,560 285,567" \
        --out annotated.png
"""

import argparse
import os

import cv2
import numpy as np

# Regulation carom playing surface (meters), measured between cushion noses.
TABLE_L, TABLE_W = 2.84, 1.42
BALL_R = 0.03075

# Corner order: far-left, far-right, near-right, near-left (image top = far).
TABLE_CORNERS = np.float32([
    [-TABLE_L / 2,  TABLE_W / 2],
    [ TABLE_L / 2,  TABLE_W / 2],
    [ TABLE_L / 2, -TABLE_W / 2],
    [-TABLE_L / 2, -TABLE_W / 2],
])

# HSV ranges (OpenCV H in 0..179). Tuned for red/yellow/white balls on blue cloth.
# Balls are small, saturated color patches; the wood cushion is a desaturated
# tan that overlaps the yellow hue, so yellow needs a high saturation floor and
# we reject over-large blobs (see MAX_AREA).
COLOR_RANGES = {
    "red":    [((0, 120, 90), (10, 255, 255)), ((170, 120, 90), (179, 255, 255))],
    "yellow": [((20, 150, 140), (35, 255, 255))],
    "white":  [((0, 0, 175), (179, 55, 255))],
}
DRAW_BGR = {"red": (0, 0, 230), "yellow": (0, 210, 230), "white": (240, 240, 240)}
MIN_AREA, MAX_AREA = 20, 2500


def detect_ball(hsv, table_mask, ranges):
    """Centroid (u, v) of the largest ball-sized color blob inside the table."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.bitwise_and(mask, table_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    best, best_area = None, MIN_AREA
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA < area <= MAX_AREA and area > best_area:
            best, best_area = centroids[i], area
    return (float(best[0]), float(best[1]), int(best_area)) if best is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--corners", required=True,
                    help='4 image corners "x,y x,y x,y x,y" as far-L far-R near-R near-L')
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--scene-out", help="write ball table-coords as a .scene file for the editor")
    args = ap.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"could not read {args.frame}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    image_corners = np.float32([[float(a) for a in p.split(",")] for p in args.corners.split()])
    assert image_corners.shape == (4, 2), "need exactly 4 corners"

    # Homography image -> table (meters), and the table region mask.
    H = cv2.getPerspectiveTransform(image_corners, TABLE_CORNERS)
    table_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(table_mask, image_corners.astype(np.int32), 255)
    # Pull the mask in from the cushion so the tan wood nose isn't sampled.
    table_mask = cv2.erode(table_mask, np.ones((13, 13), np.uint8))

    def to_table(u, v):
        p = H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    annotated = img.copy()
    cv2.polylines(annotated, [image_corners.astype(np.int32)], True, (60, 220, 60), 2)

    print(f"reconstructing {args.frame}  ({img.shape[1]}x{img.shape[0]})")
    scene = {}
    for color, ranges in COLOR_RANGES.items():
        det = detect_ball(hsv, table_mask, ranges)
        if det is None:
            print(f"  {color:6}: not found")
            continue
        u, v, area = det
        x, y = to_table(u, v)
        scene[color] = (x, y)
        inside = abs(x) <= TABLE_L / 2 + 0.02 and abs(y) <= TABLE_W / 2 + 0.02
        print(f"  {color:6}: image ({u:6.1f},{v:6.1f}) area {area:4d} -> table ({x:+.3f},{y:+.3f}) m {'' if inside else '  ⚠ off table'}")
        cv2.circle(annotated, (int(u), int(v)), 12, DRAW_BGR[color], 2)
        cv2.putText(annotated, f"{color} ({x:+.2f},{y:+.2f})", (int(u) + 14, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_BGR[color], 1, cv2.LINE_AA)

    cv2.imwrite(args.out, annotated)
    print(f"  wrote {args.out}")

    if args.scene_out:
        with open(args.scene_out, "w") as f:
            f.write("# billiards scene (table coords, meters) — import into billiards-ui\n")
            # Source frame + calibration, so the editor can overlay the reconstruction
            # on the actual image for verification.
            f.write(f"image {os.path.abspath(args.frame)}\n")
            f.write("corners " + " ".join(f"{int(round(c[0]))},{int(round(c[1]))}" for c in image_corners) + "\n")
            f.write("orient horizontal\n")
            for c in ("white", "yellow", "red"):
                if c in scene:
                    f.write(f"{c} {scene[c][0]:.4f} {scene[c][1]:.4f}\n")
        print(f"  wrote scene {args.scene_out}")


if __name__ == "__main__":
    main()
