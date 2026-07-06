#!/usr/bin/env python3
"""Evaluate the fine-tuned detector on several real frames.

    python3 eval_frames.py FRAME1.png FRAME2.png ...
Writes <name>.det.png per frame and prints per-ball detections.
"""

import sys

import cv2
import numpy as np
import torch

from finetune_detector import BALL_CLASSES, DRAW, ID_CLS, build_model


def detect(model, img, thr=0.3):
    t = torch.tensor(np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255
    with torch.no_grad():
        out = model([t])[0]
    best = {}
    for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
        c = ID_CLS.get(int(l))
        if c and s >= thr and (c not in best or s > best[c][0]):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best[c] = (float(s), int(cx), int(cy))
    return best


def main():
    model = build_model()
    model.load_state_dict(torch.load("finetuned_detector.pt"))
    model.eval()
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read"); continue
        best = detect(model, img)
        print(f"{path} ({img.shape[1]}x{img.shape[0]}):")
        vis = img.copy()
        for c in BALL_CLASSES:
            if c in best:
                s, x, y = best[c]
                print(f"  {c:6}: ({x},{y}) score {s:.2f}")
                cv2.circle(vis, (x, y), 12, DRAW[c], 2)
                cv2.putText(vis, f"{c} {s:.2f}", (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW[c], 1, cv2.LINE_AA)
            else:
                print(f"  {c:6}: MISSED")
        out = path.rsplit(".", 1)[0] + ".det.png"
        cv2.imwrite(out, vis)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
