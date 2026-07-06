#!/usr/bin/env python3
"""Fine-tune a COCO-pretrained detector for three-cushion ball detection.

The winning combination (see docs/DESIGN.md): a detector whose backbone was
pretrained on real photos has no sim-to-real gap (it already recognized billiard
balls as "sports ball" with zero training), and fine-tuning its head on our
domain-randomized synthetic data teaches it all three ball colors at billiard
sizes. We use the lightweight mobilenet-v3-320 Faster R-CNN so it trains on CPU.

    python3 finetune_detector.py --real FRAME.png
"""

import argparse
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights as W,
    fasterrcnn_mobilenet_v3_large_fpn as build,  # 800px input: resolves tiny inset balls
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from synth_data import BALL_CLASSES, render_sample

CLS_ID = {"white": 1, "yellow": 2, "red": 3}
ID_CLS = {v: k for k, v in CLS_ID.items()}
DRAW = {"white": (240, 240, 240), "yellow": (0, 210, 240), "red": (40, 40, 220)}


def make_sample(rng, w=384, h=256):
    while True:
        # Wide ball-size range: small (overhead inset) through medium (main cam).
        img, labels = render_sample(rng, w=w, h=h, base_r_range=(2.0, 9.0))
        boxes, ids = [], []
        for cls, x, y, vis, r in labels:
            if vis:
                boxes.append([x - r - 1, y - r - 1, x + r + 1, y + r + 1])
                ids.append(CLS_ID[cls])
        if boxes:  # need at least one visible ball
            t = torch.tensor(np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                      "labels": torch.tensor(ids, dtype=torch.int64)}
            return t, target


def build_model():
    model = build(weights=W.DEFAULT)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, len(CLS_ID) + 1)  # + background
    return model


def train(model, rng, n=400, epochs=4, bs=4, lr=0.005):
    print(f"rendering {n} synthetic samples...", flush=True)
    data = [make_sample(rng) for _ in range(n)]
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=5e-4)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total, t0 = 0.0, time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            imgs = [data[j][0] for j in idx]
            targs = [data[j][1] for j in idx]
            loss = sum(model(imgs, targs).values())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss {total / (n // bs):.3f}  ({time.time() - t0:.0f}s)", flush=True)


def eval_real(model, path, thr=0.3):
    model.eval()
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    t = torch.tensor(np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))).float() / 255
    with torch.no_grad():
        out = model([t])[0]
    best = {}
    for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
        c = ID_CLS.get(int(l))
        if c and s >= thr and (c not in best or s > best[c][0]):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best[c] = (float(s), int(cx), int(cy))
    print(f"\ndetections in {path}:", flush=True)
    vis = img.copy()
    for c in BALL_CLASSES:
        if c in best:
            s, x, y = best[c]
            print(f"  {c:6}: ({x},{y})  score {s:.2f}", flush=True)
            cv2.circle(vis, (x, y), 12, DRAW[c], 2)
            cv2.putText(vis, f"{c} {s:.2f}", (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW[c], 1, cv2.LINE_AA)
        else:
            print(f"  {c:6}: MISSED", flush=True)
    cv2.imwrite("finetuned_real.png", vis)
    print("  wrote finetuned_real.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="real frame to evaluate on after training")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build_model()
    print("fine-tuning fasterrcnn_mobilenet_v3_large_320_fpn (COCO-pretrained backbone)", flush=True)
    train(model, rng, n=args.n, epochs=args.epochs)
    torch.save(model.state_dict(), "finetuned_detector.pt")
    print("saved finetuned_detector.pt", flush=True)
    if args.real:
        eval_real(model, args.real)


if __name__ == "__main__":
    main()
