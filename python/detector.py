#!/usr/bin/env python3
"""A tiny learned ball detector for three-cushion.

A fully-convolutional network predicts one heatmap per ball color (white /
yellow / red); the peak of each heatmap is that ball's location. It's trained
entirely on domain-randomized synthetic data (`synth_data.py`) — so it learns
*ball appearance* and shrugs off the red rail objects, cue sticks, and blur that
defeated classical color thresholding. Small enough to train on CPU in minutes
and to export to ONNX for the Rust runtime later.

    python3 detector.py                 # train + evaluate on synthetic
    python3 detector.py --real F.png    # then localize balls in a real crop
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from synth_data import BALL_CLASSES, render_sample

IN_H, IN_W = 128, 192
OUT_H, OUT_W = IN_H // 4, IN_W // 4
DEVICE = "cpu"


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b, s):
            return nn.Sequential(nn.Conv2d(a, b, 3, s, 1), nn.BatchNorm2d(b), nn.ReLU())
        self.net = nn.Sequential(
            block(3, 16, 2),   # /2
            block(16, 32, 2),  # /4
            block(32, 32, 1),
            block(32, 32, 1),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def _gaussian(t, cls_idx, cx, cy, sigma=1.2):
    ys, xs = np.ogrid[:OUT_H, :OUT_W]
    t[cls_idx] = np.maximum(t[cls_idx], np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))


def make_pool(rng, n):
    imgs, targets = [], []
    for _ in range(n):
        img, labels = render_sample(rng, w=IN_W, h=IN_H)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(x, (2, 0, 1)))
        t = np.zeros((3, OUT_H, OUT_W), np.float32)
        for ci, (_, px, py, vis, _r) in enumerate(labels):
            if vis:
                _gaussian(t, ci, px / 4.0, py / 4.0)
        targets.append(t)
    return torch.tensor(np.array(imgs)), torch.tensor(np.array(targets))


def decode(pred, thr=0.3):
    """pred: (3, OUT_H, OUT_W) -> [(x, y) or None] in input-pixel coords."""
    out = []
    for c in range(3):
        hm = pred[c]
        idx = int(hm.argmax())
        y, x = divmod(idx, OUT_W)
        out.append((x * 4.0, y * 4.0) if hm[y, x] > thr else None)
    return out


def train(model, rng, pool_size=1200, epochs=12, bs=16, lr=1e-3):
    imgs, targets = make_pool(rng, pool_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(imgs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            pred = model(imgs[b])
            w = 1.0 + 20.0 * targets[b]  # emphasize ball regions (sparse positives)
            loss = (w * (pred - targets[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss {total / n:.5f}")


def evaluate(model, rng, n=300, tol=6.0):
    model.eval()
    hit = {c: [0, 0] for c in BALL_CLASSES}  # [detected-correct, visible-total]
    errs, false_pos = [], 0
    with torch.no_grad():
        for _ in range(n):
            img, labels = render_sample(rng, w=IN_W, h=IN_H)
            x = torch.tensor(np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1)))[None]
            det = decode(model(x)[0].numpy())
            for ci, (cls, px, py, vis, _r) in enumerate(labels):
                d = det[ci]
                if vis:
                    hit[cls][1] += 1
                    if d and np.hypot(d[0] - px, d[1] - py) <= tol:
                        hit[cls][0] += 1
                        errs.append(np.hypot(d[0] - px, d[1] - py))
                elif d is not None:
                    false_pos += 1
    print("evaluation on fresh synthetic:")
    for c in BALL_CLASSES:
        ok, tot = hit[c]
        print(f"  {c:6}: {ok}/{tot} localized ({100 * ok / max(1, tot):.0f}%)")
    print(f"  mean localization error: {np.mean(errs):.1f}px (of {IN_W}px width)")
    print(f"  false positives on occluded balls: {false_pos}")


def run_real(model, path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    resized = cv2.resize(img, (IN_W, IN_H))
    x = torch.tensor(np.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1)))[None]
    model.eval()
    with torch.no_grad():
        det = decode(model(x)[0].numpy())
    sx, sy = img.shape[1] / IN_W, img.shape[0] / IN_H
    draw = {"white": (240, 240, 240), "yellow": (0, 210, 240), "red": (40, 40, 220)}
    vis = img.copy()
    print(f"detections in {path}:")
    for cls, d in zip(BALL_CLASSES, det):
        if d:
            p = (int(d[0] * sx), int(d[1] * sy))
            print(f"  {cls:6}: ({p[0]},{p[1]})")
            cv2.circle(vis, p, 10, draw[cls], 2)
            cv2.putText(vis, cls, (p[0] + 12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw[cls], 1, cv2.LINE_AA)
        else:
            print(f"  {cls:6}: not found")
    cv2.imwrite("detector_real.png", vis)
    print("  wrote detector_real.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="real image crop to run the trained detector on")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    model = Detector().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"detector: {params/1000:.1f}k params, input {IN_W}x{IN_H}")

    if os.path.exists("detector.pt") and not args.retrain:
        model.load_state_dict(torch.load("detector.pt"))
        print("loaded detector.pt")
    else:
        print("training on synthetic:")
        train(model, rng, epochs=args.epochs)
        evaluate(model, np.random.default_rng(99))
        torch.save(model.state_dict(), "detector.pt")
        print("saved detector.pt")

    if args.real:
        run_real(model, args.real)


if __name__ == "__main__":
    main()
