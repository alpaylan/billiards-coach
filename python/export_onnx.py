#!/usr/bin/env python3
"""Export the fine-tuned ball detector to ONNX at a FIXED canonical input size,
then PROVE parity against the .pt model on real broadcast frames.

Why fixed-size: the torchscript tracer bakes input-size-dependent constants
into the graph (interpolation scales, anchor grids). A "dynamic axes" export
passes the checker but computes DIFFERENT activations at other resolutions —
measured here as fabricated/shifted detections on 332×184 frames — and the
dynamo/torch.export path cannot trace detection models' data-dependent control
flow at all. So the graph gets ONE input shape, the native overhead-inset frame
(3×515×290), and every consumer letterboxes into that canvas and maps boxes
back. `letterbox()` below is the reference implementation the Rust
`OnnxDetector` must mirror.

Parity gate: identical inputs through .pt (CPU) and .onnx (onnxruntime CPU)
must agree det-for-det — same labels, boxes within a pixel, scores within a few
thousandths — at the tracker's working threshold. Native frames are checked
as-is; odd-sized frames are checked through the letterbox path.

    python3 export_onnx.py            # export detector.onnx + run the gate
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch

from finetune_detector import build_model

THR = 0.04  # track.py's collection threshold — parity must hold at least here
CANON_H, CANON_W = 515, 290  # native overhead-inset frame (rows, cols)


def letterbox(bgr):
    """Fit an arbitrary frame into the canonical canvas (scale to fit, pad
    bottom/right with black). Returns (canvas, scale) — box coords in canvas
    space map back to source pixels as `(x / scale, y / scale)`. The Rust
    OnnxDetector must reproduce this exactly."""
    h, w = bgr.shape[:2]
    s = min(CANON_H / h, CANON_W / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((CANON_H, CANON_W, 3), dtype=bgr.dtype)
    canvas[:nh, :nw] = resized
    return canvas, s


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.tensor(np.transpose(rgb, (2, 0, 1))).float() / 255


def frame_paths():
    native = glob.glob("../data/masa4_day/game_0*_frames/shot_*/f_0030.png") + glob.glob(
        "../data/masa4/game_00_frames/shot_*/f_0030.png"
    )
    random.Random(7).shuffle(native)
    odd = sorted(glob.glob("../data/masa4_shot_frames/f_*.png"))[:8]
    return native[:20], odd


def dets(model, t):
    with torch.no_grad():
        out = model([t])[0]
    keep = out["scores"] >= THR
    return out["boxes"][keep].numpy(), out["labels"][keep].numpy(), out["scores"][keep].numpy()


def compare(tag, ref, onnx_out, worst, mismatches):
    rb, rl, rs = ref
    ob, ol, os_ = onnx_out
    k = os_ >= THR
    ob, ol, os_ = ob[k], ol[k], os_[k]
    if len(rb) != len(ob):
        mismatches.append(f"{tag}: {len(rb)} torch dets vs {len(ob)} onnx")
        return
    used = set()
    for i in range(len(rb)):
        best, bj = 1e9, None
        for j in range(len(ob)):
            if j not in used:
                d = float(np.abs(rb[i] - ob[j]).max())
                if d < best:
                    best, bj = d, j
        if bj is None or int(rl[i]) != int(ol[bj]):
            mismatches.append(f"{tag}: det {i} label/pair mismatch")
            continue
        used.add(bj)
        worst["box"] = max(worst["box"], best)
        worst["score"] = max(worst["score"], float(abs(rs[i] - os_[bj])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="finetuned_detector.pt")
    ap.add_argument("--out", default="detector.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # Trace with a REAL frame: random noise produces (near-)zero detections and
    # the tracer then bakes degenerate shapes through the ROI heads (observed:
    # a Reshape {73,15}->{-1,4} runtime failure). A frame with all three balls
    # exercises every branch with representative shapes.
    native_paths, _ = frame_paths()
    dummy_bgr = cv2.imread(native_paths[0])
    if dummy_bgr.shape[:2] != (CANON_H, CANON_W):
        dummy_bgr, _ = letterbox(dummy_bgr)
    dummy = to_tensor(dummy_bgr)
    print(f"exporting {args.weights} -> {args.out} (opset {args.opset}, fixed {CANON_W}x{CANON_H})", flush=True)
    torch.onnx.export(
        model,
        ([dummy],),
        args.out,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={"boxes": {0: "dets"}, "labels": {0: "dets"}, "scores": {0: "dets"}},
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(args.out)
    print(f"exported ok ({os.path.getsize(args.out) / 1e6:.0f} MB)")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    worst = {"box": 0.0, "score": 0.0}
    mismatches = []

    native, odd = frame_paths()
    for p in native:
        bgr = cv2.imread(p)
        if bgr.shape[:2] != (CANON_H, CANON_W):
            bgr, _ = letterbox(bgr)
        t = to_tensor(bgr)
        compare("/".join(p.split("/")[-3:]), dets(model, t), sess.run(None, {"image": t.numpy()}), worst, mismatches)
    for p in odd:
        canvas, _ = letterbox(cv2.imread(p))
        t = to_tensor(canvas)
        compare("LB:" + p.split("/")[-1], dets(model, t), sess.run(None, {"image": t.numpy()}), worst, mismatches)

    print(f"parity over {len(native)} native + {len(odd)} letterboxed frames: "
          f"worst box Δ {worst['box']:.3f} px · worst score Δ {worst['score']:.4f} · {len(mismatches)} mismatches")
    for m in mismatches[:8]:
        print("  MISMATCH", m)
    if mismatches or worst["box"] > 1.0 or worst["score"] > 0.01:
        raise SystemExit("PARITY GATE FAILED")
    print("PARITY GATE PASSED")


if __name__ == "__main__":
    main()
