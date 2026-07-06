# Snapshot testing

Large-scale regression gates for the perception→reconstruction pipeline: the
**approved** outputs for the whole tracked corpus live here, and any change to
the fit, engine, extraction, tracker, or detector is judged against them.
Both pipelines are deterministic (fixed seeds, ordered parallelism, one ONNX
graph), so `check` diffs exactly; a diff is a *decision*, not noise.

## Reconstruction snapshots — `recon/*.tsv`

One line per shot across every game dir (~520 shots): fitted action
(aim/speed/english/follow), cue & object RMS, event penalty, and the compact
simulated event story (`x02@0.44` = balls 0⇄2 collide, `r0+y@1.13` = ball 0
bounces off the +y rail).

```bash
# judge current code against the approved corpus (exits 1 on any change):
cargo run -p billiards-solver --example snapshot_recon --release -- \
    check data/masa4_day/game_00 data/masa4_day/game_01 data/masa4_day/game_02 \
          data/masa4_day/game_03 data/masa4_day/game_04 data/masa4/game_00

# after reviewing a change report (improved vs worse counts), approve:
cargo run -p billiards-solver --example snapshot_recon --release -- bless <dirs…>
```

The check report lists every changed shot with its `cue_rms old -> new` so
regressions are visible individually, not just in aggregate.

## Track snapshots — `tracks/*.rows`

Full tracker output (every `color,t,x,y` row + segments) for the curated
sample in `tracks/manifest.txt` — one or two shots per game plus known-tricky
cases (fast strokes, white/yellow swaps, corner balls). Inference costs ~45 s
per shot, which is why this layer samples instead of sweeping.

```bash
ORT_DYLIB_PATH=…/site-packages/onnxruntime/capi/libonnxruntime.<ver>.dylib \
cargo run -p billiards-vision --features onnx --example snapshot_tracks --release -- check   # or bless
```

## Rules of engagement

- **Never bless blind.** A `CHANGED` report is the tool doing its job: read
  the per-shot deltas, decide, then bless. Blessing without reading converts
  the gate into noise.
- Snapshots depend on the local `data/` corpus (gitignored, real footage) and
  the per-game `calibration.json` files. Changing a calibration changes its
  game's recon snapshot — re-bless that game together with the calibration.
- New games: add their dir to the check/bless invocations (recon) and a line
  or two to `tracks/manifest.txt` (tracks), then bless once.
