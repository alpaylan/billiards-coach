# Billiards Coach

A three-cushion (carom) billiards coaching engine: a validated physics simulator,
a shot solver, an interactive editor, and a real-video perception front-end that
turns tournament footage into reconstructed shots (force + spin).

Full design + status: [`docs/DESIGN.md`](docs/DESIGN.md).

## See it in 2 minutes

**The reconstruction, side by side with the real video** — actual footage +
tracking on the left, tracked-vs-reconstructed trajectory on the right:

```
open media/inset_reconstruction_compare.mp4     # top-down inset: reconstruction ALIGNS (73 mm)
open media/reconstruction_compare.mp4           # angled camera: reconstruction diverges (568 mm)
```

(Parallax-free top-down beats hi-res-but-angled — that contrast is the point.)

**The interactive editor** — drag the balls, set the cue action (the strike
diagram shows where you're hitting), play the simulation, hit *Solve*:

```
cargo run -p billiards-ui --release
```

**Import a real configuration from a video** — reconstruct a frame into a scene,
load it into the editor, and **toggle Actual ↔ Reconstructed** to confirm the
reconstruction matches the real frame (the *Actual* view shows the video frame
with the reconstructed positions ringed on top):

```
cargo run -p billiards-ui --release -- media/example.scene    # ready-made example (real World Cup frame)

# or make your own from any frame:
python3 python/reconstruct_frame.py FRAME.png --corners "x,y x,y x,y x,y" --scene-out shot.scene
cargo run -p billiards-ui --release -- shot.scene             # or the "📂 Import scene from video…" button
```

The `.scene` format is plain text (`image`, `corners`, `orient`, then `color x y`
ball lines in table meters) — easy to hand-edit or generate from the tracker.

**Import a whole shot — with the real video beside it** — load a tracked shot;
the editor reconstructs the cue action (force + spin, shown in the panel), plays
the *actual* tracked motion against the *reconstructed* simulation (filled ball =
reconstructed, ring = actual), **and plays the real broadcast clip in a panel on
the right**, synced to the same playhead, with the reconstructed positions ringed
on each frame so you can watch the model track reality:

```bash
cargo run -p billiards-ui --release -- data/masa4.shot   # a real live-match stroke + its clip
```

A shot exported by `track.py` carries a back-link to its source frames (`frames`,
`fps`, `start`, `corners`), so the editor finds the actual footage on its own —
the right-hand panel shows the *literal* video, not a reconstruction.

**Browse a whole match** — point the editor at a *directory* of `.shot` files and
it loads the entire tracked game: step through shots (◀ ▶ / list), each with its
real inset clip beside the reconstruction:

```bash
cargo run -p billiards-ui --release -- data/masa4_match     # browse every shot in the match
```

**Shot repair (coaching)** — set up (or load) a shot and hit **🔧 Repair this
shot**: it searches near your aim/speed/spin for the *smallest* change that turns
the miss into a score, and tells you the adjustment ("hit 0.4 m/s softer, more
follow"). If nothing nearby scores, it says so — that was the wrong shot, not a
mis-hit. Powered by `billiards_solver::repair`.

**From a live match to a browsable game, end to end.** The MASA 4 /
bilardo.com.tr feed is a live tournament with a top-down overhead inset (the
parallax-free source) and a **green shot-clock ring** that resets once per shot —
so a match can be auto-segmented into its individual shots without OCR:

```bash
cd python
# 1. record a window of the live match and keep it on disk (data/ is gitignored)
python3 capture_live.py "https://www.youtube.com/watch?v=X-CnEnG5hB4" --secs 720
# 2. one command: shot-clock segmentation -> per-shot inset -> learned-detector tracking
python3 build_match.py ../data/masa4_live/masa4_match_*.mp4 --name masa4_match
cargo run -p billiards-ui --release -- data/masa4_match      # browse the tracked game
```

`detect_shots.py` alone writes a shot manifest + a contact-sheet montage
(`scoreboard.py` reads the score boxes and the shot clock). To track a single
inset clip instead, use `track.py --detector learned` as before.

**Calibrate the physics to a table.** The engine's cushion/cloth parameters can
be fit to real tracked shots so the reconstructions match that table:

```bash
cargo run -p billiards-solver --example calibrate_shots --release -- data/masa4_match/*.shot
```

This recovers cushion restitution + rolling resistance from the observed
trajectories; the fitted preset is `PhysicsParams::carom_calibrated()`, which the
editor uses by default.

## Prerequisites

- **Rust** (edition 2024 — recent toolchain). Everything core builds with `cargo`.
- **Python 3** with `numpy opencv-python-headless torch torchvision yt-dlp Pillow`
  (`pip install -r python/requirements.txt`) — only for the video/perception side.

## Rust

```bash
cargo test                                            # 28 tests (physics, solver, fit, calibration)
cargo run -p billiards-ui --release                   # interactive editor
cargo run -p billiards-cli                            # headless: simulate a 3-ball shot, score it

# demos (each prints a result; --release recommended)
cargo run -p billiards-engine --example diamond_system     # bank physics vs the diamond system
cargo run -p billiards-solver --example solve_scene --release      # find the most forgiving scoring shot
cargo run -p billiards-solver --example reconstruct_hit --release  # recover force/spin from a trajectory
cargo run -p billiards-vision --example reconstruct_demo           # image -> table coords -> Scene (writes a PNG)
```

## Python — perception pipeline

No downloads needed (synthetic, validated against ground truth):

```bash
cd python
python3 synth_data.py                 # domain-randomized training frames -> synth_samples.png
python3 track.py --synthetic          # ball tracking + shot segmentation, validated (~2 mm)
```

Real footage (needs `yt-dlp`; the overhead **inset** is the good source):

```bash
cd python
python3 ingest.py <youtube-url> --at 2500 --out frame.png      # pull a frame
python3 eval_frames.py frame.png                               # run the learned ball detector
# detect -> track -> segment -> export a shot, then reconstruct + render the side-by-side:
python3 track.py --frames FRAMES_DIR --corners "x,y x,y x,y x,y" --orient vertical \
    --detector learned --fit-out shot.csv
cargo run -p billiards-solver --example compare_dump --release -- shot.csv recon.csv
python3 compare_video.py FRAMES_DIR shot.csv recon.csv --corners "..." --orient vertical \
    --start-frame N --out compare.mp4
```

## Layout

```text
crates/
  billiards-core      domain model: table, balls, trajectory, scene, three-cushion scoring
  billiards-engine    event-based physics (Han-2005 cushions, collisions+throw, scheduler)
  billiards-solver    shot solver, hit reconstruction (fit), physics calibration (system-ID)
  billiards-vision    homography calibration, ball detection interface, scene reconstruction
  billiards-ui        egui interactive editor + cue-tip strike diagram
  billiards-cli       headless demo runner
python/               CV/ML: ingest, synthetic data, learned detector, tracking, comparison video
docs/DESIGN.md        architecture, decisions, phase-by-phase status
media/                generated comparison videos (gitignored)
```
