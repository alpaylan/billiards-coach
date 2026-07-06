# Billiards Coach — Design & Roadmap

A coaching application for **three-cushion (carom) billiards**, built to become a
real, usable product. Four headline capabilities:

1. Reconstruct 3D billiards scenes from 2D game video.
2. Manipulate a scene for "as-if" scenarios (remove/move a ball, re-simulate).
3. "Solve" a configuration — find the shot that scores, and how forgiving it is.
4. Classify scenarios and analyze a player's success rate per scenario type.

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Form factor | **Headless-first** (library + CLI); UI chosen in Phase 1 | Lowest early risk; prove the core to correctness before committing a UI stack. |
| Language split | **Polyglot**: Rust core; Python for CV/ML training, exported to ONNX and run in Rust via `ort` | Rust runtime + mature ML tooling; single-binary inference. |
| Starting point | **Phase 0: the physics engine spine** | Needs no data; unblocks editor, solver, and analytics simultaneously. |

## The core idea: the engine is the spine, not one of four features

Reconstruction, editing, solving, and analytics are all **consumers** of one
shared core: a domain model (`TableSpec` / `BallState` / `Shot` / `Trajectory`)
and a **deterministic, event-based physics engine**. If every subsystem speaks
the same types and runs through the same simulator, they compose for free.

A single concept unifies three of the four goals: **the size of a shot's success
basin under execution noise** (Monte-Carlo perturbation of aim/speed/spin) is
simultaneously the *solver's* quality metric, the *difficulty* score for
classification, and the *expected-success* number for player analytics.

### Where ML actually earns its keep
- **Perception** (ball detect/track, table calibration, scenario classify) — genuinely ML, trained on video.
- **Solving** — the highest-value use of real video is **system identification**:
  fit the simulator's physical parameters so simulated trajectories match real
  ones, then *plan against the calibrated simulator*. This stays explainable
  ("scores, tolerates ±3° of aim error") — which is what a coach needs. A
  black-box config→shot policy is deferred/optional.

### Design constraint to respect now
**Spin is not directly observable** from standard video — you infer it from path
curvature and cushion rebound. Treat spin as a *latent/estimated* quantity in
reconstruction and calibration, never a measured one.

## Architecture

Rust workspace (`crates/`):

- **`billiards-core`** — the shared vocabulary. No physics *policy*, only types:
  `BallState`, `BallSpec`, `TableSpec`, `PhysicsParams`, and the `Trajectory`
  contract (`MotionSegment` / `MotionPhase`) that everything downstream samples.
- **`billiards-engine`** — the event-based physics simulator. Advances the world
  event-to-event, solving each phase in closed form (no time-stepping) so the
  solver can call it millions of times.
- **`billiards-cli`** — headless `billiards` binary; demo today, real subcommands
  (`sim` / `solve` / `reconstruct`) arrive with their phases.
- *(future)* `billiards-solver`, `billiards-vision` (Rust inference), plus a
  Python `training/` tree for models exported to ONNX.

### Coordinate & unit conventions
SI throughout, `f64`. Table surface is the `z = 0` plane, `+z` up; a resting
ball's center is at `z = radius`. `angular_vel.z` is the english (sidespin).

## Physics references (accuracy bar)

- **Mathavan, Jackson & Parkin (2010)** — billiard ball / cushion impact model.
- **Han (2005)** — cushion model (as used by `pooltool`).
- **`pooltool`** (Evan Kiefl) — open-source event-based billiard sim; the prior
  art to study, not re-invent.
- Alciatore (Dr. Dave) — throw, spin transfer, diamond-system references.

The engine's cloth-friction model (implemented) gives the classic result: a ball
struck flat enters natural roll at exactly `(5/7)·v₀`. This is a regression test.

## Roadmap

Each phase is independently demoable; the data-hungry/ML work is deliberately last.

- **Phase 0 — Spine.** Domain model + event-based engine.
  - [x] Shared domain model & `Trajectory` contract.
  - [x] Single-ball free motion (slide→roll→stop), validated by closed-form tests.
  - [x] Ball–cushion rebound (Han 2005 carom model), single ball on a bounded table.
  - [x] Ball–ball collision with throw (equal-mass impulse, Alciatore/pooltool-style).
  - [x] Multi-ball event scheduler (`simulate`): soonest of transition / cushion /
        ball–ball (quartic root) across all balls; ordered contact-event log.
  - [x] Validate banks against the corner-5 diamond system (TP 7.2): with running
        english the arrival tracks `T = D − F` (slope −1.03…−1.05), intercept
        `D ≈ 5.5` matching the corner, residuals ≤ ~0.3 diamond. Regression test +
        `examples/diamond_system.rs`.
  - [ ] English (`ω_z`) decay via spinning friction on the cloth (minor refinement).
- **Phase 1 — Editor/viz.** Render table+balls, scrub trajectories, drag balls,
  set aim/speed/spin, re-sim. (This *is* the "as-if" tool minus video.) Pick UI here.
  - [x] `billiards-ui` (egui/eframe): top-down table with diamonds; drag balls;
        aim/speed sliders + aim arrow; live trajectory rendering; play/scrub;
        three-cushion scoring readout; "Solve" loads the solver's best shot with
        success %/difficulty. UI stack decision: **egui** (2D is the right view
        for a planar game; fast path). Run: `cargo run -p billiards-ui`.
  - [x] **Cue-tip strike diagram**: interactive ball-face showing/setting the tip
        contact point from spin+speed via `offset/R = 2Rω/(5v)` (natural roll =
        0.4R, unit-tested). `CueAction` now carries `follow` (topspin/draw),
        simulated by the engine. Surfaces the speed↔offset coupling and miscue
        limit; the solver now rejects strikes past ½R as un-hittable.
  - [ ] Cue elevation (massé); solver to *search* follow/draw; save/load scenes.
- **Phase 2 — Solver.** Search the action space with the engine as forward model;
  Monte-Carlo robustness → best shot + difficulty score.
  - [x] `billiards-solver`: search (aim, speed, **2D tip offset = english +
        follow/draw** over the miscue disk) → scoring actions; Monte-Carlo
        robustness under execution noise → most forgiving shot + success
        probability + difficulty. **Parallel (rayon), deterministic** (fixed
        per-candidate seeds, index tie-break). Searching follow/draw ~doubled the
        demo scene's best success (20%→41%). ~0.1s.
  - [ ] Cluster scoring actions into distinct shot "families"; add cue elevation
        (massé) to the action space.
- **Phase 3 — Perception.** Table homography → ball detect/track → shot
  segmentation → same `Shot`/`Trajectory` types. Python-trained, ONNX inference.
  - [x] `billiards-vision`: geometric backbone, end-to-end & tested without data.
        `Homography`/`Calibration` (4-corner DLT, image↔table); `BallDetector`
        trait (the ONNX seam) + classical `ColorBlobDetector`; `reconstruct_scene`
        (detections → core `Scene`); synthetic `render_scene` camera for
        ground-truth tests. Demo recovers a scene from a perspective image to
        ~1 mm and solves it. `cargo run -p billiards-vision --example reconstruct_demo`.
  - [x] **Real footage, Python workbench** (`python/`): `ingest.py` pulls frames
        from a YouTube stream (yt-dlp section download + ffmpeg); `reconstruct_frame.py`
        calibrates a homography from 4 table corners, detects the balls by HSV
        color inside the (eroded) table region, and lifts to table coordinates.
        Validated on a 2026 World Cup 3-cushion broadcast: all three balls
        correctly located (annotated overlay confirmed by eye). This is the
        polyglot CV bench; settled algorithms port to Rust / an ONNX detector.
  - [x] **Tracking + shot segmentation** (`python/track.py`): three-cushion's
        3 distinctly-colored balls make per-color detection *be* the data
        association. Streams frames → per-frame HSV detection (round-blob +
        teleport gating rejects the cue stick / false blobs) → lift to table
        coords → motion-based shot segments. Validated on a synthetic ground-truth
        clip (mean 1.8 mm, correct segmentation). Runs on **real live-stream inset
        footage**: all 3 balls tracked over 1351 frames, 5 shots segmented with
        plausible travel (0.24–1.39 m).
  - [x] **Force/spin reconstruction** (`billiards-solver::fit`): the inverse
        problem — fit the `CueAction` (aim, speed, english, follow/draw) whose
        simulated trajectory matches the observed tracks. Engine as forward model,
        the solver's search in reverse. Aim is read directly from the cue's
        pre-collision heading (a level shot goes straight regardless of spin), so
        the search is a *narrow* window around it + multi-start coordinate descent
        (trajectory-matching across collisions is very non-convex). Validated on
        ground truth: recovers aim/force/english/follow to <1% (rms 0.6 mm, 54 ms),
        robust to ~5 mm tracking noise. `examples/reconstruct_hit.rs`.
  - [x] **Event-skeleton matching in the fit** (`ObservedEvents` + `event_penalty`):
        pointwise RMS is shape-blind — on a long track the resting tail dominates,
        so a path telling the wrong physical story (other rails, invented contacts)
        can out-score the faithful one. The fit now extracts the *observed* skeleton
        (ball first-motion + cushion bounces as Vs in distance-to-rail, with
        corner/ball-collision disambiguation and near-rail-bias-tolerant
        thresholds) and charges candidates for every missing/fabricated element —
        effectively lexicographic: right causal class first, RMS second. A sim-only
        bounce is charged only when the tracks *positively* contradict it (gap ≠
        veto). Real-match verify: median cue error 196→159 mm; wrong-element
        "good RMS" fits (tail-coasting) are now rejected. `examples/skeleton_diag.rs`.
  - [x] **Identity-swap guard in first-move extraction** (`FirstMove` three-state):
        when the cue rolls to rest near an object ball the tracker can re-label
        the cue's blob as that ball — the ball's track "moves" while riding the
        other ball's path (physically impossible: centers can't be within 2R),
        then snaps back to its rest. The old extractor read this as a late strike
        and the missing-contact penalty *forced the fit to fabricate a collision
        that never happened* (game_00/shot_12: phantom cue⇄red @11.3 s — which
        would have *scored*, yet the shot is an annotated miss). Motion claims are
        now vetted against other balls' interpolated paths; contradictory tracks
        are `Unreliable` and constrain nothing, in either direction.
  - [x] **Cue-label sanity + contact-aim override** (found via game_00/shot_15,
        confirmed frame-by-frame): (a) the header's `cue` color can be wrong —
        the labeled cue never moved while white flew from frame one; `shotfile::parse`
        now relabels the cue to the earliest mover (white/yellow only) when the
        labeled cue verifiably never leaves its rest. (b) A strike right at clip
        start blurs the launch chord AND its speed slope — the measured aim
        missed the verifiably-struck ball by 15 cm, so the pinned windows made
        the true action unreachable and the fit ate the missing-contact penalty.
        When the earliest credible object motion is geometrically unreachable
        from the launch window (and no cue cushion precedes it), the aim window
        is *enlarged* to span both hypotheses and the speed pin opens; the event
        skeleton + whole-track error arbitrate. shot_15: 756→151 mm; the deployed
        game verifies 12 PASS / 0 WARN / 1 FAIL (median cue 119 mm).
  - [x] **Gap-tolerant first-move + ordered skeleton** (game_00 shots 03/05): a
        hard-struck ball can fly undetected for 0.5–0.7 s, so (a) the causality
        window now opens at the mover's own last pre-motion sample (not a fixed
        0.35 s before re-detection), dating the contact by the striker's closest
        approach; (b) credibility counts the next few samples wherever they fall.
        And cushion matching is now an *order-preserving* alignment (LCS) — greedy
        time-window pairing let a fabricated early bounce hide behind a real later
        one on the same rail (rail-first path impersonating ball-first). Plus: the
        aim window always spans the verified first-struck ball's full contact cone
        (thickness freedom), and the speed pin opens when the struck ball's
        departure speed crowds the launch ceiling (equal-mass energy bound).
        shot_05: FAIL 507→257 mm with the true contact at 0.44 s. game_00:
        11 PASS / 2 WARN / 0 FAIL. Known frontier: ball–ball collision response
        (throw/spin transfer, uncalibrated) — shot_03's residual ~530 mm.
  - [x] **End-to-end pipeline wired**: real clip → `ingest.py` → `track.py`
        (`--fit-out shot.csv`) → `billiards-solver` `fit_csv` example → force/spin.
        Runs; and the fit's trajectory **RMS is a built-in quality gauge**: 0.6 mm
        on clean data vs **44–81 mm on the real inset shots** — flagging that the
        classical detector mis-identifies balls (red latches a fixed rail object;
        the cue is mis-assigned) and the motion segmenter catches mid-stroke
        fragments. So the plumbing is done; the *numbers* await better perception.
  - [x] **Ball-detection approach decided** (the key perception finding):
        - Tiny from-scratch net on **domain-randomized synthetic** (`synth_data.py`,
          `detector.py`, 24k params, CPU ~1 min) → 88–93% on synthetic but a
          **sim-to-real gap** (real balls activate at only 0.1–0.2 vs 0.92).
        - A **COCO-pretrained detector** (torchvision `fasterrcnn`) recognizes
          billiard balls as *"sports ball"* with **zero training** (yellow 0.82,
          white found; red missed — small/low-contrast). Its real-image backbone
          has no sim-to-real gap.
        - ⇒ **Plan: fine-tune a pretrained detector on our synthetic data.** Real
          backbone (robustness) + synthetic fine-tuning (all 3 ball colors at
          billiard sizes). Our synthetic generator is the fine-tuning data.
  - [x] **Fine-tuned detector WORKS on real footage** (`python/finetune_detector.py`,
        `eval_frames.py`): `fasterrcnn_mobilenet_v3_large_320_fpn` (COCO-pretrained),
        head swapped for 3 ball classes, fine-tuned ~8 min CPU on 500 synthetic
        frames. Real results: World Cup crop **all 3 @ 0.83–0.95**; the *heavily
        angled* bilardo main camera (classical failed here) **red+yellow @ 0.97**.
        Limits + fixes: table must fill the frame (crop via calibration first);
        occasional rail false-positive (reject via table mask); 5 px inset still
        marginal (use main cam). This is the perception unlock.
  - [x] **Detector wired into the tracker** (`track.py --detector learned`, with
        table-mask filtering to drop rail false-positives). On a real World Cup
        overhead clip: **all 3 balls @ 0.86–0.94** per frame; on broadcast
        **cut-away** frames it correctly returns nothing — so the detector is also
        a free **view classifier** (confident 3-ball detections = analyzable
        overhead segment; cuts = detection gaps).
  - [x] **End-to-end on a real bilardo shot** (the right source — uniform fixed
        main camera, no cuts): learned detector tracks all 3 balls **675/675
        frames**, 4 shots segmented, real cue ball (vs classical: yellow invisible,
        cue mis-ID'd as red, phantom shots). Reconstruction runs (fit RMS ~570 mm
        on a 3 m multi-cushion stroke). **Perception is solved; the bottleneck has
        shifted downstream** to: rough (hand-read) calibration, **uncalibrated
        physics (Phase 4)** — dominant error on long multi-cushion shots — plus
        detection jitter and a naive first-mover cue heuristic.
  - [x] **Use the top-down inset, not the angled camera** — parallax-free ⇒
        *unbiased* positions. Detecting the ~5 px inset balls needed a retrain with
        the higher-input-res model (`fasterrcnn_mobilenet_v3_large_fpn`) + small-ball
        synthetic; then all 3 track cleanly, no jitter. **Reconstruction on the
        inset: 73 mm RMS vs 568 mm on the angled camera (~8× better)** — low-res-but-
        unbiased beats hi-res-but-biased. Side-by-side videos in `media/` (angled
        diverges, inset aligns). Strokes located via a table-interior motion profile.
  - [x] **ONNX export + parity gate** (`python/export_onnx.py` → `detector.onnx`,
        76 MB): the whole torchvision pipeline (internal resize/normalize, RPN,
        ROI heads, NMS) exports as one graph — raw float RGB in, boxes/labels/
        scores out. Two traps found and gated: (1) "dynamic axes" exports pass
        the checker but bake input-size-dependent constants — at 332×184 the
        graph fabricated detections (and dynamo/torch.export can't trace
        detection models at all), so the graph is FIXED at the native inset
        size 3×515×290 and consumers letterbox into it (`letterbox()` is the
        reference the Rust `OnnxDetector` must mirror); (2) tracing with random
        noise bakes degenerate ROI shapes (runtime Reshape failure) — trace with
        a real 3-ball frame. Parity: bit-exact (Δbox 0.000 px, Δscore 0.0000)
        over 20 native + 8 letterboxed real frames; ort CPU is **3.0× faster**
        than torch CPU (104 vs 314 ms/frame).
  - [x] **`OnnxDetector` in Rust** (`billiards-vision::onnx`, feature `onnx`):
        implements the `BallDetector` seam + `detect_scored()` (all candidates
        with scores, for the tracker's joint color assignment); letterboxes any
        frame into the canonical canvas, maps boxes back. Cross-language parity
        vs the .pt model: native frames **bit-exact det-for-det**; letterboxed
        frames identical dets with ≤0.05 score drift (bilinear resampler diff).
        macOS gotcha: pyke's prebuilt static onnxruntime references libc++
        `to_chars` overloads Apple never shipped → use `load-dynamic` with
        `ORT_DYLIB_PATH` pointing at the pip wheel's dylib; `ort` pinned
        `=2.0.0-rc.10` (rc.12 doesn't compile with default-features off).
        `examples/onnx_detect.rs` is the runner.
  - [x] **Tracker core ported to Rust** (`billiards-vision::track`): the full
        learned-detector path of track.py — table mask, blob merge + JOINT
        color assignment (exclusivity), classical HSV fallback (`classical_ball`:
        the CNN misses motion-blurred fast balls — i.e. the stroke — so the
        fallback is load-bearing, not a corner case), white/yellow continuity
        swap check (with the REPAIRED 0.055/0.35 thresholds), gap-scaled
        teleport gating, coincidence resolution, `fill_gaps`, `segment_shots`.
        Cross-language validation (`examples/track_frames.rs` vs track.py on
        the same real 360-frame shot): identical row sets, identical segments,
        **max position delta 0.1 mm**. Port traps: track.py's `mask_pad` is a
        cv2 KERNEL SIZE (12 ⇒ ~6 px), not a distance; blob list is truncated
        to 8 AFTER building, not capped during.
  - [x] **`billiards track` — the single-binary tracker** (`billiards-cli`,
        feature `track`): frames dir OR any video via ffmpeg raw-RGB piping →
        detector → tracker → `rest_bounds`/`busiest_shot`/`shot_cue` →
        `shot_NN.shot` per segment (skipping mid-shot starts), format-identical
        to track.py's `export_for_fit` and consumable by the editor/fit as-is.
        Validated: raw-pipe vs decoded-PNG tracking agree to **0.00 mm**;
        NOTE the bundle's display MP4s (crf 26, 4:2:0) measurably degrade
        8-px ball detection (1045→757 rows on a test shot) — track from
        source-quality video, the display clips are for the viewer only.
        Remaining: clock-based match segmentation port (detect_shots.py),
        `video_t0` plumbing, Linux static-ort build for CI/VM deployment.
  - [x] **`billiards track --match` — whole-game segmentation in Rust**
        (`match_cmd.rs`, port of detect_shots.py + build_match.py's loop): one
        ffmpeg pass reads the green shot-clock ring (resets = turn boundaries,
        no OCR) + inset motion; rest-to-rest stroke bounding per clock window;
        per-shot inset extraction (ffmpeg crop) → tracking → `shot_NN.shot`
        with `frames`/`video_t0` back-links. Validated on the first 15 min of
        masa4_full.mp4 vs the Python-built game_00: **21/23 shots at the
        identical frame (Δt = 0.0 s)**, 2 marginal clean-start disagreements,
        4 extra segments recovered; produced shots fit cleanly (124 mm on a
        spot check). Port trap: build_match PRESETS are ABSOLUTE per-resolution
        pixels; only the scoreboard overlay scales from the 720p base.
        Still Python: montage, make/miss annotation (annotate_results.py).
  - [x] **Input-size question CLOSED by measurement**: the Python pipeline's ×3
        inset pre-upscale (build_match `scale`) is (near-)redundant — the
        model's internal `Resize(min=800, max=1333)` already brings a native
        515×290 frame to the same 1333-px working resolution the upscaled input
        lands at. Measured on real frames: same balls, positions within 0.2 px,
        scores within a few hundredths (cubic pre-upscale adds slight
        sharpening only). The fixed-size ONNX export forfeits essentially
        nothing; the Rust tracker correctly skips the upscale.
  - [x] **Linux deployment build**: ort linkage is now target-specific in
        `billiards-vision/Cargo.toml` — Linux gets the normal STATIC link
        (single 24 MB `billiards` binary, zero onnxruntime dylib deps,
        verified in a rust:1 container), macOS keeps `load-dynamic` +
        `ORT_DYLIB_PATH` (Apple SDK lacks the libc++ symbols pyke's static lib
        needs). This is the cheap-hosting artifact: the binary + detector.onnx
        + ffmpeg/yt-dlp run a match on any Linux box or CI runner, no Python.
  - [x] **Snapshot testing infrastructure** (`snapshots/`, see its README):
        the regression gate the port (and all future fit/engine work) is judged
        against. Two layers: `snapshot_recon` — one line per shot across the
        whole corpus (~523 shots: action, cue/obj RMS, event penalty, compact
        event story), per game dir; `snapshot_tracks` — full tracker rows for a
        curated 10-shot manifest spanning games + tricky cases (swaps, fast
        strokes). Both deterministic (verified: re-run byte-identical), so
        `check` diffs exactly and classifies changes (improved/worse cue-RMS
        per shot) for a human to `bless`. Workflow: change code → `check` →
        read the per-shot report → bless or fix.
  - [ ] **Also on the critical path**: (a) accurate calibration — auto corner
        detection + parallax correction; (b) white/yellow discrimination at
        track time (biggest report cluster); (c) detection-jitter smoothing;
        (d) scoreboard OCR hardening.

## Real-video architecture (Phase 3, in progress)

Polyglot split in action: **Python** (`python/`) owns video I/O and CV/ML
prototyping; **Rust** owns the geometry/engine and will run trained models via
ONNX. Stream → frames (`ingest.py`) → calibrate + detect (`reconstruct_frame.py`,
mirrors Rust `billiards-vision`) → `Scene`. The prize downstream is **hit
reconstruction**: observed cue-ball trajectory ⇒ fit `(aim, speed, english,
follow/draw)` by matching the engine's forward simulation — the same
forward-model-as-oracle idea as the solver, run in reverse. yt-dlp must be kept
current (YouTube breaks old versions); progressive itag 18 is the reliable format.
- **Phase 4 — Calibrate + analytics.** System-ID the physics params against real
  shots; shot database, scenario classifier, per-player success rates.
  - [x] **Physics calibration method** (`billiards-solver::calibrate`): nested
        optimization — outer over the physics parameters (cushion
        restitution/friction, cloth sliding/rolling friction), inner `fit_action`
        recovers each shot's unknown cue action. Validated on ground truth: from a
        *wrong* (default) start it recovers hidden params (e_c 0.80→0.79,
        μ_r 0.013→0.0115, …) in ~3 s. The sim-to-real bridge.
  - [ ] Run it on real bilardo shots (needs clean multi-shot data first: accurate
        corners + parallax + jitter smoothing); then shot DB, scenario classifier,
        per-player success rates.

## Status (2026-07-03)

Phase 0 effectively complete; **Phase 2 solver** has a working first version.
17 passing tests. Landed: workspace + core model; single-ball cloth motion;
**Han 2005 ball–cushion rebound**, **validated against the corner-5 diamond
system** (banks track `T = D − F`, slope ≈ −1 with running english, `D ≈ 5` at
the corner); **ball–ball collision with throw**; the **multi-ball event
scheduler** (`simulate`) with a contact-event log; three-cushion **scoring**; and
the **`billiards-solver`** (grid search + Monte-Carlo robustness → most forgiving
shot, success probability, difficulty); the **`billiards-ui`** egui editor with a
cue-tip strike diagram; and the **`billiards-vision`** perception backbone
(homography calibration → detection → `Scene`, recovering a synthetic frame to
~1 mm). Six crates. The deterministic core (engine, solver, editor) is complete;
perception has its geometric first slice. Remaining: learned detectors on real
video (Python→ONNX), tracking, shot segmentation; and physics calibration
(Phase 4). Minor Phase 0 item: english decay via spinning friction.
