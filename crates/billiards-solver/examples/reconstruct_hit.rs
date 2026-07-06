//! Reconstruct the exact hit — force and spin — from observed ball trajectories.
//!
//! Here we know the "actual" hit and generate the observation from it (as the
//! tracker would), so we can show how faithfully the fit recovers force, aim and
//! spin. In the real pipeline the observation comes from `python/track.py`.
//!
//! Run: `cargo run -p billiards-solver --example reconstruct_hit --release`

use std::time::Instant;

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, CueAction, PhysicsParams, Scene, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, fit_action, sample_tracks};

fn main() {
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let table = TableSpec::carom_match();
    let r = ball.radius;

    let scene = Scene::new(
        DVec3::new(-1.0, -0.3, r),
        vec![DVec3::new(0.5, 0.15, r), DVec3::new(0.9, -0.25, r)],
    );
    // The shot the player actually made (the solver doesn't get to see this).
    let actual = CueAction::from_tip_offset(0.29, 3.5, 0.20, 0.15, r);

    // What tracking observes: every ball's position at 30 fps.
    let sim = simulate(&scene.ball_states(&actual), &table, &ball, &phys);
    let observed = sample_tracks(&sim, 30.0);

    let t0 = Instant::now();
    let fit = fit_action(&scene, &observed, &table, &ball, &phys, &FitConfig::default());
    let elapsed = t0.elapsed();

    let (h, v) = fit.action.tip_offset(r);
    let (ah, av) = actual.tip_offset(r);
    println!("hit reconstruction");
    println!("  observed {} cue-ball samples over {:.2}s", observed[0].len(), sim.trajectories[0].time_to_rest());
    println!("  fit in {:.0} ms, trajectory rms {:.1} mm\n", elapsed.as_secs_f64() * 1000.0, fit.rms_m * 1000.0);
    println!("  quantity     reconstructed     actual");
    println!("  aim          {:+7.1}°        {:+7.1}°", fit.action.aim.to_degrees(), actual.aim.to_degrees());
    println!("  force        {:6.2} m/s      {:6.2} m/s", fit.action.speed, actual.speed);
    println!("  english      {:+5.0}% R        {:+5.0}% R", h * 100.0, ah * 100.0);
    println!("  follow/draw  {:+5.0}% R        {:+5.0}% R", v * 100.0, av * 100.0);
}
