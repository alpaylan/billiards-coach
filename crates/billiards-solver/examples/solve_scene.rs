//! Solve a three-cushion scene: find the most forgiving scoring shot and report
//! its success probability and the scene's difficulty.
//!
//! Run: `cargo run -p billiards-solver --example solve_scene --release`

use std::time::Instant;

use billiards_core::math::DVec3;
use billiards_core::{BallSpec, PhysicsParams, Scene, TableSpec};
use billiards_solver::{SolveConfig, solve};

fn main() {
    let ball = BallSpec::carom();
    let phys = PhysicsParams::default();
    let table = TableSpec::carom_match();
    let r = ball.radius;

    let scene = Scene::new(
        DVec3::new(-1.15, -0.45, r),                     // cue
        vec![DVec3::new(0.65, 0.28, r), DVec3::new(1.05, -0.10, r)], // objects
    );

    let cfg = SolveConfig::default();
    let t0 = Instant::now();
    let solution = solve(&scene, &table, &ball, &phys, &cfg);
    let elapsed = t0.elapsed();

    println!("three-cushion solver");
    println!("  cue    at ({:+.2},{:+.2})", scene.cue.x, scene.cue.y);
    for (i, o) in scene.objects.iter().enumerate() {
        println!("  obj {} at ({:+.2},{:+.2})", i + 1, o.x, o.y);
    }
    println!(
        "  searched aim×speed×tip grid ({}×{}×~{}²) in {:.2}s\n",
        cfg.aim_steps, cfg.speed_steps, cfg.offset_steps, elapsed.as_secs_f64()
    );

    match solution {
        Some(s) => {
            let (h, v) = s.action.tip_offset(r);
            println!("  BEST SHOT");
            println!("    aim      {:+.1}°", s.action.aim.to_degrees());
            println!("    speed    {:.2} m/s", s.action.speed);
            println!("    tip      {:+.0}%R english, {:+.0}%R follow/draw", h * 100.0, v * 100.0);
            println!("    spin     english {:+.0}, follow {:+.0} rad/s", s.action.sidespin, s.action.follow);
            println!("    success  {:.0}%  ({} scoring cells found)", s.success_prob * 100.0, s.scoring_cells);
            println!("    difficulty {:.2} — {}", s.difficulty(), s.category());
        }
        None => println!("  no scoring shot found at this resolution — try finer search or more english"),
    }
}
