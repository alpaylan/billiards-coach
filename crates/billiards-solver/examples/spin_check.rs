//! One-off: per-object contact structure, observed vs simulated, for a shot.
use std::fs;
use billiards_core::{BallSpec, PhysicsParams, TableSpec};
use billiards_engine::simulate;
use billiards_solver::fit::{FitConfig, ObservedEvents, fit_action};
use billiards_solver::shotfile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let (ball, table) = (BallSpec::carom(), TableSpec::carom_match());
    let phys = PhysicsParams::carom_calibrated();
    let text = fs::read_to_string(&path).unwrap();
    let s = shotfile::parse(&text, &table, ball.radius).unwrap();
    let ev = ObservedEvents::from_tracks(&s.observed, table.center_bounds(ball.radius));
    let cfg = FitConfig { aim_window: 0.03, ..FitConfig::default() };
    let fit = fit_action(&s.scene, &s.observed, &table, &ball, &phys, &cfg);
    let sim = simulate(&s.scene.ball_states(&fit.action), &table, &ball, &phys);
    let (h, v) = fit.action.tip_offset(ball.radius);
    println!("fit: aim {:.1}° speed {:.2} english {:+.0}%R follow {:+.0}%R  rms {:.0}mm",
        fit.action.aim.to_degrees().rem_euclid(360.0), fit.action.speed, h*100.0, v*100.0, fit.rms_m*1000.0);
    for (k, name) in s.order.iter().skip(1).enumerate() {
        let tr = &sim.trajectories[k + 1];
        let p0 = tr.state_at(0.0).pos;
        let mut sm = None;
        let mut t = 0.0;
        while t <= sim.settled_time() {
            if (tr.state_at(t).pos - p0).length() > 0.02 { sm = Some(t); break; }
            t += 1.0 / 30.0;
        }
        println!("{name}: observed first-move {:?} · simulated {:?}", ev.first_move[k], sm);
    }
}
