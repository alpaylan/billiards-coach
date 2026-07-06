//! Native test harness for the web viewer: serves the same UI against a bundle
//! over HTTP (run `python3 -m http.server` in the bundle's parent first).
//!
//!     cargo run -p billiards-viewer --release -- http://localhost:8000/bundle

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let base = std::env::args().nth(1).unwrap_or_else(|| "http://localhost:8000/bundle".into());
    eframe::run_native(
        "Billiards Viewer",
        eframe::NativeOptions::default(),
        Box::new(move |_cc| Ok(Box::new(billiards_viewer::ViewerApp::new(base)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
