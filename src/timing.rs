//! Opt-in elapsed timings; no clock reads on the normal path.
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable(value: bool) {
    ENABLED.store(value, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub struct Span {
    label: &'static str,
    start: Option<Instant>,
}
impl Span {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: enabled().then(Instant::now),
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            eprintln!(
                "timing: {} {:.2} ms",
                self.label,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
