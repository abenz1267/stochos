use crate::backend::Backend;
use crate::config::config;
use crate::hint::build_hint_mode;
use crate::input::InputState;
use crate::macro_store::MacroStore;
use crate::mode::{Mode, ModeTransition};

#[derive(Clone, Copy)]
pub enum InitialMode {
    Normal,
    Bisect,
    Hint,
    Free,
    FreeCenter,
}

pub fn run<B: Backend>(backend: &mut B, initial: InitialMode) -> anyhow::Result<()> {
    Session::new().activate(backend, initial)
}

pub struct Session {
    pixels: Vec<u8>,
    macros: MacroStore,
}
impl Session {
    pub fn new() -> Self {
        Self {
            pixels: Vec::new(),
            macros: MacroStore::load(),
        }
    }
    pub fn activate<B: Backend>(
        &mut self,
        backend: &mut B,
        initial: InitialMode,
    ) -> anyhow::Result<()> {
        let first_frame = crate::timing::Span::new("activation-to-first-submit");
        let (w, h) = backend.screen_size();
        self.pixels.resize((w * h * 4) as usize, 0);
        let pixels = &mut self.pixels;
        let macro_store = &mut self.macros;
        let mut transition_stack: Vec<Mode> = Vec::new();
        let mut mode = match initial {
            InitialMode::Normal => Mode::Normal {
                input_state: InputState::First,
                target: None,
                drag_origin: None,
            },
            InitialMode::Bisect => Mode::Bisect {
                region: (0, 0, w, h),
            },
            InitialMode::Hint => Mode::Hint {
                elements: build_hint_mode(backend, w, h)?.into(),
                typed: Vec::new(),
                target: None,
            },
            InitialMode::Free => {
                let (x, y) = backend.mouse_pos()?;
                Mode::Free {
                    x,
                    y,
                    speed: config().free.base_speed.max(1),
                }
            }
            InitialMode::FreeCenter => Mode::Free {
                x: w / 2,
                y: h / 2,
                speed: config().free.base_speed.max(1),
            },
        };

        if !matches!(
            initial,
            InitialMode::Free | InitialMode::FreeCenter | InitialMode::Hint
        ) {
            backend.move_mouse(w / 2, h / 2)?;
        }

        {
            let _timer = crate::timing::Span::new("first-render-and-present");
            mode.draw(backend, pixels, w, h, macro_store)?;
        }
        drop(first_frame);

        while let Some(key) = backend.next_key()? {
            match mode.handle_key(w, h, backend, &key, macro_store)? {
                ModeTransition::Stay => continue,
                ModeTransition::Redraw => {
                    mode.draw(backend, pixels, w, h, macro_store)?;
                }
                ModeTransition::Enter(m) => {
                    let prev = std::mem::replace(&mut mode, m);
                    transition_stack.push(prev);
                    mode.draw(backend, pixels, w, h, macro_store)?;
                }
                ModeTransition::Replace(m) => {
                    mode = m;
                    mode.draw(backend, pixels, w, h, macro_store)?;
                }
                ModeTransition::Back => {
                    if let Some(prev) = transition_stack.pop() {
                        // Keep the timing baseline only for plain cell-selection undo within
                        // recording: same action count AND same drag state. Anything else (action
                        // dropped, drag attempt abandoned, return from a foreign mode) is a
                        // discarded attempt whose elapsed time shouldn't count toward the next
                        // wait_ms.
                        let keep = matches!(
                            (&mode, &prev),
                            (
                                Mode::MacroRecording {
                                    recorded_actions: cur, drag_origin: cur_drag, ..
                                },
                                Mode::MacroRecording {
                                    recorded_actions: prev_acts, drag_origin: prev_drag, ..
                                },
                            ) if cur.len() == prev_acts.len()
                                && cur_drag.is_some() == prev_drag.is_some()
                        );
                        mode = if keep {
                            prev
                        } else {
                            prev.reset_timing_baseline()
                        };
                        mode.draw(backend, pixels, w, h, macro_store)?;
                    }
                }
                ModeTransition::Exit => {
                    backend.exit()?;
                    break;
                }
            };
        }

        Ok(())
    }
}
