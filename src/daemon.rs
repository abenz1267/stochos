//! Optional persistent process. Display objects stay on their owning thread;
//! a small socket thread handles activation/status/stop even while it is active.
use crate::{
    app::{InitialMode, Session},
    backend::Backend,
    cli::Args,
};
use anyhow::{bail, Context, Result};
use nix::{
    poll::{poll, PollFd, PollFlags, PollTimeout},
    sys::signal::{self, SigAction, SigHandler, SigSet, Signal},
    unistd::Uid,
};
use std::{
    fs::{self, DirBuilder},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Read, Write},
    os::{
        fd::AsFd,
        unix::{
            fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt},
            net::{UnixListener, UnixStream},
        },
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, OnceLock,
    },
    time::Duration,
};

const PROTOCOL: &str = "stochos-1";
const IO_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Default)]
struct Control {
    busy: AtomicBool,
    stop: AtomicBool,
}
static CONTROL: OnceLock<Arc<Control>> = OnceLock::new();
static SIGNAL_STOP: AtomicBool = AtomicBool::new(false);

pub fn running() -> bool {
    CONTROL.get().is_some()
}
pub fn shutdown_requested() -> bool {
    SIGNAL_STOP.load(Ordering::Relaxed)
        || CONTROL
            .get()
            .is_some_and(|c| c.stop.load(Ordering::Relaxed))
}

extern "C" fn stop_signal(_: i32) {
    SIGNAL_STOP.store(true, Ordering::Relaxed);
}

pub fn socket_path() -> Result<PathBuf> {
    let uid = Uid::current().as_raw();
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let directory = base.join(format!("stochos-{uid}"));
    match DirBuilder::new().mode(0o700).create(&directory) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e).context("create daemon socket directory"),
    }
    let metadata = fs::symlink_metadata(&directory)?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        bail!(
            "daemon socket directory must be owned by this user with mode 0700: {}",
            directory.display()
        );
    }
    let mut display = std::collections::hash_map::DefaultHasher::new();
    // XWayland may start after the daemon. Its DISPLAY must not change the
    // endpoint used by clients on the same native Wayland display.
    if let Some(name) = std::env::var_os("WAYLAND_DISPLAY") {
        ("wayland", name).hash(&mut display);
    } else {
        ("x11", std::env::var_os("DISPLAY")).hash(&mut display);
    }
    Ok(directory.join(format!("{:016x}.sock", display.finish())))
}

fn mode_name(mode: InitialMode) -> &'static str {
    match mode {
        InitialMode::Normal => "normal",
        InitialMode::Hint => "hint",
        InitialMode::Bisect => "bisect",
        InitialMode::Free => "free",
        InitialMode::FreeCenter => "free-center",
    }
}
fn parse_mode(mode: &str) -> Result<InitialMode> {
    Ok(match mode {
        "normal" => InitialMode::Normal,
        "hint" => InitialMode::Hint,
        "bisect" => InitialMode::Bisect,
        "free" => InitialMode::Free,
        "free-center" => InitialMode::FreeCenter,
        _ => bail!("unknown activation mode"),
    })
}

fn send(path: &Path, command: &str) -> Result<Option<String>> {
    let mut socket = match UnixStream::connect(path) {
        Ok(socket) => socket,
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            return Ok(None)
        }
        Err(e) => return Err(e).context("connect to stochos daemon"),
    };
    socket.set_read_timeout(Some(IO_TIMEOUT))?;
    socket.set_write_timeout(Some(IO_TIMEOUT))?;
    writeln!(socket, "{PROTOCOL} {command}")?;
    let mut response = String::new();
    BufReader::new(socket)
        .take(256)
        .read_line(&mut response)
        .context("read daemon reply")?;
    Ok(Some(response.trim().to_owned()))
}

/// Return before backend/config initialization when a resident daemon handles
/// this invocation. An absent daemon preserves the original one-shot behavior.
pub fn route(args: &Args) -> Result<bool> {
    if args.daemon || args.no_daemon || args.allow_multiple {
        return Ok(false);
    }
    let path = socket_path()?;
    let command = if args.stop_daemon {
        "stop".to_string()
    } else if args.daemon_status {
        "status".to_string()
    } else {
        format!(
            "activate {} {}",
            mode_name(args.initial_mode()),
            u8::from(crate::timing::enabled())
        )
    };
    match send(&path, &command)? {
        None if args.stop_daemon || args.daemon_status => bail!("stochos daemon is not running"),
        None => Ok(false),
        Some(reply) if reply == "ok" || reply == "busy" || reply == "stopping" => Ok(true),
        Some(reply) if args.daemon_status && reply.starts_with("running ") => {
            println!("{reply}");
            Ok(true)
        }
        Some(reply) => bail!("daemon rejected request: {reply}"),
    }
}

struct SocketGuard {
    listener: UnixListener,
    path: PathBuf,
}
impl SocketGuard {
    fn bind(path: PathBuf) -> Result<Self> {
        // Only recover a refused Unix socket. Never unlink a live endpoint or
        // an arbitrary file left at the expected path.
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_socket() {
                bail!("daemon socket path is not a socket");
            }
            match UnixStream::connect(&path) {
                Ok(_) => bail!("stochos daemon is already running"),
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                    fs::remove_file(&path)?
                }
                Err(e) => return Err(e).context("check previous daemon socket"),
            }
        }
        let listener = UnixListener::bind(&path).context("bind stochos daemon socket")?;
        let guard = Self { listener, path };
        fs::set_permissions(&guard.path, fs::Permissions::from_mode(0o600))?;
        guard.listener.set_nonblocking(true)?;
        Ok(guard)
    }
}
impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct Activation {
    mode: InitialMode,
    timings: bool,
}

fn handle(
    socket: &mut UnixStream,
    tx: &mpsc::SyncSender<Activation>,
    control: &Control,
) -> Result<()> {
    socket.set_read_timeout(Some(Duration::from_millis(250)))?;
    socket.set_write_timeout(Some(Duration::from_millis(250)))?;
    let mut request = String::new();
    BufReader::new(&mut *socket)
        .take(256)
        .read_line(&mut request)?;
    let words: Vec<_> = request.split_whitespace().collect();
    let reply = match words.as_slice() {
        [PROTOCOL, "status"] => {
            if control.busy.load(Ordering::Acquire) {
                "running active"
            } else {
                "running idle"
            }
        }
        [PROTOCOL, "stop"] => {
            control.stop.store(true, Ordering::Release);
            "stopping"
        }
        [PROTOCOL, "activate", mode, timings @ ("0" | "1")] => {
            let mode = parse_mode(mode)?;
            if control.stop.load(Ordering::Acquire) {
                "stopping"
            } else if control
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                "busy"
            } else if tx
                .try_send(Activation {
                    mode,
                    timings: *timings == "1",
                })
                .is_ok()
            {
                "ok"
            } else {
                control.busy.store(false, Ordering::Release);
                "error unavailable"
            }
        }
        _ => "error invalid request",
    };
    writeln!(socket, "{reply}")?;
    Ok(())
}

pub fn serve<B: Backend>(backend: &mut B) -> Result<()> {
    let socket = SocketGuard::bind(socket_path()?)?;
    let control = Arc::new(Control::default());
    CONTROL
        .set(control.clone())
        .map_err(|_| anyhow::anyhow!("daemon already initialized"))?;
    let action = SigAction::new(
        SigHandler::Handler(stop_signal),
        signal::SaFlags::empty(),
        SigSet::empty(),
    );
    // The handler only stores an atomic flag; GUI teardown remains on the
    // backend's thread. Poll timeouts wake active sessions and the socket loop.
    unsafe {
        signal::sigaction(Signal::SIGTERM, &action)?;
        signal::sigaction(Signal::SIGINT, &action)?;
    }
    let (tx, rx) = mpsc::sync_channel(1);
    let socket_control = control.clone();
    let listener = std::thread::spawn(move || {
        while !shutdown_requested() {
            let mut fds = [PollFd::new(socket.listener.as_fd(), PollFlags::POLLIN)];
            if poll(&mut fds, PollTimeout::from(100u16)).is_err() {
                continue;
            }
            if let Ok((mut connection, _)) = socket.listener.accept() {
                if let Err(e) = handle(&mut connection, &tx, &socket_control) {
                    let _ = writeln!(connection, "error {e}");
                }
            }
        }
    });
    let default_timings = crate::timing::enabled();
    crate::hint::warm_up();
    let mut session = Session::new();
    eprintln!("stochos daemon ready");
    while !shutdown_requested() {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(activation) => {
                crate::timing::enable(default_timings || activation.timings);
                let reopen = {
                    let _span = crate::timing::Span::new("overlay-reopen");
                    backend.reopen()
                };
                let result = reopen.and_then(|_| session.activate(backend, activation.mode));
                // Both errors and successful actions must release keyboard and
                // hide the overlay before the daemon accepts another request.
                let cleanup = backend.exit();
                if let Err(e) = result.and(cleanup) {
                    if !shutdown_requested() {
                        eprintln!("daemon activation failed: {e:#}");
                    }
                }
                control.busy.store(false, Ordering::Release);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    control.stop.store(true, Ordering::Release);
    let _ = backend.exit();
    let _ = listener.join();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exchange(command: &str, tx: &mpsc::SyncSender<Activation>, control: &Control) -> String {
        let (mut client, mut server) = UnixStream::pair().unwrap();
        writeln!(client, "{PROTOCOL} {command}").unwrap();
        handle(&mut server, tx, control).unwrap();
        drop(server);
        let mut reply = String::new();
        client.read_to_string(&mut reply).unwrap();
        reply.trim().to_string()
    }
    #[test]
    fn activation_busy_status_stop_and_invalid_protocol() {
        let (tx, rx) = mpsc::sync_channel(1);
        let control = Control::default();
        assert_eq!(exchange("status", &tx, &control), "running idle");
        for mode in ["normal", "hint", "bisect", "free", "free-center"] {
            assert_eq!(exchange(&format!("activate {mode} 1"), &tx, &control), "ok");
            let event = rx.recv().unwrap();
            assert_eq!(mode_name(event.mode), mode);
            assert!(event.timings);
            assert_eq!(exchange("activate normal 0", &tx, &control), "busy");
            assert_eq!(exchange("status", &tx, &control), "running active");
            control.busy.store(false, Ordering::Release);
        }
        assert_eq!(exchange("bogus", &tx, &control), "error invalid request");
        assert_eq!(exchange("stop", &tx, &control), "stopping");
        assert_eq!(exchange("activate normal 0", &tx, &control), "stopping");
    }
    #[test]
    fn stale_socket_recovery_and_cleanup_preserve_other_files() {
        let directory =
            std::env::temp_dir().join(format!("stochos-socket-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("daemon.sock");
        drop(UnixListener::bind(&path).unwrap());
        let guard = SocketGuard::bind(path.clone()).unwrap();
        assert!(SocketGuard::bind(path.clone()).is_err());
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        drop(guard);
        assert!(!path.exists());
        fs::write(&path, "keep").unwrap();
        assert!(SocketGuard::bind(path.clone()).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "keep");
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
