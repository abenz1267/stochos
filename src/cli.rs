use clap::Parser;

use crate::app::InitialMode;

#[derive(Parser)]
#[command(version, about = "Keyboard-driven mouse overlay")]
pub struct Args {
    /// Start in bisect mode (recursive grid subdivision)
    #[arg(long, group = "mode")]
    pub bisect: bool,

    /// Start in free mode (move the cursor manually)
    #[arg(long, group = "mode")]
    pub free: bool,

    /// Start in free mode at the center of the screen
    #[arg(long, group = "mode")]
    pub free_center: bool,

    /// Start in hint mode (clickable-element labels)
    #[arg(long, group = "mode")]
    pub hint: bool,

    /// Run a persistent foreground daemon (start it with your session)
    #[arg(long, group = "service", conflicts_with_all = ["mode", "allow_multiple", "no_daemon"])]
    pub daemon: bool,

    /// Stop the daemon, including an active overlay
    #[arg(long, group = "service", conflicts_with_all = ["mode", "allow_multiple", "no_daemon"])]
    pub stop_daemon: bool,

    /// Report whether the daemon is idle or active
    #[arg(long, group = "service", conflicts_with_all = ["mode", "allow_multiple", "no_daemon"])]
    pub daemon_status: bool,

    /// Bypass daemon routing (the single-instance lock still applies)
    #[arg(long)]
    pub no_daemon: bool,

    /// Print startup, capture, detection, ranking, and rendering timings
    #[arg(long)]
    pub timings: bool,

    /// Allow multiple concurrent instances
    #[arg(long)]
    pub allow_multiple: bool,
    #[arg(
        long,
        help = "Print the default config (TOML) to stdout and exit. Diff against your config.toml to see new options."
    )]
    pub print_default_config: bool,
}

impl Args {
    pub fn initial_mode(&self) -> InitialMode {
        if self.bisect {
            return InitialMode::Bisect;
        }

        if self.free {
            return InitialMode::Free;
        }

        if self.free_center {
            return InitialMode::FreeCenter;
        }

        if self.hint {
            return InitialMode::Hint;
        }

        InitialMode::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn daemon_flags_preserve_mode_selection_and_reject_conflicts() {
        assert!(Args::try_parse_from(["stochos", "--daemon", "--hint"]).is_err());
        assert!(Args::try_parse_from(["stochos", "--daemon", "--stop-daemon"]).is_err());
        assert!(Args::try_parse_from(["stochos", "--daemon-status", "--no-daemon"]).is_err());
        assert!(Args::try_parse_from(["stochos", "--daemon", "--allow-multiple"]).is_err());
        let args = Args::try_parse_from(["stochos", "--hint", "--timings"]).unwrap();
        assert!(matches!(args.initial_mode(), InitialMode::Hint));
        assert!(args.timings);
        assert!(
            Args::try_parse_from(["stochos", "--no-daemon", "--free"])
                .unwrap()
                .no_daemon
        );
    }
}
