use once_cell::sync::Lazy;

#[derive(Debug, Default)]
pub struct DebugSettings {
    pub mem: bool,
    pub stream: bool,
    pub event: bool,
    pub init: bool,
    pub ops: bool,
    pub prof: bool,
}

pub static DEBUG_LOG_SETTINGS: Lazy<DebugSettings> = Lazy::new(|| {
    let env_var = std::env::var("PLUG_DEBUG");
    match env_var {
        Ok(opts_str) => {
            let mut debug_opts = DebugSettings::default();

            if opts_str.contains("mem") {
                debug_opts.mem = true;
            }
            if opts_str.contains("stream") {
                debug_opts.stream = true;
            }
            if opts_str.contains("event") {
                debug_opts.event = true;
            }
            if opts_str.contains("init") {
                debug_opts.init = true;
            }
            if opts_str.contains("ops") {
                debug_opts.ops = true;
            }
            if opts_str.contains("prof") {
                debug_opts.prof = true;
            }
            if opts_str.contains("all") {
                debug_opts.mem = true;
                debug_opts.stream = true;
                debug_opts.event = true;
                debug_opts.init = true;
                debug_opts.ops = true;
                debug_opts.prof = true;
            }

            debug_opts
        }
        _ => DebugSettings::default(),
    }
});

#[macro_export]
macro_rules! log_mem {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.mem == true{
                debug!($($arg)+)
            }
        }
    };
}

#[macro_export]
macro_rules! log_stream {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.stream == true{
                debug!($($arg)+)
            }
        }
    };
}

#[macro_export]
macro_rules! log_event {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.event == true{
                debug!($($arg)+)
            }
        }
    };
}

#[macro_export]
macro_rules! log_init {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.init == true{
                debug!($($arg)+)
            }
        }
    };
}

#[macro_export]
macro_rules! log_ops {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.ops == true{
                debug!($($arg)+)
            }
        }
    };
}

#[macro_export]
macro_rules! log_prof {
    ( $($arg:tt)+) => {
        {
            use $crate::log::DEBUG_LOG_SETTINGS;
            use tracing::debug;
            if DEBUG_LOG_SETTINGS.prof == true{
                debug!($($arg)+)
            }
        }
    };
}
