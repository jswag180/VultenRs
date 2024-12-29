use std::time::Instant;

use super::{IS_PROFILING, PROFILER_DATA};

#[macro_export]
macro_rules! profile {
    ($arg:expr, $arg1:expr) => {{
        use $crate::profiler::{profile_data::ScopeProfiler, IS_PROFILING};

        if IS_PROFILING.load(std::sync::atomic::Ordering::SeqCst) {
            Some(ScopeProfiler::start($arg, $arg1, true))
        } else {
            None
        }
    }};
}

#[macro_export]
macro_rules! profile_add_stat {
    ($prof:expr, $name:expr, $val:expr) => {{
        if let Some(p) = $prof.as_mut() {
            p.add_stat($name, $val);
        }
    }};
}

#[derive(Debug)]
pub struct ScopeProfiler {
    name: String,
    device_id: u64,
    start: Instant,
    stop: Option<Instant>,
    report: bool,
    stats: Vec<(String, String)>,
}

impl ScopeProfiler {
    pub fn start(name: String, device_id: u64, report: bool) -> Self {
        Self {
            name,
            device_id,
            start: Instant::now(),
            stop: None,
            report,
            stats: Vec::new(),
        }
    }

    pub fn stop(&mut self) {
        self.stop = Some(Instant::now());
    }

    pub fn duration_as_ps(&self) -> Result<i64, &str> {
        if let Some(stop) = self.stop.as_ref() {
            Ok((stop.duration_since(self.start).as_nanos() * 1000) as i64)
        } else {
            Err("Timer has not stoped!")
        }
    }

    pub fn add_stat(&mut self, name: String, val: String) {
        self.stats.push((name, val));
    }

    pub fn generate_report(&self) -> ProfileData {
        ProfileData {
            name: self.name.clone(),
            device_id: self.device_id,
            start_time: self.start,
            durration_ps: self.duration_as_ps().unwrap(),
            stats: self.stats.clone(),
        }
    }
}

impl Drop for ScopeProfiler {
    fn drop(&mut self) {
        self.stop();
        if self.report && IS_PROFILING.load(std::sync::atomic::Ordering::SeqCst) {
            let mut pf_data = PROFILER_DATA.lock().unwrap();
            pf_data.push_back(self.generate_report());
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProfileData {
    pub name: String,
    pub device_id: u64,
    pub start_time: Instant,
    pub durration_ps: i64,
    pub stats: Vec<(String, String)>,
}
