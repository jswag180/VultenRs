use std::time::Instant;

use backend::GLOBAL_INSTANCES;

use crate::{
    profiler::{PROFILER_DATA, START_TIME},
    protos::generated::xplane::{
        XEvent, XEventMetadata, XLine, XPlane, XSpace, XStat, XStatMetadata,
    },
};

use super::profile_data::ProfileData;

#[inline]
fn get_start_offset_ps(start: Instant, op_start: Instant) -> i64 {
    (op_start.duration_since(start).as_nanos() * 1000) as i64
}

fn create_event(id: &i64, stat_meta_id: &mut i64, plane: &mut XPlane, pf_data: ProfileData) {
    let line = plane.lines.last_mut().unwrap();

    let mut event_meta = XEventMetadata::new();
    event_meta.id = *id;
    event_meta.name = pf_data.name;

    let mut event = XEvent::new();
    event.metadata_id = event_meta.id;
    event.set_offset_ps(get_start_offset_ps(
        START_TIME.read().unwrap().unwrap(),
        pf_data.start_time,
    ));
    event.duration_ps = pf_data.durration_ps;

    for (name, val) in pf_data.stats.iter() {
        let mut stat = XStat::new();
        stat.set_str_value(val.clone());
        stat.metadata_id = *stat_meta_id;
        event.stats.push(stat);

        let mut stat_meta = XStatMetadata::new();
        stat_meta.id = *stat_meta_id;
        stat_meta.name = name.clone();
        plane.stat_metadata.insert(*stat_meta_id, stat_meta);

        *stat_meta_id += 1;
    }

    plane.event_metadata.insert(*id, event_meta);
    line.events.push(event);
}

pub fn generate_xspace() -> XSpace {
    let mut space = XSpace::new();

    let total_devices = unsafe { GLOBAL_INSTANCES.read().unwrap() }.len();
    for dev in 0..total_devices {
        let mut plane = XPlane::new();
        plane.id = dev as i64;
        plane.name = format!("/device:GPU:{:}", dev);

        plane.lines.push(XLine::new());
        let line = plane.lines.last_mut().unwrap();
        line.id = 0;
        line.name = "PluginDevice stream".to_string();

        space.planes.push(plane);
    }

    let mut pf_data = PROFILER_DATA.lock().unwrap();
    let mut event_id = 0;
    let mut stat_meta_id = 0;
    while let Some(data) = pf_data.pop_front() {
        let plane = &mut space.planes[data.device_id as usize];

        create_event(&event_id, &mut stat_meta_id, plane, data);

        event_id += 1;
    }

    space
}
