use tonic::transport::Endpoint;

use std::{
    borrow::Cow, fs, fs::OpenOptions, io::Write, path::Path, str::FromStr, sync::Arc, time::Instant,
};

#[cfg(feature = "rsm-integration")]
use std::sync::{Mutex, OnceLock};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::{safetensor::ExpertKey, weight_manager::LocalWeightManager};
#[cfg(feature = "rsm-integration")]
use rsm_expert_kit::ExpertKitRsmBridge;
use tokio::sync::RwLock;

use crate::ffn::ExpertBackend;

use super::manager::ExpertDB;

/// Load expert task - fetches weight bytes from the LocalWeightManager,
/// builds the ExpertBackend, and inserts it into the shared ExpertDB.
///
/// Zero-copy: `wm.get_expert()` returns an `Arc<Bytes>` whose refcount is
/// incremented atomically. The bytes are borrowed by SafeTensors for the
/// duration of `ExpertBackend::build`, then the Arc is dropped here while
/// the WM retains its own reference for future callers.
pub async fn load_expert_task(
    weight_manager: Arc<LocalWeightManager>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    instance: crate::x::EKInstance,
    expert_key: &ExpertKey,
) -> EKResult<()> {
    let expert_str_key = expert_key.as_object_key();
    let total_started = Instant::now();

    // Mark expert as loading in shared database
    let mark_started = Instant::now();
    {
        let mut wg = expert_db.write().await;
        let result = wg.mark_loading(&expert_str_key);
        record_stage_timing(
            &expert_str_key,
            "mark_loading",
            mark_started.elapsed(),
            0,
            0,
            0,
            result.is_ok(),
        )?;
        result?;
    }

    // Fetch bytes and build backend within a scoped block so that
    // `bytes` (and thus `st`) are dropped before we take the expert_db write lock.
    // On any error, unmark_loading so the expert can be retried on the next update.
    let backend = match async {
        let get_started = Instant::now();
        let bytes = weight_manager.get_expert(expert_key).await?;
        let weight_bytes = bytes.len();
        record_stage_timing(
            &expert_str_key,
            "local_weight_manager_get_expert",
            get_started.elapsed(),
            0,
            weight_bytes,
            0,
            true,
        )?;

        let route_started = Instant::now();
        let routed = maybe_route_expert_bytes_through_rsm(&expert_str_key, bytes.as_ref())?;
        let route_stage = if routed.rsm_event_count > 0 {
            "rsm_acquire_release"
        } else {
            "rsm_bypass"
        };
        record_stage_timing(
            &expert_str_key,
            route_stage,
            route_started.elapsed(),
            weight_bytes,
            routed.bytes.len(),
            routed.rsm_event_count,
            true,
        )?;

        let deserialize_started = Instant::now();
        let st = safetensors::SafeTensors::deserialize(routed.bytes.as_ref())?;
        record_stage_timing(
            &expert_str_key,
            "safetensors_deserialize",
            deserialize_started.elapsed(),
            routed.bytes.len(),
            0,
            routed.rsm_event_count,
            true,
        )?;

        let build_started = Instant::now();
        let backend = ExpertBackend::build(instance, &st).await?;
        record_stage_timing(
            &expert_str_key,
            "expert_backend_build",
            build_started.elapsed(),
            routed.bytes.len(),
            0,
            routed.rsm_event_count,
            true,
        )?;
        Ok(backend)
        // `bytes` and `st` are dropped here
    }
    .await
    {
        Ok(b) => b,
        Err(e) => {
            let mut wg = expert_db.write().await;
            wg.unmark_loading(&expert_str_key);
            return Err(e);
        }
    };

    // Insert loaded expert into shared database
    let mut edb_wg = expert_db.write().await;
    let insert_started = Instant::now();
    let result = edb_wg.insert(&expert_str_key, backend).await;
    record_stage_timing(
        &expert_str_key,
        "expert_db_insert",
        insert_started.elapsed(),
        0,
        0,
        0,
        result.is_ok(),
    )?;
    result?;

    record_stage_timing(
        &expert_str_key,
        "load_expert_task_total",
        total_started.elapsed(),
        0,
        0,
        0,
        true,
    )?;

    Ok(())
}

struct RoutedExpertBytes<'a> {
    bytes: Cow<'a, [u8]>,
    rsm_event_count: usize,
}

fn maybe_route_expert_bytes_through_rsm<'a>(
    expert_str_key: &str,
    bytes: &'a [u8],
) -> EKResult<RoutedExpertBytes<'a>> {
    #[cfg(not(feature = "rsm-integration"))]
    let _ = expert_str_key;

    #[cfg(feature = "rsm-integration")]
    {
        if rsm_host_mode_enabled() {
            let routed = route_expert_bytes_through_rsm(expert_str_key, bytes)?;
            return Ok(RoutedExpertBytes {
                bytes: Cow::Owned(routed.bytes),
                rsm_event_count: routed.rsm_event_count,
            });
        }
    }
    Ok(RoutedExpertBytes {
        bytes: Cow::Borrowed(bytes),
        rsm_event_count: 0,
    })
}

#[cfg(feature = "rsm-integration")]
struct RsmRoutedBytes {
    bytes: Vec<u8>,
    rsm_event_count: usize,
}

#[cfg(feature = "rsm-integration")]
fn route_expert_bytes_through_rsm(expert_str_key: &str, bytes: &[u8]) -> EKResult<RsmRoutedBytes> {
    let bridge = get_rsm_bridge();
    let mut bridge = bridge.lock().map_err(|err| {
        ek_base::error::EKError::RuntimeError(format!(
            "RSM Expert-Kit bridge mutex poisoned while loading {expert_str_key}: {err}"
        ))
    })?;
    let load = bridge
        .load_host_bytes(expert_str_key, bytes.to_vec())
        .map_err(|err| {
            ek_base::error::EKError::RuntimeError(format!(
                "RSM Expert-Kit host load failed for {expert_str_key}: {err}"
            ))
        })?;
    if let Ok(path) = std::env::var("EK_RSM_EVENT_LOG") {
        rsm_expert_kit::append_events_jsonl(&path, &load.events).map_err(|err| {
            ek_base::error::EKError::RuntimeError(format!(
                "failed to append RSM Expert-Kit events to {path}: {err}"
            ))
        })?;
    }
    let acquire_line = format!(
        "[rsm] acquire object={} tier={:?} path={:?} lease={} events={}",
        load.object_id,
        load.selected_tier,
        load.access_path,
        load.lease_id.0,
        load.events.len()
    );
    let release_line = format!(
        "[rsm] release lease={} released={}",
        load.lease_id.0, load.lease_released
    );
    println!("{acquire_line}");
    println!("{release_line}");
    log::info!("{acquire_line}");
    log::info!("{release_line}");
    log::info!(
        "loaded expert bytes through RSM host bridge expert={} object_id={} access_path={:?} events={}",
        expert_str_key,
        load.object_id,
        load.access_path,
        load.events.len()
    );
    Ok(RsmRoutedBytes {
        bytes: load.returned_bytes,
        rsm_event_count: load.events.len(),
    })
}

fn record_stage_timing(
    expert_str_key: &str,
    stage: &str,
    elapsed: std::time::Duration,
    input_bytes: usize,
    output_bytes: usize,
    rsm_event_count: usize,
    success: bool,
) -> EKResult<()> {
    let Ok(path) = std::env::var("EK_RSM_STAGE_LOG") else {
        return Ok(());
    };
    let path = Path::new(&path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let needs_header = fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if needs_header {
        writeln!(
            file,
            "schema_version,mode,expert_key,stage,elapsed_ns,input_bytes,output_bytes,rsm_event_count,success"
        )?;
    }
    let mode = std::env::var("EK_RSM_WEIGHT_MODE").unwrap_or_else(|_| "off".to_string());
    writeln!(
        file,
        "1,{},{},{},{},{},{},{},{}",
        csv_field(&mode),
        csv_field(expert_str_key),
        csv_field(stage),
        elapsed.as_nanos(),
        input_bytes,
        output_bytes,
        rsm_event_count,
        success
    )?;
    Ok(())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(feature = "rsm-integration")]
fn get_rsm_bridge() -> &'static Mutex<ExpertKitRsmBridge> {
    static BRIDGE: OnceLock<Mutex<ExpertKitRsmBridge>> = OnceLock::new();
    BRIDGE.get_or_init(|| Mutex::new(ExpertKitRsmBridge::default()))
}

#[cfg(feature = "rsm-integration")]
fn rsm_host_mode_enabled() -> bool {
    is_rsm_host_mode(std::env::var("EK_RSM_WEIGHT_MODE").ok().as_deref())
}

#[cfg(feature = "rsm-integration")]
fn is_rsm_host_mode(value: Option<&str>) -> bool {
    matches!(value, Some("rsm-host"))
}

/// Get worker ID from settings
pub fn get_worker_id() -> String {
    let settings = get_ek_settings();
    settings.worker.id.clone()
}

/// Get controller endpoint from settings
pub fn get_controller_addr() -> Endpoint {
    let settings = get_ek_settings();
    let addr = format!(
        "http://{}:{}",
        settings.controller.broadcast, settings.controller.ports.intra
    );
    Endpoint::from_str(addr.as_str()).unwrap()
}

#[cfg(all(test, feature = "rsm-integration"))]
mod tests {
    use super::is_rsm_host_mode;

    #[test]
    fn rsm_host_mode_requires_explicit_runtime_opt_in() {
        assert!(is_rsm_host_mode(Some("rsm-host")));
        assert!(!is_rsm_host_mode(Some("off")));
        assert!(!is_rsm_host_mode(None));
    }
}
