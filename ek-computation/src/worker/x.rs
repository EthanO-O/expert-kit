use tonic::transport::Endpoint;

use std::{borrow::Cow, str::FromStr, sync::Arc};

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

    // Mark expert as loading in shared database
    {
        let mut wg = expert_db.write().await;
        wg.mark_loading(&expert_str_key)?;
    }

    // Fetch bytes and build backend within a scoped block so that
    // `bytes` (and thus `st`) are dropped before we take the expert_db write lock.
    // On any error, unmark_loading so the expert can be retried on the next update.
    let backend = match async {
        let bytes = weight_manager.get_expert(expert_key).await?;
        let bytes = maybe_route_expert_bytes_through_rsm(&expert_str_key, bytes.as_ref())?;
        let st = safetensors::SafeTensors::deserialize(bytes.as_ref())?;
        ExpertBackend::build(instance, &st).await
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
    edb_wg.insert(&expert_str_key, backend).await?;

    Ok(())
}

fn maybe_route_expert_bytes_through_rsm<'a>(
    expert_str_key: &str,
    bytes: &'a [u8],
) -> EKResult<Cow<'a, [u8]>> {
    #[cfg(feature = "rsm-integration")]
    {
        if rsm_host_mode_enabled() {
            return route_expert_bytes_through_rsm(expert_str_key, bytes).map(Cow::Owned);
        }
    }
    Ok(Cow::Borrowed(bytes))
}

#[cfg(feature = "rsm-integration")]
fn route_expert_bytes_through_rsm(expert_str_key: &str, bytes: &[u8]) -> EKResult<Vec<u8>> {
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
    log::info!(
        "loaded expert bytes through RSM host bridge expert={} object_id={} access_path={:?} events={}",
        expert_str_key,
        load.object_id,
        load.access_path,
        load.events.len()
    );
    Ok(load.returned_bytes)
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
