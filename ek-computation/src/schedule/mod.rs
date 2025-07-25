pub mod profiler;
pub mod deployment;
pub mod scaler;

pub use profiler::{
    LayeredExpertProfiler, ExpertMetrics, 
    init_profiler, get_profiler, record_activation_completion, parse_expert_id
};
pub use deployment::{DeploymentCache, init_deployment_cache, get_deployment_cache};
pub use scaler::{AutoScaler, ScalingPlan, ScaleOutAction, ScaleDownAction};

use ek_base::{config::AutoScalingConfig, error::EKResult};

/// Initialize the schedule module
pub fn init_schedule_module(config: AutoScalingConfig) -> EKResult<()> {
    log::info!(
        "Auto scaling config loaded: enabled={}, topk_per_layer={}, memory_utilization={}, window_duration={}s",
        config.enabled,
        config.topk_per_layer, 
        config.memory_utilization,
        config.window_duration
    );
    
    if config.enabled {
        // Initialize profiler
        let _profiler = init_profiler(config);
        log::info!("Expert profiler initialized");
        log::info!("Schedule module framework initialized - auto scaling enabled");
    } else {
        log::info!("Schedule module framework initialized - auto scaling disabled");
    }
    
    Ok(())
}