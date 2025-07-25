use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;
use ek_base::{config::AutoScalingConfig, error::{EKError, EKResult}};
use crate::{
    state::{
        io::{StateReader, StateReaderImpl},
        models::{NewExpert, Node},
        writer::StateWriterImpl,
    },
};
use super::{
    profiler::{get_profiler, parse_expert_id},
    deployment::{get_deployment_cache, DeploymentCache},
};

/// Scale out action definition
#[derive(Debug, Clone)]
pub struct ScaleOutAction {
    pub expert_id: String,
    pub target_workers: Vec<String>,
    pub layer: u32,
}

/// Scale down action definition
#[derive(Debug, Clone)]
pub struct ScaleDownAction {
    pub expert_id: String,
    pub worker_id: String,
    pub grace_period: Duration,
    pub layer: u32,
}

/// Scaling plan
#[derive(Debug)]
pub struct ScalingPlan {
    pub scale_out: Vec<ScaleOutAction>,
    pub scale_down: Vec<ScaleDownAction>,
}

impl ScalingPlan {
    pub fn new() -> Self {
        Self {
            scale_out: Vec::new(),
            scale_down: Vec::new(),
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.scale_out.is_empty() && self.scale_down.is_empty()
    }
}

/// Auto scaler main implementation
pub struct AutoScaler {
    profiler: Arc<super::profiler::LayeredExpertProfiler>,
    deployment_cache: Arc<DeploymentCache>,
    state_reader: Box<dyn StateReader + Send + Sync>,
    state_writer: StateWriterImpl,  // 新增
    config: AutoScalingConfig,
    last_scaling: Instant,
}

impl AutoScaler {
    pub fn new(config: AutoScalingConfig) -> Self {
        log::info!("Initializing AutoScaler with check_interval={}s, topk_per_layer={}", 
                   config.check_interval, config.topk_per_layer);
        
        Self {
            profiler: get_profiler(),
            deployment_cache: get_deployment_cache(),
            state_reader: Box::new(StateReaderImpl::new()),
            state_writer: StateWriterImpl::new(),  // 新增
            config,
            last_scaling: Instant::now(),
        }
    }
    
    /// Main scaling loop - core function
    pub async fn run_scaling_loop(&mut self) -> EKResult<()> {
        let mut interval = time::interval(Duration::from_secs(self.config.check_interval));
        
        log::info!("Starting auto scaling loop with interval: {}s", self.config.check_interval);
        
        loop {
            interval.tick().await;
            
            log::info!("=== Auto Scaling Check Started ===");
            
            // Print statistics
            let stats = self.profiler.get_stats().await;
            log::info!("Current expert tracking stats: {:?}", stats);
            
            // Generate and execute scaling plan
            match self.make_scaling_decision().await {
                Ok(plan) => {
                    log::info!("Generated scaling plan: scale_out={}, scale_down={}", 
                             plan.scale_out.len(), plan.scale_down.len());
                    
                    if !plan.is_empty() {
                        self.log_scaling_plan(&plan).await;
                        
                        match self.execute_scaling_plan(plan).await {
                            Ok(_) => {
                                log::info!("Scaling plan executed successfully");
                                self.last_scaling = Instant::now();
                            }
                            Err(e) => {
                                log::error!("Failed to execute scaling plan: {}", e);
                                // Simple error handling: log and continue
                            }
                        }
                    } else {
                        log::debug!("No scaling actions needed");
                    }
                }
                Err(e) => {
                    log::error!("Failed to make scaling decision: {}", e);
                }
            }
            
            log::info!("=== Auto Scaling Check Completed ===");
        }
    }
    
    /// Generate scaling decision - core algorithm using worker-reported capacity
    pub async fn make_scaling_decision(&self) -> EKResult<ScalingPlan> {
        let mut plan = ScalingPlan::new();
        
        // Get all active nodes and capacity information
        let active_nodes = self.state_reader.active_nodes().await?;
        let worker_capacities = self.deployment_cache.get_all_worker_capacities().await;
        
        log::info!("Active nodes: {}, Worker capacities tracked: {}", 
                  active_nodes.len(), worker_capacities.len());
        
        // Calculate expert memory usage
        let settings = ek_base::config::get_ek_settings();
        let expert_memory_mb = self.deployment_cache.calculate_expert_memory_mb(
            settings.inference.hidden_dim,
            settings.inference.intermediate_dim,
            self.config.dtype_bytes
        );
        
        log::debug!("Expert memory calculation: {}MB per expert (3 matrices)", expert_memory_mb);
        
        // Get current expert deployment from memory cache
        let current_deployment = self.deployment_cache.get_all_deployments().await;
        
        // Get all model layers
        let stats = self.profiler.get_stats().await;
        let layers: Vec<u32> = stats.keys().cloned().collect();
        
        // Analyze each layer for TopK
        for &layer in &layers {
            log::info!("Analyzing layer {}", layer);
            
            // Get TopK hot experts for this layer
            let topk_experts = self.profiler
                .get_topk_hot_experts(layer, self.config.topk_per_layer).await;
            
            if topk_experts.is_empty() {
                log::debug!("No experts found for layer {}, skipping", layer);
                continue;
            }
            
            // Generate scale out plan for TopK experts
            for expert_id in &topk_experts {
                let current_workers = current_deployment.get(expert_id)
                    .map(|workers| workers.len())
                    .unwrap_or(0);
                
                log::debug!("Expert {} currently on {} workers", expert_id, current_workers);
                
                // Select nodes with sufficient capacity that don't have this expert deployed
                let deployed_workers: HashSet<String> = current_deployment
                    .get(expert_id)
                    .map(|workers| workers.iter().cloned().collect())
                    .unwrap_or_default();
                
                let mut available_workers = Vec::new();
                for node in &active_nodes {
                    // Skip workers that already have this expert
                    if deployed_workers.contains(&node.hostname) {
                        continue;
                    }
                    
                    // Check worker capacity
                    if let Some(capacity) = worker_capacities.get(&node.hostname) {
                        let available_slots = capacity.calculate_available_capacity(
                            expert_memory_mb, 
                            self.config.memory_utilization
                        );
                        
                        if available_slots > 0 {
                            available_workers.push(node.hostname.clone());
                            log::debug!("Worker {} has {} available slots for experts", 
                                       node.hostname, available_slots);
                        } else {
                            log::debug!("Worker {} has no available capacity", node.hostname);
                        }
                    } else {
                        log::warn!("No capacity info for worker {}, skipping", node.hostname);
                    }
                }
                
                if !available_workers.is_empty() {
                    log::info!("Scale out plan: expert={} -> workers={:?}", 
                              expert_id, available_workers);
                    
                    plan.scale_out.push(ScaleOutAction {
                        expert_id: expert_id.clone(),
                        target_workers: available_workers,
                        layer,
                    });
                }
            }
            
            // Generate scale down plan: experts not in TopK
            let deployed_experts_in_layer: Vec<String> = current_deployment
                .keys()
                .filter(|expert_id| {
                    let (_, expert_layer, _) = parse_expert_id(expert_id);
                    expert_layer == layer
                })
                .cloned()
                .collect();
            
            for expert_id in deployed_experts_in_layer {
                if !topk_experts.contains(&expert_id) {
                    // This expert is not in TopK, consider scale down
                    if let Some(workers) = current_deployment.get(&expert_id) {
                        // Keep at least one replica, scale down others
                        for (i, worker_id) in workers.iter().enumerate() {
                            if i > 0 { // Keep first replica
                                log::info!("Scale down plan: expert={} from worker={}", 
                                          expert_id, worker_id);
                                
                                plan.scale_down.push(ScaleDownAction {
                                    expert_id: expert_id.clone(),
                                    worker_id: worker_id.clone(),
                                    grace_period: Duration::from_secs(self.config.scale_down_delay),
                                    layer,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(plan)
    }
    
    /// Log scaling plan details
    async fn log_scaling_plan(&self, plan: &ScalingPlan) {
        log::info!("=== Detailed Scaling Plan ===");
        
        for action in &plan.scale_out {
            log::info!("SCALE OUT: expert={}, layer={}, targets={:?}", 
                      action.expert_id, action.layer, action.target_workers);
            
            // Log capacity information for target workers
            let worker_capacities = self.deployment_cache.get_all_worker_capacities().await;
            for worker in &action.target_workers {
                if let Some(capacity) = worker_capacities.get(worker) {
                    let settings = ek_base::config::get_ek_settings();
                    let expert_memory_mb = self.deployment_cache.calculate_expert_memory_mb(
                        settings.inference.hidden_dim,
                        settings.inference.intermediate_dim,
                        self.config.dtype_bytes
                    );
                    let available_slots = capacity.calculate_available_capacity(
                        expert_memory_mb, 
                        self.config.memory_utilization
                    );
                    log::info!("  -> Worker {}: {}GB total, {} experts loaded, {} slots available", 
                              worker, capacity.memory_gb, capacity.current_expert_count, available_slots);
                }
            }
        }
        
        for action in &plan.scale_down {
            log::info!("SCALE DOWN: expert={}, layer={}, worker={}, grace_period={:?}", 
                      action.expert_id, action.layer, action.worker_id, action.grace_period);
        }
        
        if plan.is_empty() {
            log::info!("No scaling actions needed - system is optimally configured");
        }
        
        log::info!("=== End Scaling Plan ===");
    }
    
    /// Execute scaling plan
    pub async fn execute_scaling_plan(&self, plan: ScalingPlan) -> EKResult<()> {
        log::info!("Executing scaling plan with {} scale-out and {} scale-down actions", 
                   plan.scale_out.len(), plan.scale_down.len());
        
        // Execute scale out operations
        for action in &plan.scale_out {
            log::info!("Executing scale out: expert={}, workers={:?}", 
                      action.expert_id, action.target_workers);
            
            for worker_hostname in &action.target_workers {
                match self.deploy_expert_to_worker(&action.expert_id, worker_hostname).await {
                    Ok(_) => {
                        // Update memory cache
                        self.deployment_cache.add_expert_to_worker(&action.expert_id, worker_hostname).await;
                        log::info!("Successfully deployed expert {} to worker {}", 
                                  action.expert_id, worker_hostname);
                    }
                    Err(e) => {
                        log::error!("Failed to deploy expert {} to worker {}: {}", 
                                  action.expert_id, worker_hostname, e);
                        // Simple error handling: log and continue
                    }
                }
            }
        }
        
        // Execute scale down operations  
        for action in &plan.scale_down {
            log::info!("Executing scale down: expert={}, worker={}", 
                      action.expert_id, action.worker_id);
            
            match self.scale_down_expert(&action.expert_id, &action.worker_id, action.grace_period).await {
                Ok(_) => {
                    // Update memory cache
                    self.deployment_cache.remove_expert_from_worker(&action.expert_id, &action.worker_id).await;
                    log::info!("Successfully scaled down expert {} from worker {}", 
                              action.expert_id, action.worker_id);
                }
                Err(e) => {
                    log::error!("Failed to scale down expert {} from worker {}: {}", 
                              action.expert_id, action.worker_id, e);
                    // Simple error handling: log and continue
                }
            }
        }
        
        Ok(())
    }
    
    /// Deploy expert to worker - simplified database operation
    async fn deploy_expert_to_worker(&self, expert_id: &str, worker_hostname: &str) -> EKResult<()> {
        let settings = ek_base::config::get_ek_settings();
        
        // Get worker node information
        let nodes = self.state_reader.active_nodes().await?;
        let target_node = nodes.iter()
            .find(|node| node.hostname == worker_hostname)
            .ok_or_else(|| EKError::NotFound(format!("Worker {} not found", worker_hostname)))?;
        
        // Get instance information
        let instance = self.state_reader
            .instance_by_name(&settings.inference.instance_name).await?
            .ok_or_else(|| EKError::NotFound("Instance not found".to_string()))?;
        
        // Create expert record in database
        let new_expert = NewExpert {
            instance_id: instance.id,
            node_id: target_node.id,
            expert_id: expert_id.to_string(),
            replica: 1,
            state: serde_json::json!({}),
        };
        
        self.state_writer.expert_upsert(new_expert).await?;
        
        log::info!("Database record created: expert={}, node={}", expert_id, worker_hostname);
        Ok(())
    }
    
    /// Scale down expert from worker - simplified operation
    async fn scale_down_expert(&self, expert_id: &str, worker_hostname: &str, grace_period: Duration) -> EKResult<()> {
        log::info!("Scaling down expert {} from worker {} with grace period {:?}", 
                  expert_id, worker_hostname, grace_period);
        
        tokio::time::sleep(grace_period).await;
        
        // TODO: In future phases, implement:
        // 1. Send RPC to worker for graceful shutdown
        // 2. Remove database record
        // For now, just log the operation
        
        log::info!("Grace period completed for expert {} on worker {}", 
                  expert_id, worker_hostname);
        log::info!("Expert {} should be unloaded from worker {} (simulated)", 
                  expert_id, worker_hostname);
        
        Ok(())
    }
}