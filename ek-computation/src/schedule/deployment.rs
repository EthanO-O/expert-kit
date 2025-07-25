use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use ek_base::error::EKResult;
use crate::state::{io::StateReaderImpl, models::{Expert, Node}};

/// Worker capacity information
#[derive(Debug, Clone)]
pub struct WorkerCapacity {
    pub hostname: String,
    pub memory_gb: f64,
    pub current_expert_count: usize,
    pub last_updated: u64,  // unix timestamp
}

impl WorkerCapacity {
    /// Calculate available capacity for this worker
    pub fn calculate_available_capacity(&self, expert_memory_mb: f64, memory_utilization: f64) -> usize {
        let available_memory_mb = self.memory_gb * 1024.0 * memory_utilization;
        let expert_memory_with_overhead = expert_memory_mb * 1.2; // 20% overhead
        let max_experts = (available_memory_mb / expert_memory_with_overhead) as usize;
        max_experts.saturating_sub(self.current_expert_count)
    }
}

/// Expert deployment status cache + Worker capacity management
#[derive(Debug, Clone)]
pub struct DeploymentCache {
    // expert_id -> worker_hostnames
    deployments: Arc<RwLock<HashMap<String, Vec<String>>>>,
    // worker_hostname -> expert_ids
    worker_experts: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    // worker_hostname -> capacity_info
    worker_capacities: Arc<RwLock<HashMap<String, WorkerCapacity>>>,
    last_sync: Arc<tokio::sync::Mutex<std::time::Instant>>,
}

impl DeploymentCache {
    pub fn new() -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
            worker_experts: Arc::new(RwLock::new(HashMap::new())),
            worker_capacities: Arc::new(RwLock::new(HashMap::new())),
            last_sync: Arc::new(tokio::sync::Mutex::new(std::time::Instant::now())),
        }
    }
    
    /// Update worker capacity information - reported via StateService
    pub async fn update_worker_capacity(&self, hostname: &str, memory_gb: f64, current_expert_count: usize) {
        let mut capacities_guard = self.worker_capacities.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        capacities_guard.insert(hostname.to_string(), WorkerCapacity {
            hostname: hostname.to_string(),
            memory_gb,
            current_expert_count,
            last_updated: now,
        });
        
        log::info!("Worker capacity updated: {} - {}GB, {} experts", 
                   hostname, memory_gb, current_expert_count);
    }
    
    /// Get worker capacity information
    pub async fn get_worker_capacity(&self, hostname: &str) -> Option<WorkerCapacity> {
        let capacities_guard = self.worker_capacities.read().await;
        capacities_guard.get(hostname).cloned()
    }
    
    /// Get all worker capacity information
    pub async fn get_all_worker_capacities(&self) -> HashMap<String, WorkerCapacity> {
        let capacities_guard = self.worker_capacities.read().await;
        capacities_guard.clone()
    }
    
    /// Calculate expert memory usage
    pub fn calculate_expert_memory_mb(&self, hidden_dim: usize, intermediate_dim: usize, dtype_bytes: usize) -> f64 {
        // Each expert contains 3 weight matrices: up_proj, down_proj, gate_proj
        (hidden_dim * intermediate_dim * dtype_bytes * 3) as f64 / (1024.0 * 1024.0)
    }
    
    /// Sync deployment state from database - every 60s
    pub async fn sync_from_database(&self) -> EKResult<()> {
        let settings = ek_base::config::get_ek_settings();
        let reader = StateReaderImpl::new();
        
        log::debug!("Syncing deployment state from database...");
        
        // Get all experts for current instance
        if let Some(instance) = reader.instance_by_name(&settings.inference.instance_name).await? {
            let experts = reader.experts_by_instance(instance.id).await?;
            let nodes = reader.active_nodes().await?;
            
            // Create node_id to hostname mapping
            let node_map: HashMap<i32, String> = nodes.iter()
                .map(|node| (node.id, node.hostname.clone()))
                .collect();
            
            // Rebuild cache
            let mut deployments_guard = self.deployments.write().await;
            let mut worker_experts_guard = self.worker_experts.write().await;
            
            deployments_guard.clear();
            worker_experts_guard.clear();
            
            for expert in experts {
                if let Some(worker_hostname) = node_map.get(&expert.node_id) {
                    // Update expert_id -> worker_hostnames mapping
                    deployments_guard.entry(expert.expert_id.clone())
                        .or_default()
                        .push(worker_hostname.clone());
                    
                    // Update worker_hostname -> expert_ids mapping
                    worker_experts_guard.entry(worker_hostname.clone())
                        .or_default()
                        .insert(expert.expert_id.clone());
                }
            }
            
            log::info!("Synced {} expert deployments across {} workers", 
                      deployments_guard.len(), worker_experts_guard.len());
        }
        
        *self.last_sync.lock().await = std::time::Instant::now();
        Ok(())
    }
    
    /// Get all expert deployments
    pub async fn get_all_deployments(&self) -> HashMap<String, Vec<String>> {
        let deployments_guard = self.deployments.read().await;
        deployments_guard.clone()
    }
    
    /// Add expert to worker (after scale out)
    pub async fn add_expert_to_worker(&self, expert_id: &str, worker_hostname: &str) {
        let mut deployments_guard = self.deployments.write().await;
        let mut worker_experts_guard = self.worker_experts.write().await;
        
        deployments_guard.entry(expert_id.to_string())
            .or_default()
            .push(worker_hostname.to_string());
        
        worker_experts_guard.entry(worker_hostname.to_string())
            .or_default()
            .insert(expert_id.to_string());
        
        // Update worker's current expert count
        let mut capacities_guard = self.worker_capacities.write().await;
        if let Some(capacity) = capacities_guard.get_mut(worker_hostname) {
            capacity.current_expert_count += 1;
        }
        
        log::debug!("Added expert {} to worker {} in cache", expert_id, worker_hostname);
    }
    
    /// Remove expert from worker (after scale down)
    pub async fn remove_expert_from_worker(&self, expert_id: &str, worker_hostname: &str) {
        let mut deployments_guard = self.deployments.write().await;
        let mut worker_experts_guard = self.worker_experts.write().await;
        
        if let Some(workers) = deployments_guard.get_mut(expert_id) {
            workers.retain(|w| w != worker_hostname);
            if workers.is_empty() {
                deployments_guard.remove(expert_id);
            }
        }
        
        if let Some(experts) = worker_experts_guard.get_mut(worker_hostname) {
            experts.remove(expert_id);
            if experts.is_empty() {
                worker_experts_guard.remove(worker_hostname);
            }
        }
        
        // Update worker's current expert count
        let mut capacities_guard = self.worker_capacities.write().await;
        if let Some(capacity) = capacities_guard.get_mut(worker_hostname) {
            capacity.current_expert_count = capacity.current_expert_count.saturating_sub(1);
        }
        
        log::debug!("Removed expert {} from worker {} in cache", expert_id, worker_hostname);
    }
    
    /// Background sync loop
    pub async fn run_sync_loop(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60)); // 60s sync
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.sync_from_database().await {
                log::error!("Failed to sync deployment state from database: {}", e);
            }
        }
    }
}

// Global deployment cache instance
use once_cell::sync::OnceCell;

static DEPLOYMENT_CACHE: OnceCell<Arc<DeploymentCache>> = OnceCell::new();

pub fn init_deployment_cache() -> Arc<DeploymentCache> {
    let cache = Arc::new(DeploymentCache::new());
    DEPLOYMENT_CACHE.set(cache.clone()).expect("DeploymentCache already initialized");
    cache
}

pub fn get_deployment_cache() -> Arc<DeploymentCache> {
    DEPLOYMENT_CACHE.get().expect("DeploymentCache not initialized").clone()
}