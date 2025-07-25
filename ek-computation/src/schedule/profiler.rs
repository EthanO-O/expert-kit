use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use once_cell::sync::OnceCell;
use ek_base::{config::AutoScalingConfig};

/// Expert comprehensive metrics structure
#[derive(Debug)]
pub struct ExpertMetrics {
    total_activations: AtomicU64,      // Total activation count
    recent_activations: AtomicU64,     // Activations in current window  
    last_activation_time: AtomicU64,   // Last activation timestamp (epoch seconds)
    window_start_time: AtomicU64,      // Current window start time
    avg_batch_size: AtomicU64,         // Average batch size (stored * 1000)
}

impl ExpertMetrics {
    pub fn new() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            total_activations: AtomicU64::new(0),
            recent_activations: AtomicU64::new(0),
            last_activation_time: AtomicU64::new(now),
            window_start_time: AtomicU64::new(now),
            avg_batch_size: AtomicU64::new(0),
        }
    }
    
    /// Lock-free activation recording - called on request completion
    pub fn record_activation_completion(&self, count: u64, _completion_time_ms: u64) {
        let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        self.total_activations.fetch_add(count, Ordering::Relaxed);
        self.recent_activations.fetch_add(count, Ordering::Relaxed);
        self.last_activation_time.store(now_secs, Ordering::Relaxed);
        
        // Exponential smoothing for average batch size (α=0.125)
        let old_avg = self.avg_batch_size.load(Ordering::Relaxed);
        let new_avg = (old_avg * 7 + count * 1000) / 8;
        self.avg_batch_size.store(new_avg, Ordering::Relaxed);
    }
    
    /// Calculate heat score - core algorithm
    pub fn get_heat_score(&self, now_secs: u64, window_duration: u64) -> f64 {
        let last_activation = self.last_activation_time.load(Ordering::Relaxed);
        let recent_count = self.recent_activations.load(Ordering::Relaxed);
        let avg_batch = self.avg_batch_size.load(Ordering::Relaxed) as f64 / 1000.0;
        
        // Time decay factor calculation
        let age = now_secs.saturating_sub(last_activation);
        let time_factor = if age > window_duration * 2 {
            0.1  // Very old, low heat
        } else if age > window_duration {
            0.5  // Somewhat old, half heat  
        } else {
            1.0  // Recent activation, full heat
        };
        
        // Comprehensive score: frequency × time_factor × batch_weight
        let batch_weight = (avg_batch / 10.0).min(2.0).max(0.5); // Weight between 0.5-2.0
        recent_count as f64 * time_factor * batch_weight
    }
    
    /// Reset window - called every window period
    pub fn reset_window(&self, new_window_start: u64) {
        self.recent_activations.store(0, Ordering::Relaxed);
        self.window_start_time.store(new_window_start, Ordering::Relaxed);
    }
    
    /// Get statistics for monitoring
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let total = self.total_activations.load(Ordering::Relaxed);
        let recent = self.recent_activations.load(Ordering::Relaxed);
        let avg_size = self.avg_batch_size.load(Ordering::Relaxed) as f64 / 1000.0;
        (total, recent, avg_size)
    }
}

/// Layered expert profiler
#[derive(Debug)]
pub struct LayeredExpertProfiler {
    // layer_id -> expert_id -> metrics
    layers: Arc<RwLock<HashMap<u32, HashMap<String, Arc<ExpertMetrics>>>>>,
    config: AutoScalingConfig,
    last_window_reset: AtomicU64,
    last_cleanup: AtomicU64,
}

impl LayeredExpertProfiler {
    pub fn new(config: AutoScalingConfig) -> Self {
        log::info!("Initializing LayeredExpertProfiler with window_duration={}s", config.window_duration);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        Self {
            layers: Arc::new(RwLock::new(HashMap::new())),
            config,
            last_window_reset: AtomicU64::new(now),
            last_cleanup: AtomicU64::new(now),
        }
    }
    
    /// High-performance activation recording - called on request completion
    pub fn record_completion(&self, expert_id: &str, batch_size: usize, completion_time_ms: u64) {
        let (_, layer, _) = parse_expert_id(expert_id);
        
        // Try fast path: read-only lock to find existing metrics
        if let Ok(layers_guard) = self.layers.try_read() {
            if let Some(layer_experts) = layers_guard.get(&layer) {
                if let Some(metrics) = layer_experts.get(expert_id) {
                    // Found! Record directly, completely lock-free
                    metrics.record_activation_completion(batch_size as u64, completion_time_ms);
                    log::debug!("Recording activation: expert={}, batch_size={}, completion_time={}ms", 
                               expert_id, batch_size, completion_time_ms);
                    return;
                }
            }
        }
        
        // Fallback path: need to create new metrics 
        self.record_completion_slow(expert_id, layer, batch_size, completion_time_ms);
    }
    
    /// Create new expert metrics fallback path
    fn record_completion_slow(&self, expert_id: &str, layer: u32, batch_size: usize, completion_time_ms: u64) {
        let mut layers_guard = self.layers.write().unwrap();
        let layer_experts = layers_guard.entry(layer).or_default();
        let metrics = layer_experts.entry(expert_id.to_string()).or_insert_with(|| {
            log::debug!("Creating new metrics for expert: {}", expert_id);
            Arc::new(ExpertMetrics::new())
        });
        
        metrics.record_activation_completion(batch_size as u64, completion_time_ms);
        log::debug!("Recording activation: expert={}, batch_size={}, completion_time={}ms", 
                   expert_id, batch_size, completion_time_ms);
    }
    
    /// Get TopK hot experts - called by AutoScaler
    pub async fn get_topk_hot_experts(&self, layer: u32, k: usize) -> Vec<String> {
        let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        let layers_guard = self.layers.read().unwrap();
        let layer_experts = match layers_guard.get(&layer) {
            Some(experts) => experts,
            None => {
                log::debug!("No experts found for layer {}", layer);
                return vec![];
            }
        };
        
        let mut expert_scores: Vec<(String, f64)> = layer_experts
            .iter()
            .map(|(id, metrics)| {
                let score = metrics.get_heat_score(now_secs, self.config.window_duration);
                (id.clone(), score)
            })
            .collect();
        
        // Sort by heat score descending
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let topk: Vec<String> = expert_scores
            .into_iter()
            .take(k)
            .map(|(id, score)| {
                log::debug!("Expert {} heat score: {:.2}", id, score);
                id
            })
            .collect();
        
        log::info!("Layer {} TopK experts: {:?}", layer, topk);
        topk
    }
    
    /// Reset windows periodically
    pub async fn reset_windows_if_needed(&self) {
        let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let last_reset = self.last_window_reset.load(Ordering::Relaxed);
        
        if now_secs - last_reset >= self.config.window_duration {
            log::info!("Resetting activation windows");
            
            let layers_guard = self.layers.read().unwrap();
            for layer_experts in layers_guard.values() {
                for metrics in layer_experts.values() {
                    metrics.reset_window(now_secs);
                }
            }
            
            self.last_window_reset.store(now_secs, Ordering::Relaxed);
        }
    }
    
    /// Get statistics for monitoring and debugging
    pub async fn get_stats(&self) -> HashMap<u32, (usize, u64, f64)> {
        let layers_guard = self.layers.read().unwrap();
        let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        layers_guard.iter()
            .map(|(&layer, experts)| {
                let expert_count = experts.len();
                let (total_activations, _recent_activations): (u64, u64) = experts
                    .values()
                    .map(|metrics| {
                        let (total, recent, _) = metrics.get_stats();
                        (total, recent)
                    })
                    .fold((0, 0), |(acc_total, acc_recent), (total, recent)| {
                        (acc_total + total, acc_recent + recent)
                    });
                
                let avg_heat = if expert_count > 0 {
                    experts.values()
                        .map(|metrics| metrics.get_heat_score(now_secs, self.config.window_duration))
                        .sum::<f64>() / expert_count as f64
                } else {
                    0.0
                };
                
                (layer, (expert_count, total_activations, avg_heat))
            })
            .collect()
    }
    
    /// Background maintenance loop
    pub async fn run_maintenance_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30)); // 30s maintenance interval
        
        loop {
            interval.tick().await;
            
            // Reset windows
            self.reset_windows_if_needed().await;
            
            // Print statistics
            let stats = self.get_stats().await;
            for (layer, (count, total, avg_heat)) in stats {
                log::info!("Layer stats: layer={}, experts={}, total_activations={}, avg_heat={:.2}", 
                           layer, count, total, avg_heat);
            }
        }
    }
}

/// Parse expert ID function
pub fn parse_expert_id(expert_id: &str) -> (String, u32, u32) {
    // Format: {model_name}/l{layer}-e{expert}
    // Example: "qwen3/l12-e32"
    let parts: Vec<&str> = expert_id.split('/').collect();
    if parts.len() != 2 {
        log::warn!("Invalid expert_id format: {}", expert_id);
        return ("unknown".to_string(), 0, 0);
    }
    
    let model_name = parts[0].to_string();
    let layer_expert = parts[1];
    
    // Parse l{layer}-e{expert}
    if let Some(dash_pos) = layer_expert.find('-') {
        let layer_part = &layer_expert[1..dash_pos]; // Skip 'l'
        let expert_part = &layer_expert[dash_pos+2..]; // Skip '-e'
        
        let layer = layer_part.parse().unwrap_or(0);
        let expert = expert_part.parse().unwrap_or(0);
        
        (model_name, layer, expert)
    } else {
        log::warn!("Invalid layer-expert format: {}", layer_expert);
        (model_name, 0, 0)
    }
}

// Global Profiler instance management
static PROFILER: OnceCell<Arc<LayeredExpertProfiler>> = OnceCell::new();

pub fn init_profiler(config: AutoScalingConfig) -> Arc<LayeredExpertProfiler> {
    let profiler = Arc::new(LayeredExpertProfiler::new(config));
    PROFILER.set(profiler.clone()).expect("Profiler already initialized");
    profiler
}

pub fn get_profiler() -> Arc<LayeredExpertProfiler> {
    PROFILER.get().expect("Profiler not initialized").clone()
}

/// High-performance activation completion recording API
pub fn record_activation_completion(expert_id: &str, batch_size: usize, completion_time_ms: u64) {
    if let Some(profiler) = PROFILER.get() {
        profiler.record_completion(expert_id, batch_size, completion_time_ms);
    }
}