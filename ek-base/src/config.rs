use std::{net::SocketAddr, path::Path, sync::LazyLock};

use config::{Config, Environment};
use once_cell::sync::OnceCell;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct Addr {
    pub host: String,
    pub port: u16,
}

impl Addr {
    pub fn to_socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port).parse().unwrap()
    }
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct InferenceSettings {
    pub instance_name: String,
    pub model_name: String,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct DBSettings {
    pub db_dsn: String,
    pub max_conn_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct AutoScalingConfig {
    /// Enable auto scaling feature
    pub enabled: bool,
    /// Time window duration in seconds for expert activation tracking
    pub window_duration: u64,
    /// Interval in seconds between scaling checks
    pub check_interval: u64,
    /// Number of top-K hot experts per layer to scale out
    pub topk_per_layer: usize,
    /// Scale down delay in seconds
    pub scale_down_delay: u64,
    /// Memory utilization threshold (0.0 - 1.0)
    pub memory_utilization: f64,
    /// Data type size in bytes (e.g., 2 for fp16)
    pub dtype_bytes: usize,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_duration: 30,
            check_interval: 30,
            topk_per_layer: 3,
            scale_down_delay: 6,
            memory_utilization: 0.8,
            dtype_bytes: 2,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct ControllerSettings {
    pub listen: String,
    pub broadcast: String,
    pub ports: ControllerPorts,
    #[serde(default)]
    pub auto_scaling: AutoScalingConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct ControllerPorts {
    pub intra: u16,
    pub inter: u16,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerPorts {
    pub main: u16,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct CpuAffinityConfig {
    // List of CPU core IDs to bind the worker to
    // Example: [0, 1, 2, 3] to bind to cores 0-3
    pub cores: Option<Vec<usize>>,
    // NUMA node IDs to bind the worker to
    // Example: [0] for NUMA node 0, [0, 1] for NUMA nodes 0 and 1
    pub numa_nodes: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerAdvancedSettings {
    // CPU affinity and NUMA configuration
    pub cpu_affinity: Option<CpuAffinityConfig>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerSettings {
    #[serde(default = "default_worker_id")]
    pub id: String,
    pub listen: String,
    pub broadcast: String,
    pub ports: WorkerPorts,
    pub device: String,
    #[serde(default = "default_worker_memory_gb")]
    pub memory_gb: f64,
    #[serde(default = "default_worker_metrics")]
    pub metrics: String,
    pub advanced: Option<WorkerAdvancedSettings>,
}

fn default_worker_memory_gb() -> f64 {
    16.0
}

fn default_worker_metrics() -> String {
    "0.0.0.0:9091".to_string()
}

fn default_worker_id() -> String {
    use gethostname::gethostname;
    gethostname().into_string().unwrap()
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightSettings {
    pub server: Option<WeightServerSettings>,
    pub cache: OpenDALStorage,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightServerSettings {
    pub addr: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct S3Config {
    pub access_key_id: String,
    pub access_key_secret: String,
    pub endpoint: String,
    pub region: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct FSConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub enum OpenDALStorage {
    Fs(FSConfig),
    S3(S3Config),
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct LogSettings {
    #[serde(default = "default_log_enable")]
    pub enable: bool,
    #[serde(default = "default_log_root")]
    pub root: String,
}
fn default_log_enable() -> bool {
    false
}

fn default_log_root() -> String {
    "/var/log/expert-kit".to_string()
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct Settings {
    pub inference: InferenceSettings,
    pub db: DBSettings,
    pub weight: WeightSettings,
    pub controller: ControllerSettings,
    pub worker: WorkerSettings,
}

pub fn env_source() -> Environment {
    static ENV_SRC: LazyLock<Environment> = std::sync::LazyLock::new(|| {
        Environment::with_prefix("EK")
            .try_parsing(false)
            .separator("_")
    });
    ENV_SRC.clone()
}
pub fn get_ek_settings_base(src: &[&str]) -> &'static Settings {
    static CONFIG: OnceCell<Settings> = OnceCell::new();

    (CONFIG.get_or_init(|| {
        let mut settings = Config::builder();
        let candidates = src.iter().chain(["/etc/expert-kit/config.yaml"].iter());

        for path in candidates {
            if Path::new(path).exists() {
                log::info!("Loading config from {path}");
                settings = settings.add_source(config::File::with_name(path));
                break;
            }
        }
        settings = settings.add_source(env_source());
        let settings = settings.build().unwrap();

        settings.try_deserialize::<Settings>().unwrap()
    })) as _
}

pub fn get_ek_settings() -> &'static Settings {
    get_ek_settings_base(&[])
}

#[cfg(test)]
mod test {
    use config::{File, FileFormat};

    use crate::config::env_source;

    use super::Settings;

    fn get_example_config() -> &'static str {
        r#"
inference:
  instance_name: qwen3_moe_30b_local_test
  model_name: ds-tiny
  hidden_dim: 2048
  intermediate_dim: 768
  
db:
  db_dsn: postgres://dev:dev@localhost:5432/dev
  max_conn_size: 32

weight:
  server:
    addr: http://?
  cache:
    Fs:
      path: /

worker:
  id: local_test
  listen: 0.0.0.0
  broadcast: 0.0.0.0
  ports:
    main: 51234
  device: cpu
  memory_gb: 16.0
  advanced:
    cpu_affinity:
      cores: [0, 1, 2, 3]
      numa_node: [0, 1]

controller:
  listen: 0.0.0.0
  broadcast: localhost
  ports:
    intra: 5001
    inter: 5002
  auto_scaling:
    enabled: true
    window_duration: 30
    check_interval: 30
    topk_per_layer: 3
    scale_down_delay: 6
    memory_utilization: 0.8
    dtype_bytes: 2
"#
    }

    #[test]
    fn basic_test() {
        let example_yaml = get_example_config();
        let config = config::Config::builder()
            .add_source(File::from_str(example_yaml, FileFormat::Yaml))
            .build()
            .unwrap();
        let res = config.try_deserialize::<Settings>().unwrap();
        assert_eq!(res.inference.hidden_dim, 2048);
        assert_eq!(res.worker.metrics, "0.0.0.0:9091");

        // Test advanced settings
        let advanced = res.worker.advanced.as_ref().unwrap();
        let cpu_affinity = advanced.cpu_affinity.as_ref().unwrap();
        assert_eq!(cpu_affinity.cores.as_ref().unwrap(), &vec![0, 1, 2, 3]);
        assert_eq!(cpu_affinity.numa_nodes.as_ref().unwrap(), &vec![0, 1]);
    }

    #[test]
    fn test_env_override() {
        let example_yaml = get_example_config();
        unsafe { std::env::set_var("EK_WORKER_ID", "override_test") };
        let config = config::Config::builder()
            .add_source(File::from_str(example_yaml, FileFormat::Yaml))
            .add_source(env_source())
            .build()
            .unwrap();
        let res = config.try_deserialize::<Settings>().unwrap();
        assert_eq!(res.worker.id, "override_test");
    }
}
