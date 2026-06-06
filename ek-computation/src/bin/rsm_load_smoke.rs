use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use ek_base::{
    config::{FSConfig, OpenDALStorage},
    error::{EKError, EKResult},
};
use ek_computation::{
    backend::Device,
    worker::{get_expert_db, x::load_expert_task},
    x::{EKInstance, ExpertBackendType},
};
use ek_db::{dal::op_from_settings, safetensor::ExpertKey, weight_manager::LocalWeightManager};
use safetensors::{Dtype, tensor::TensorView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeMode {
    Baseline,
    RsmHost,
}

impl SmokeMode {
    fn as_str(self) -> &'static str {
        match self {
            SmokeMode::Baseline => "baseline",
            SmokeMode::RsmHost => "rsm-host",
        }
    }

    fn env_value(self) -> &'static str {
        match self {
            SmokeMode::Baseline => "off",
            SmokeMode::RsmHost => "rsm-host",
        }
    }
}

#[derive(Debug)]
struct Options {
    mode: SmokeMode,
    event_log: Option<PathBuf>,
    stage_log: Option<PathBuf>,
    out_dir: PathBuf,
    cache_dir: PathBuf,
    repeat: usize,
    hidden: usize,
    intermediate: usize,
}

#[derive(Debug)]
struct SmokeOutcome {
    mode: SmokeMode,
    expert_key: String,
    cache_dir: PathBuf,
    event_log: Option<PathBuf>,
    stage_log: Option<PathBuf>,
    new_event_count: usize,
    new_stage_rows: usize,
    loaded_count: usize,
    repeat: usize,
    hidden: usize,
    intermediate: usize,
    weight_bytes: usize,
    total_elapsed_ms: u128,
    median_elapsed_us: u128,
    p95_elapsed_us: u128,
    report_path: PathBuf,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> EKResult<()> {
    init_logger();
    let options = parse_options()?;

    // Rust 2024 makes process environment mutation unsafe because other threads
    // may read it concurrently. This smoke binary sets env vars before it starts
    // any worker tasks and then uses a tiny Tokio runtime.
    unsafe {
        std::env::set_var("EK_RSM_WEIGHT_MODE", options.mode.env_value());
        if let Some(event_log) = options.event_log.as_ref() {
            std::env::set_var("EK_RSM_EVENT_LOG", event_log);
        } else {
            std::env::remove_var("EK_RSM_EVENT_LOG");
        }
        if let Some(stage_log) = options.stage_log.as_ref() {
            std::env::set_var("EK_RSM_STAGE_LOG", stage_log);
        } else {
            std::env::remove_var("EK_RSM_STAGE_LOG");
        }
    }

    let outcome = run_smoke(options).await?;
    println!(
        "[expert-loader] loaded {} mode={} loaded_count={} new_rsm_events={} new_stage_rows={} elapsed_ms={}",
        outcome.expert_key,
        outcome.mode.as_str(),
        outcome.loaded_count,
        outcome.new_event_count,
        outcome.new_stage_rows,
        outcome.total_elapsed_ms
    );
    println!(
        "[expert-loader] report written to {}",
        outcome.report_path.display()
    );
    Ok(())
}

async fn run_smoke(options: Options) -> EKResult<SmokeOutcome> {
    fs::create_dir_all(&options.out_dir)?;
    fs::create_dir_all(options.cache_dir.join("toy-moe"))?;

    let expert_key = ExpertKey::new("toy-moe".to_string(), 0, 1);
    let expert_object_key = expert_key.as_object_key();
    let weight_bytes = tiny_expert_safetensors(options.hidden, options.intermediate)?;
    seed_cache(&options.cache_dir, &expert_key, &weight_bytes).await?;

    println!(
        "[expert-loader] {} mode: EK_RSM_WEIGHT_MODE={}",
        options.mode.as_str(),
        options.mode.env_value()
    );
    println!(
        "[expert-loader] seeded tiny SafeTensors expert {} bytes={} cache={}",
        expert_object_key,
        weight_bytes.len(),
        options.cache_dir.display()
    );

    let event_count_before = options
        .event_log
        .as_ref()
        .map(|path| count_lines(path))
        .transpose()?
        .unwrap_or(0);
    let stage_count_before = options
        .stage_log
        .as_ref()
        .map(|path| count_data_rows(path))
        .transpose()?
        .unwrap_or(0);

    let wm = make_weight_manager(&options.cache_dir);
    let expert_db = get_expert_db();
    let instance = EKInstance {
        hidden: options.hidden,
        intermediate: options.intermediate,
        backend: ExpertBackendType::Ggml,
        device: Device::CPU,
    };

    let total_started = Instant::now();
    let mut per_run_elapsed_us = Vec::with_capacity(options.repeat);
    for _ in 0..options.repeat {
        let started = Instant::now();
        load_expert_task(wm.clone(), expert_db.clone(), instance, &expert_key).await?;
        per_run_elapsed_us.push(started.elapsed().as_micros());
    }
    let total_elapsed_ms = total_started.elapsed().as_millis();

    let loaded_count = {
        let guard = expert_db.read().await;
        guard.loaded()
    };
    let event_count_after = options
        .event_log
        .as_ref()
        .map(|path| count_lines(path))
        .transpose()?
        .unwrap_or(0);
    let new_event_count = event_count_after.saturating_sub(event_count_before);
    let stage_count_after = options
        .stage_log
        .as_ref()
        .map(|path| count_data_rows(path))
        .transpose()?
        .unwrap_or(0);
    let new_stage_rows = stage_count_after.saturating_sub(stage_count_before);

    if options.mode == SmokeMode::RsmHost {
        println!(
            "[rsm] event-log={} new_events={}",
            options
                .event_log
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string()),
            new_event_count
        );
    }

    let report_path = options.out_dir.join("expert_load_smoke_report.md");
    let outcome = SmokeOutcome {
        mode: options.mode,
        expert_key: expert_object_key,
        cache_dir: options.cache_dir,
        event_log: options.event_log,
        stage_log: options.stage_log,
        new_event_count,
        new_stage_rows,
        loaded_count,
        repeat: options.repeat,
        hidden: options.hidden,
        intermediate: options.intermediate,
        weight_bytes: weight_bytes.len(),
        total_elapsed_ms,
        median_elapsed_us: percentile(&per_run_elapsed_us, 50),
        p95_elapsed_us: percentile(&per_run_elapsed_us, 95),
        report_path,
    };
    fs::write(&outcome.report_path, smoke_report_markdown(&outcome))?;
    Ok(outcome)
}

fn init_logger() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .try_init();
}

fn parse_options() -> EKResult<Options> {
    let mut mode = SmokeMode::Baseline;
    let mut event_log = None;
    let mut stage_log = None;
    let mut out_dir = PathBuf::from("target/rsm-load-smoke");
    let mut cache_dir = None;
    let mut repeat = 1usize;
    let mut hidden = 4usize;
    let mut intermediate = 3usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --mode".to_string())
                })?;
                mode = parse_mode(&value)?;
            }
            "--event-log" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --event-log".to_string())
                })?;
                event_log = Some(PathBuf::from(value));
            }
            "--stage-log" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --stage-log".to_string())
                })?;
                stage_log = Some(PathBuf::from(value));
            }
            "--out" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --out".to_string())
                })?;
                out_dir = PathBuf::from(value);
            }
            "--cache-dir" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --cache-dir".to_string())
                })?;
                cache_dir = Some(PathBuf::from(value));
            }
            "--repeat" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --repeat".to_string())
                })?;
                repeat = value.parse()?;
                if repeat == 0 {
                    return Err(EKError::InvalidInput("--repeat must be > 0".to_string()));
                }
            }
            "--hidden" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --hidden".to_string())
                })?;
                hidden = value.parse()?;
                if hidden == 0 {
                    return Err(EKError::InvalidInput("--hidden must be > 0".to_string()));
                }
            }
            "--intermediate" => {
                let value = args.next().ok_or_else(|| {
                    EKError::InvalidInput("missing value after --intermediate".to_string())
                })?;
                intermediate = value.parse()?;
                if intermediate == 0 {
                    return Err(EKError::InvalidInput(
                        "--intermediate must be > 0".to_string(),
                    ));
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(EKError::InvalidInput(format!("unknown argument: {arg}")));
            }
        }
    }

    let cache_dir = cache_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "ek-rsm-load-smoke-{}-{}",
            std::process::id(),
            mode.as_str()
        ))
    });

    Ok(Options {
        mode,
        event_log,
        stage_log,
        out_dir,
        cache_dir,
        repeat,
        hidden,
        intermediate,
    })
}

fn parse_mode(value: &str) -> EKResult<SmokeMode> {
    match value {
        "baseline" | "off" => Ok(SmokeMode::Baseline),
        "rsm-host" => Ok(SmokeMode::RsmHost),
        _ => Err(EKError::InvalidInput(format!(
            "unknown mode {value}; expected baseline or rsm-host"
        ))),
    }
}

fn print_help() {
    println!(
        "Usage: rsm_load_smoke --mode baseline|rsm-host [--event-log PATH] [--out DIR] [--cache-dir DIR]\n\
         [--stage-log PATH] [--repeat N] [--hidden N] [--intermediate N]\n\
         Seeds a tiny SafeTensors expert into LocalWeightManager FS cache and calls real Expert-Kit load_expert_task."
    );
}

fn make_weight_manager(cache_dir: &Path) -> Arc<LocalWeightManager> {
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_dir.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    LocalWeightManager::new_with_parts(operator, None, 1, false)
}

async fn seed_cache(cache_dir: &Path, key: &ExpertKey, bytes: &[u8]) -> EKResult<()> {
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_dir.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    operator.write(&key.as_object_key(), bytes.to_vec()).await?;
    Ok(())
}

fn tiny_expert_safetensors(hidden: usize, intermediate: usize) -> EKResult<Vec<u8>> {
    let up_data = f32_bytes(intermediate * hidden, 0.01);
    let down_data = f32_bytes(hidden * intermediate, 0.02);
    let gate_data = f32_bytes(intermediate * hidden, 0.03);

    let up = TensorView::new(Dtype::F32, vec![intermediate, hidden], &up_data)?;
    let down = TensorView::new(Dtype::F32, vec![hidden, intermediate], &down_data)?;
    let gate = TensorView::new(Dtype::F32, vec![intermediate, hidden], &gate_data)?;

    Ok(safetensors::tensor::serialize(
        vec![
            ("layers.0.experts.1.w1.weight".to_string(), up),
            ("layers.0.experts.1.w2.weight".to_string(), down),
            ("layers.0.experts.1.w3.weight".to_string(), gate),
        ],
        &None,
    )?)
}

fn f32_bytes(count: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * std::mem::size_of::<f32>());
    for idx in 0..count {
        out.extend_from_slice(&((idx as f32 + 1.0) * scale).to_le_bytes());
    }
    out
}

fn count_lines(path: &Path) -> std::io::Result<usize> {
    match fs::read_to_string(path) {
        Ok(content) => Ok(content.lines().count()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(err) => Err(err),
    }
}

fn count_data_rows(path: &Path) -> std::io::Result<usize> {
    Ok(count_lines(path)?.saturating_sub(1))
}

fn percentile(samples: &[u128], percentile: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) * percentile).div_ceil(100);
    sorted[idx]
}

fn smoke_report_markdown(outcome: &SmokeOutcome) -> String {
    format!(
        "# Real Expert-Kit load_expert_task Smoke Report\n\n\
         This artifact is **real Expert-Kit loading-path evidence**. It calls `ek_computation::worker::x::load_expert_task` with a tiny SafeTensors expert and proves the loading path can build an `ExpertBackend`. In `rsm-host` mode, the same call routes bytes through RSM HostMem acquire/release before backend construction.\n\n\
         - Mode: `{}`\n\
         - Expert object key: `{}`\n\
         - load_expert_task invoked: `true`\n\
         - ExpertBackend build succeeded: `true`\n\
         - Loaded ExpertDB entries: `{}`\n\
         - Repeat count: `{}`\n\
         - Hidden size: `{}`\n\
         - Intermediate size: `{}`\n\
         - Weight bytes: `{}`\n\
         - New RSM events: `{}`\n\
         - New stage timing rows: `{}`\n\
         - Event log: `{}`\n\
         - Stage timing CSV: `{}`\n\
         - Cache dir: `{}`\n\
         - Total elapsed ms: `{}`\n\
         - Median per-run elapsed us: `{}`\n\
         - P95 per-run elapsed us: `{}`\n\n\
         Evidence boundary: this smoke proves the local Expert-Kit loading seam and RSM event export. It does not claim production multi-node performance, full forward-path integration, RDMA/Mooncake transport, or real DeviceHBM hardware validation.\n",
        outcome.mode.as_str(),
        outcome.expert_key,
        outcome.loaded_count,
        outcome.repeat,
        outcome.hidden,
        outcome.intermediate,
        outcome.weight_bytes,
        outcome.new_event_count,
        outcome.new_stage_rows,
        outcome
            .event_log
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unset>".to_string()),
        outcome
            .stage_log
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unset>".to_string()),
        outcome.cache_dir.display(),
        outcome.total_elapsed_ms,
        outcome.median_elapsed_us,
        outcome.p95_elapsed_us
    )
}
