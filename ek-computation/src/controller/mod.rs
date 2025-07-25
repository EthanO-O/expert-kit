pub mod dispatcher;
pub mod executor;
pub mod poller;
pub mod registry;
pub mod service;

use crate::{
    metrics, 
    proto::ek::control::v1::plan_service_server::PlanServiceServer,
    schedule,
};
use ek_base::error::EKResult;
use metrics::spawn_metrics_server;
use service::control::PlanServiceImpl;

use super::{
    controller::{self, poller::start_poll},
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_server::StateServiceServer,
    },
};

pub async fn controller_main() -> EKResult<()> {
    let settings = ek_base::config::get_ek_settings();

    spawn_metrics_server("0.0.0.0:9080");

    if settings.controller.auto_scaling.enabled {
        // Initialize deployment cache
        let deployment_cache = schedule::init_deployment_cache();
        
        schedule::init_schedule_module(settings.controller.auto_scaling.clone())?;
        log::info!("Auto scaling module initialized");
        
        // Start deployment cache sync loop
        let cache_sync_task = {
            let cache = deployment_cache.clone();
            tokio::task::spawn(async move {
                cache.run_sync_loop().await;
            })
        };
        
        // Start profiler maintenance loop
        let profiler_task = {
            let profiler = schedule::get_profiler();
            tokio::task::spawn(async move {
                profiler.run_maintenance_loop().await;
            })
        };
        
        // Start auto scaler loop
        let auto_scaler_task = {
            let config = settings.controller.auto_scaling.clone();
            tokio::task::spawn(async move {
                log::info!("Auto scaling is enabled, starting scaler");
                let mut scaler = schedule::AutoScaler::new(config);
                if let Err(e) = scaler.run_scaling_loop().await {
                    log::error!("Auto scaler error: {e}");
                }
            })
        };
        
        // Register background tasks
        tokio::spawn(async move {
            tokio::try_join!(cache_sync_task, profiler_task, auto_scaler_task).ok();
        });
    } else {
        log::info!("Auto scaling is disabled in configuration");
    }

    let state_srv = tokio::task::spawn(async {
        let srv = controller::service::state::StateServerImpl::new();
        let intra_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.intra
        )
        .parse()
        .unwrap();
        log::info!("state server listening on {intra_addr}");
        let err = tonic::transport::Server::builder()
            .add_service(StateServiceServer::new(srv))
            .serve(intra_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {e:?}");
        }
    });

    let computation_srv = tokio::task::spawn(async {
        let srv = controller::service::compute::ComputationProxyServiceImpl::new();
        let inter_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.inter
        )
        .parse()
        .unwrap();

        // let layer = tower::ServiceBuilder::new()
        //     .layer_fn(OTelGrpcServerMiddleware::new)
        //     .into_inner();

        log::info!("computation server listening on {inter_addr}");
        let plan_srv = PlanServiceImpl::new();
        let err = tonic::transport::Server::builder()
            // .layer(layer)
            .add_service(
                ComputationServiceServer::new(srv)
                    .max_decoding_message_size(1024 * 1024 * 1024)
                    .max_encoding_message_size(1024 * 1024 * 1024),
            )
            .add_service(PlanServiceServer::new(plan_srv))
            .serve(inter_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {e:?}");
        }
    });

    start_poll();

    log::info!("expert kit controller started");
    state_srv.await?;
    computation_srv.await?;
    Ok(())
}
