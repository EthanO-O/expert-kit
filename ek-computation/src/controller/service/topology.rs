//! Stream ready expert routes to Frontends.

use std::{pin::Pin, sync::Arc};

use tokio::sync::mpsc;
use tokio_stream::{Stream, wrappers::ReceiverStream};
use tonic::{Request, Response, Status};

use crate::{
    controller::{
        runtime_state::ControllerRuntimeState, service::instance::DefaultInstanceResolver,
    },
    proto::ek::control::v2::{
        TopologyMessage, WatchTopologyRequest, topology_message,
        topology_service_server::TopologyService,
    },
};

use super::status::state_status;

const TOPOLOGY_STREAM_BUFFER: usize = 16;

#[derive(Clone)]
pub struct TopologyServiceImpl {
    state: ControllerRuntimeState,
    instance_resolver: Arc<dyn DefaultInstanceResolver>,
}

impl TopologyServiceImpl {
    pub fn new(
        state: ControllerRuntimeState,
        instance_resolver: Arc<dyn DefaultInstanceResolver>,
    ) -> Self {
        Self {
            state,
            instance_resolver,
        }
    }
}

#[tonic::async_trait]
impl TopologyService for TopologyServiceImpl {
    type WatchTopologyStream =
        Pin<Box<dyn Stream<Item = Result<TopologyMessage, Status>> + Send + 'static>>;

    async fn watch_topology(
        &self,
        request: Request<WatchTopologyRequest>,
    ) -> Result<Response<Self::WatchTopologyStream>, Status> {
        let request = request.into_inner();
        if request.instance_id == 0 {
            return Err(Status::invalid_argument("instance_id must be positive"));
        }
        self.instance_resolver.resolve(request.instance_id).await?;
        let newest = self.state.topology_version(request.instance_id).await;
        if request.current_version > newest {
            return Err(Status::invalid_argument(format!(
                "requested topology version {} is newer than current version {newest}",
                request.current_version
            )));
        }
        let state = self.state.clone();
        let mut changed = state.subscribe();
        let (sender, receiver) = mpsc::channel(TOPOLOGY_STREAM_BUFFER);
        tokio::spawn(async move {
            let mut installed_version = request.current_version;
            let mut sent_initial = false;
            loop {
                let newest = state.topology_version(request.instance_id).await;
                if !sent_initial || installed_version < newest {
                    let messages = match state
                        .topology_messages(request.instance_id, installed_version)
                        .await
                    {
                        Ok(messages) => messages,
                        Err(error) => {
                            let _ = sender.send(Err(state_status(error))).await;
                            return;
                        }
                    };
                    if !send_topology_messages(&sender, messages, &mut installed_version).await {
                        return;
                    }
                    sent_initial = true;
                    continue;
                }
                if changed.changed().await.is_err() {
                    return;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

async fn send_topology_messages(
    sender: &mpsc::Sender<Result<TopologyMessage, Status>>,
    messages: Vec<TopologyMessage>,
    installed_version: &mut u64,
) -> bool {
    let mut sent_version = *installed_version;
    for message in messages {
        sent_version = match message.message.as_ref() {
            Some(topology_message::Message::Snapshot(part)) => part.topology_version,
            Some(topology_message::Message::Update(part)) => part.topology_version,
            None => sent_version,
        };
        if sender.send(Ok(message)).await.is_err() {
            return false;
        }
    }
    // The message read can include revisions published after the version read.
    // Commit its actual final version only after the entire batch is sent.
    *installed_version = sent_version;
    true
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use async_trait::async_trait;
    use tokio_stream::StreamExt;

    use super::*;
    use crate::{
        controller::service::instance::ResolvedDefaultInstance,
        proto::ek::{
            control::v2::{
                ExpertState, ExpertStateKind, RegisterWorkerRequest, TargetExpert, WorkerDevice,
                WorkerRunState, WorkerTransportType, topology_message,
            },
            worker::v2::ActivationDType,
        },
    };

    struct FakeInstanceResolver;

    #[async_trait]
    impl DefaultInstanceResolver for FakeInstanceResolver {
        async fn resolve(
            &self,
            requested_instance_id: u64,
        ) -> Result<ResolvedDefaultInstance, Status> {
            if requested_instance_id != 0 && requested_instance_id != 7 {
                return Err(Status::failed_precondition(
                    "requested instance does not match Controller default",
                ));
            }
            Ok(ResolvedDefaultInstance {
                instance_id: 7,
                model_name: "model".to_owned(),
                instance_name: "default".to_owned(),
            })
        }
    }

    fn instance_resolver() -> Arc<dyn DefaultInstanceResolver> {
        Arc::new(FakeInstanceResolver)
    }

    #[tokio::test]
    async fn topology_cursor_tracks_the_batch_read_after_a_new_revision() {
        for installed in [1, 0] {
            let state = ControllerRuntimeState::new(8);
            state
                .register(RegisterWorkerRequest {
                    worker_id: "worker".to_owned(),
                    start_id: "start".to_owned(),
                    instance_id: 7,
                    computation_endpoint: "127.0.0.1:50051".to_owned(),
                    peer_weight_endpoint: "http://127.0.0.1:50052".to_owned(),
                    backend: "torch".to_owned(),
                    activation_dtype: ActivationDType::ActivationDtypeBf16 as i32,
                    device: Some(WorkerDevice {
                        device: "cuda:0".to_owned(),
                        max_experts: 256,
                    }),
                    max_batch_tokens: 4096,
                    max_active_batches_per_device: 2,
                    max_pending_batches_per_device: 2,
                    transport_type: WorkerTransportType::WorkerTransportGrpc as i32,
                })
                .await
                .unwrap();
            let lease = state.open_heartbeat("worker", "start").await.unwrap();
            state
                .heartbeat(
                    "worker",
                    "start",
                    lease,
                    1,
                    WorkerRunState::WorkerRunning as i32,
                )
                .await
                .unwrap();
            let placement = state
                .set_targets(
                    "worker",
                    "start",
                    (0..66)
                        .map(|expert_id| TargetExpert {
                            layer_id: 1,
                            expert_id,
                            target_device: "cuda:0".to_owned(),
                            peer_weight_endpoints: vec![],
                        })
                        .collect(),
                )
                .await
                .unwrap();
            let ready = |expert_id| ExpertState {
                layer_id: 1,
                expert_id,
                state: ExpertStateKind::ExpertReady as i32,
                failure: None,
            };
            state
                .apply_state_report(
                    "worker",
                    "start",
                    placement.generation,
                    1,
                    true,
                    vec![ready(0)],
                )
                .await
                .unwrap();
            let observed_version = state.topology_version(7).await;
            assert_eq!(observed_version, 1);
            // Advance real state between the watch loop's two independent reads.
            state
                .apply_state_report(
                    "worker",
                    "start",
                    placement.generation,
                    2,
                    false,
                    (1..66).map(ready).collect(),
                )
                .await
                .unwrap();
            let messages = state.topology_messages(7, installed).await.unwrap();
            assert_eq!(messages.len(), 2);
            let (sender, mut receiver) = mpsc::channel(1);
            let mut incomplete_version = installed;
            let (sent, ()) = tokio::time::timeout(Duration::from_secs(1), async {
                tokio::join!(
                    send_topology_messages(&sender, messages.clone(), &mut incomplete_version),
                    async {
                        receiver.recv().await.unwrap().unwrap();
                        receiver.close();
                    }
                )
            })
            .await
            .unwrap();
            assert!(!sent);
            assert_eq!(incomplete_version, installed);
            let (sender, mut receiver) = mpsc::channel(TOPOLOGY_STREAM_BUFFER);
            let mut installed_version = installed;
            assert!(send_topology_messages(&sender, messages, &mut installed_version).await);
            assert_eq!(installed_version, 2);
            assert!(
                state
                    .topology_messages(7, installed_version)
                    .await
                    .unwrap()
                    .is_empty()
            );
            for part_index in 0..2 {
                let message = receiver.try_recv().unwrap().unwrap();
                let (version, index, count) = match message.message.unwrap() {
                    topology_message::Message::Snapshot(part) => {
                        assert_eq!(installed, 0);
                        (part.topology_version, part.part_index, part.part_count)
                    }
                    topology_message::Message::Update(part) => {
                        assert_eq!(installed, 1);
                        assert_eq!(part.previous_version, 1);
                        (part.topology_version, part.part_index, part.part_count)
                    }
                };
                assert_eq!((version, index, count), (2, part_index, 2));
            }
        }
    }

    #[tokio::test]
    async fn topology_service_sends_an_initial_empty_snapshot() {
        let state = ControllerRuntimeState::new(8);
        let service = TopologyServiceImpl::new(state, instance_resolver());
        let response = service
            .watch_topology(Request::new(WatchTopologyRequest {
                instance_id: 7,
                current_version: 0,
            }))
            .await
            .unwrap();
        let mut stream = response.into_inner();
        let message = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Some(topology_message::Message::Snapshot(snapshot)) = message.message else {
            panic!("expected a snapshot");
        };
        assert_eq!(snapshot.topology_version, 0);
        assert_eq!(snapshot.part_count, 1);
        assert!(snapshot.routes.is_empty());
    }

    #[tokio::test]
    async fn topology_rejects_a_nondefault_instance() {
        let service = TopologyServiceImpl::new(ControllerRuntimeState::new(8), instance_resolver());

        let error = match service
            .watch_topology(Request::new(WatchTopologyRequest {
                instance_id: 8,
                current_version: 0,
            }))
            .await
        {
            Ok(_) => panic!("nondefault instance unexpectedly opened a topology stream"),
            Err(error) => error,
        };

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    }
}
