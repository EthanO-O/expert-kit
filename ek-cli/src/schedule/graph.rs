//! Validate and atomically install the expert placement required by a decode graph.

use std::{collections::HashSet, future::Future, path::PathBuf, time::Duration};

use ek_base::{
    config::{InferenceSettings, get_ek_settings},
    error::{EKError, EKResult},
};
use ek_computation::{
    controller::scheduler::max_experts,
    state::{
        io::{StateReader, StateReaderImpl},
        models::{Expert, Instance, Model, NewExpert, Node},
        writer::StateWriterImpl,
    },
};
use ek_db::{
    safetensor::{ExpertKey, transformer::VitalMeta},
    weight_srv::client::WeightSrvClient,
};
use serde::Deserialize;

// Ignore graph execution fields: this command owns placement, not graph capture.
#[derive(Clone, Debug, Deserialize)]
struct GraphConfig {
    instance_id: i32,
    weight_version: String,
    model: GraphModel,
    layer_ids: Vec<usize>,
    #[serde(default)]
    workers: Vec<GraphWorker>,
}

#[derive(Clone, Debug, Deserialize)]
struct GraphModel {
    experts_per_layer: usize,
    hidden_dim: usize,
    expert_intermediate_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct GraphWorker {
    worker_id: String,
    expert_start: usize,
    expert_end: usize,
}

fn invalid(message: impl Into<String>) -> EKError {
    EKError::RuntimeError(format!("graph placement: {}", message.into()))
}

fn require_manual_scheduling(auto_assign: bool) -> EKResult<()> {
    if auto_assign {
        return Err(invalid("controller.scaling.auto_assign must be false"));
    }
    Ok(())
}

/// Commit validated cold-start placement and optionally await reported expert loads.
pub(super) async fn execute_graph_schedule(
    graph_config: PathBuf,
    wait_ready_seconds: u64,
) -> EKResult<()> {
    let settings = get_ek_settings();
    require_manual_scheduling(settings.controller.scaling.auto_assign)?;
    let contents = tokio::fs::read(&graph_config).await?;
    let graph: GraphConfig = serde_json::from_slice(&contents)
        .map_err(|error| invalid(format!("invalid {}: {error}", graph_config.display())))?;
    let reader = StateReaderImpl::new();
    let inference = &settings.inference;
    let model = reader
        .model_by_name(&inference.model_name)
        .await?
        .ok_or_else(|| invalid(format!("model {} must already exist", inference.model_name)))?;
    let instance = reader
        .instance_by_name(&inference.instance_name)
        .await?
        .ok_or_else(|| {
            invalid(format!(
                "instance {} must already exist",
                inference.instance_name
            ))
        })?;
    let server = settings
        .weight
        .server
        .as_ref()
        .ok_or_else(|| invalid("weight.server must be configured"))?;
    let vital = WeightSrvClient::new(server.addr.clone())
        .load_meta_vital(&inference.model_name)
        .await?;
    let nodes = reader.active_nodes().await?;
    let assignments = plan_graph_assignments(&graph, &vital, &model, &instance, &nodes, inference)?;
    let targets = assignment_targets(&assignments);
    let writer = StateWriterImpl::new();
    let count = replace_graph_assignments(
        instance.id,
        assignments,
        |node_id| reader.experts_by_node(node_id),
        |instance_id, assignments| writer.replace_experts_for_instance(instance_id, assignments),
    )
    .await?;
    tracing::info!(
        instance_id = instance.id,
        workers = graph.workers.len(),
        experts = count,
        "Graph placement committed; Controller polling will load the experts"
    );
    if wait_ready_seconds > 0 {
        tokio::time::timeout(Duration::from_secs(wait_ready_seconds), async {
            loop {
                let experts = reader.experts_by_instance(instance.id).await?;
                if targets_ready(instance.id, &targets, &experts) {
                    return Ok::<_, EKError>(());
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
        .await
        .map_err(|_| invalid(format!("timed out after {wait_ready_seconds}s waiting for target experts to load; placement remains committed")))??;
        tracing::info!(
            instance_id = instance.id,
            "All graph target experts report loaded"
        );
    }
    Ok(())
}

/// Check all target nodes before writing; cold starts forbid concurrent schedulers.
async fn replace_graph_assignments<Read, ReadFuture, Replace, ReplaceFuture>(
    instance_id: i32,
    assignments: Vec<NewExpert>,
    mut read_node: Read,
    replace: Replace,
) -> EKResult<usize>
where
    Read: FnMut(i32) -> ReadFuture,
    ReadFuture: Future<Output = EKResult<Vec<Expert>>>,
    Replace: FnOnce(i32, Vec<NewExpert>) -> ReplaceFuture,
    ReplaceFuture: Future<Output = EKResult<usize>>,
{
    let targets = assignment_targets(&assignments);
    let node_ids: HashSet<_> = assignments.iter().map(|expert| expert.node_id).collect();
    for node_id in node_ids {
        let existing = read_node(node_id).await?;
        if let Some(foreign) = existing
            .iter()
            .find(|expert| expert.instance_id != instance_id)
        {
            return Err(invalid(format!(
                "worker node {node_id} has assignments for instance {}; clear or finish the other instance before a graph cold start",
                foreign.instance_id
            )));
        }
        // The writer preserves unchanged states; scheduled targets are not loaded
        // by the ordinary poller after progressive scheduling has been disabled.
        if let Some(scheduled) = existing.iter().find(|expert| {
            expert.state.get("status").and_then(|value| value.as_str()) == Some("scheduled")
                && targets.contains(&(expert.node_id, expert.expert_id.clone()))
        }) {
            return Err(invalid(format!(
                "target {} on worker node {node_id} still has scheduled state; finish or clear the previous progressive scheduling before a graph cold start",
                scheduled.expert_id
            )));
        }
    }
    replace(instance_id, assignments).await
}

/// Validate the complete placement before the caller mutates database assignments.
fn plan_graph_assignments(
    graph: &GraphConfig,
    vital: &VitalMeta,
    model: &Model,
    instance: &Instance,
    nodes: &[Node],
    inference: &InferenceSettings,
) -> EKResult<Vec<NewExpert>> {
    if graph.instance_id <= 0
        || instance.id != graph.instance_id
        || instance.name != inference.instance_name
        || model.name != inference.model_name
        || instance.model_id != model.id
    {
        return Err(invalid(
            "instance_id must identify the existing configured instance and model",
        ));
    }
    // Registration and database metadata do not expose a weight version to verify.
    if graph.weight_version.trim().is_empty() {
        return Err(invalid("weight_version must be nonempty"));
    }
    if vital.moe_layers.0 >= vital.moe_layers.1
        || vital.routed_experts == 0
        || vital.hidden_dim == 0
        || vital.inter_dim == 0
    {
        return Err(invalid("Weight Server returned invalid model geometry"));
    }
    if graph
        .layer_ids
        .iter()
        .copied()
        .ne(vital.moe_layers.0..vital.moe_layers.1)
    {
        return Err(invalid(
            "layer_ids must contain every Weight Server MoE layer in order exactly once",
        ));
    }
    if graph.model.experts_per_layer != vital.routed_experts
        || graph.model.hidden_dim != vital.hidden_dim
        || graph.model.expert_intermediate_dim != vital.inter_dim
    {
        return Err(invalid(
            "model expert count, hidden dimension and intermediate dimension must match Weight Server metadata",
        ));
    }
    if graph.workers.is_empty() {
        return Err(invalid(
            "workers must be nonempty; legacy single-worker graph configs are unsupported",
        ));
    }
    let mut worker_ids = HashSet::new();
    let mut node_ids = HashSet::new();
    let mut validated = Vec::new();
    let mut next_expert = 0;
    for worker in &graph.workers {
        if worker.worker_id.trim().is_empty() || !worker_ids.insert(&worker.worker_id) {
            return Err(invalid("worker_id values must be nonempty and unique"));
        }
        if worker.expert_start != next_expert
            || worker.expert_end <= worker.expert_start
            || worker.expert_end > vital.routed_experts
        {
            return Err(invalid(
                "worker expert ranges must be contiguous, nonempty and cover experts from zero in workers order",
            ));
        }
        let mut matching = nodes
            .iter()
            .filter(|node| node.hostname == worker.worker_id);
        let node = matching
            .next()
            .ok_or_else(|| invalid(format!("worker {} is not active", worker.worker_id)))?;
        if matching.next().is_some() || node.id <= 0 || !node_ids.insert(node.id) {
            return Err(invalid(
                "active worker names and node IDs must identify unique nodes",
            ));
        }
        if node
            .config
            .get("instance_id")
            .and_then(|value| value.as_i64())
            != Some(i64::from(instance.id))
        {
            return Err(invalid(format!(
                "worker {} is registered to a different instance",
                worker.worker_id
            )));
        }
        let required = (worker.expert_end - worker.expert_start)
            .checked_mul(graph.layer_ids.len())
            .and_then(|slots| u64::try_from(slots).ok())
            .ok_or_else(|| invalid("required worker capacity overflows"))?;
        if max_experts(node) < required {
            return Err(invalid(format!(
                "worker {} needs {required} expert slots but reports {}",
                worker.worker_id,
                max_experts(node)
            )));
        }
        validated.push((worker, node));
        next_expert = worker.expert_end;
    }
    if next_expert != vital.routed_experts {
        return Err(invalid("worker ranges must cover all experts"));
    }
    let mut assignments = Vec::new();
    for (worker, node) in validated {
        for &layer in &graph.layer_ids {
            for expert in worker.expert_start..worker.expert_end {
                assignments.push(NewExpert {
                    instance_id: instance.id,
                    node_id: node.id,
                    expert_id: ExpertKey::new(model.name.clone(), layer, expert).as_object_key(),
                    replica: 1,
                    state: serde_json::json!({"status": "pending"}),
                });
            }
        }
    }
    Ok(assignments)
}

fn assignment_targets(assignments: &[NewExpert]) -> HashSet<(i32, String)> {
    assignments
        .iter()
        .map(|expert| (expert.node_id, expert.expert_id.clone()))
        .collect()
}

/// Require the exact assignment set and the existing poller's loaded status.
fn targets_ready(instance_id: i32, targets: &HashSet<(i32, String)>, experts: &[Expert]) -> bool {
    !targets.is_empty()
        && experts.len() == targets.len()
        && experts.iter().all(|expert| {
            expert.instance_id == instance_id
                && expert.replica == 1
                && expert.state.get("status").and_then(|value| value.as_str()) == Some("loaded")
        })
        && experts
            .iter()
            .map(|expert| (expert.node_id, expert.expert_id.clone()))
            .collect::<HashSet<_>>()
            == *targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{cell::Cell, future::ready};

    struct Fixture {
        graph: GraphConfig,
        vital: VitalMeta,
        model: Model,
        instance: Instance,
        nodes: Vec<Node>,
        inference: InferenceSettings,
    }

    impl Fixture {
        fn new(
            layers: usize,
            experts: usize,
            hidden: usize,
            intermediate: usize,
            workers: usize,
        ) -> Self {
            let width = experts / workers;
            Self {
                graph: GraphConfig {
                    instance_id: 101,
                    weight_version: "checkpoint-v1".into(),
                    model: GraphModel {
                        experts_per_layer: experts,
                        hidden_dim: hidden,
                        expert_intermediate_dim: intermediate,
                    },
                    layer_ids: (0..layers).collect(),
                    workers: (0..workers)
                        .map(|index| GraphWorker {
                            worker_id: format!("worker-{index}"),
                            expert_start: index * width,
                            expert_end: (index + 1) * width,
                        })
                        .collect(),
                },
                vital: VitalMeta {
                    moe_layers: (0, layers),
                    routed_experts: experts,
                    hidden_dim: hidden,
                    inter_dim: intermediate,
                },
                model: Model {
                    id: 7,
                    name: "model".into(),
                    config: json!({}),
                },
                instance: Instance {
                    id: 101,
                    model_id: 7,
                    name: "graph".into(),
                },
                nodes: (0..workers)
                    .map(|index| Node {
                        id: index as i32 + 1,
                        hostname: format!("worker-{index}"),
                        device: "npu:0".into(),
                        config: json!({"instance_id":101,"max_experts":layers * width}),
                    })
                    .collect(),
                inference: InferenceSettings {
                    instance_name: "graph".into(),
                    model_name: "model".into(),
                    hidden_dim: hidden,
                    intermediate_dim: intermediate,
                },
            }
        }

        fn plan(&self) -> EKResult<Vec<NewExpert>> {
            plan_graph_assignments(
                &self.graph,
                &self.vital,
                &self.model,
                &self.instance,
                &self.nodes,
                &self.inference,
            )
        }
    }

    #[test]
    fn plans_complete_qwen_and_deepseek_v4_placements() {
        for fixture in [
            Fixture::new(48, 128, 2048, 768, 2),
            Fixture::new(43, 256, 4096, 2048, 8),
        ] {
            let assignments = fixture.plan().unwrap();
            assert_eq!(
                assignments.len(),
                fixture.graph.layer_ids.len() * fixture.vital.routed_experts
            );
            assert_eq!(assignment_targets(&assignments).len(), assignments.len());
            for (worker, node) in fixture.graph.workers.iter().zip(&fixture.nodes) {
                let actual: HashSet<_> = assignments
                    .iter()
                    .filter(|expert| expert.node_id == node.id)
                    .map(|expert| expert.expert_id.clone())
                    .collect();
                let expected: HashSet<_> = fixture
                    .graph
                    .layer_ids
                    .iter()
                    .flat_map(|&layer| {
                        (worker.expert_start..worker.expert_end).map(move |expert| {
                            ExpertKey::new("model".into(), layer, expert).as_object_key()
                        })
                    })
                    .collect();
                assert_eq!(actual, expected);
            }
            assert!(assignments.iter().all(|expert| expert.instance_id == 101
                && expert.replica == 1
                && expert.state == json!({"status":"pending"})));
        }
    }

    #[test]
    fn rejects_invalid_placement_before_writing() {
        type Mutation = fn(&mut Fixture);
        let cases: &[(&str, Mutation, &str)] = &[
            (
                "nonpositive instance",
                |f| f.graph.instance_id = 0,
                "instance_id",
            ),
            (
                "wrong instance",
                |f| f.graph.instance_id += 1,
                "instance_id",
            ),
            (
                "wrong model binding",
                |f| f.instance.model_id += 1,
                "instance_id",
            ),
            (
                "wrong configured instance",
                |f| f.inference.instance_name = "other".into(),
                "instance_id",
            ),
            (
                "wrong configured model",
                |f| f.inference.model_name = "other".into(),
                "instance_id",
            ),
            (
                "empty version",
                |f| f.graph.weight_version = "  ".into(),
                "weight_version",
            ),
            (
                "missing layer",
                |f| {
                    f.graph.layer_ids.pop();
                },
                "layer_ids",
            ),
            ("duplicate layer", |f| f.graph.layer_ids[1] = 0, "layer_ids"),
            (
                "wrong layer order",
                |f| f.graph.layer_ids.swap(0, 1),
                "layer_ids",
            ),
            (
                "wrong expert count",
                |f| f.graph.model.experts_per_layer += 1,
                "metadata",
            ),
            (
                "wrong hidden dimension",
                |f| f.graph.model.hidden_dim += 1,
                "metadata",
            ),
            (
                "wrong intermediate dimension",
                |f| f.graph.model.expert_intermediate_dim += 1,
                "metadata",
            ),
            (
                "invalid vital range",
                |f| f.vital.moe_layers = (1, 0),
                "invalid model geometry",
            ),
            (
                "invalid vital geometry",
                |f| f.vital.routed_experts = 0,
                "invalid model geometry",
            ),
            (
                "legacy config",
                |f| f.graph.workers.clear(),
                "legacy single-worker",
            ),
            (
                "empty worker ID",
                |f| f.graph.workers[0].worker_id.clear(),
                "worker_id",
            ),
            (
                "duplicate worker ID",
                |f| f.graph.workers[1].worker_id = f.graph.workers[0].worker_id.clone(),
                "unique",
            ),
            (
                "inactive worker",
                |f| {
                    f.nodes.pop();
                },
                "not active",
            ),
            (
                "wrong worker instance",
                |f| f.nodes[0].config["instance_id"] = json!(102),
                "different instance",
            ),
            (
                "missing worker instance",
                |f| f.nodes[0].config = json!({"max_experts":3072}),
                "different instance",
            ),
            (
                "duplicate node ID",
                |f| f.nodes[1].id = f.nodes[0].id,
                "unique nodes",
            ),
            ("nonpositive node ID", |f| f.nodes[0].id = 0, "unique nodes"),
            (
                "ambiguous active name",
                |f| f.nodes.push(f.nodes[0].clone()),
                "unique nodes",
            ),
            (
                "range starts above zero",
                |f| f.graph.workers[0].expert_start = 1,
                "contiguous",
            ),
            (
                "empty interval",
                |f| f.graph.workers[0].expert_end = 0,
                "contiguous",
            ),
            (
                "gap",
                |f| f.graph.workers[1].expert_start += 1,
                "contiguous",
            ),
            (
                "overlap",
                |f| f.graph.workers[1].expert_start -= 1,
                "contiguous",
            ),
            (
                "out of range",
                |f| f.graph.workers[1].expert_end += 1,
                "contiguous",
            ),
            (
                "missing tail",
                |f| f.graph.workers[1].expert_end -= 1,
                "cover all experts",
            ),
            (
                "insufficient capacity",
                |f| f.nodes[0].config["max_experts"] = json!(3071),
                "expert slots",
            ),
            (
                "missing capacity",
                |f| f.nodes[0].config["max_experts"] = json!(null),
                "expert slots",
            ),
            (
                "capacity overflow",
                |f| {
                    f.vital.routed_experts = usize::MAX;
                    f.graph.model.experts_per_layer = usize::MAX;
                    f.graph.workers.truncate(1);
                    f.graph.workers[0].expert_end = usize::MAX;
                },
                "overflows",
            ),
        ];
        for (name, mutate, expected) in cases {
            let mut fixture = Fixture::new(48, 128, 2048, 768, 2);
            mutate(&mut fixture);
            let error = fixture.plan().unwrap_err().to_string();
            assert!(error.contains(expected), "{name}: {error}");
        }
        assert!(require_manual_scheduling(false).is_ok());
        assert!(
            require_manual_scheduling(true)
                .unwrap_err()
                .to_string()
                .contains("auto_assign")
        );
    }

    #[test]
    fn reads_only_placement_fields_and_rejects_legacy_workers() {
        let graph: GraphConfig = serde_json::from_value(json!({
            "session_id":"capture", "instance_id":101, "weight_version":"v1",
            "model":{"experts_per_layer":128,"hidden_dim":2048,"expert_intermediate_dim":768,"top_k":8},
            "layer_ids":[0,1], "worker_id":"old-single-worker", "capture_sizes":[1,2]
        })).unwrap();
        assert!(graph.workers.is_empty());
    }

    #[test]
    fn plans_the_weight_servers_nonzero_moe_layer_range() {
        let mut fixture = Fixture::new(2, 4, 8, 16, 2);
        fixture.vital.moe_layers = (1, 3);
        fixture.graph.layer_ids = vec![1, 2];
        assert_eq!(fixture.plan().unwrap().len(), 8);
        fixture.graph.layer_ids = vec![0, 1];
        assert!(
            fixture
                .plan()
                .unwrap_err()
                .to_string()
                .contains("layer_ids")
        );
    }

    #[test]
    fn readiness_requires_exact_loaded_targets() {
        let fixture = Fixture::new(2, 4, 8, 16, 2);
        let assignments = fixture.plan().unwrap();
        let targets = assignment_targets(&assignments);
        let loaded: Vec<_> = assignments
            .iter()
            .enumerate()
            .map(|(index, assignment)| Expert {
                id: index as i32,
                instance_id: assignment.instance_id,
                node_id: assignment.node_id,
                expert_id: assignment.expert_id.clone(),
                replica: 1,
                state: json!({"status":"loaded"}),
            })
            .collect();
        assert!(targets_ready(101, &targets, &loaded));
        assert!(!targets_ready(101, &targets, &loaded[1..]));
        let mutations: &[fn(&mut Vec<Expert>)] = &[
            |experts| experts[0].state = json!({"status":"pending"}),
            |experts| experts[0].state = json!({"status":"scheduled"}),
            |experts| experts[0].state = json!({}),
            |experts| experts[0].node_id += 10,
            |experts| experts[0].instance_id += 1,
            |experts| experts[0].replica += 1,
            |experts| experts[0].expert_id = "other".into(),
            |experts| experts[0] = experts[1].clone(),
            |experts| experts.push(experts[0].clone()),
        ];
        for mutate in mutations {
            let mut experts = loaded.clone();
            mutate(&mut experts);
            assert!(!targets_ready(101, &targets, &experts));
        }
        assert!(!targets_ready(101, &HashSet::new(), &[]));
    }

    #[tokio::test]
    async fn foreign_assignments_or_read_errors_prevent_replacement() {
        let fixture = Fixture::new(2, 4, 8, 16, 2);
        let assignments = fixture.plan().unwrap();
        for status in ["pending", "scheduled", "loaded"] {
            let foreign = Expert {
                id: 99,
                instance_id: 102,
                node_id: 2,
                expert_id: "other-model/l0-e0".into(),
                replica: 1,
                state: json!({"status":status}),
            };
            let existing = vec![foreign];
            let before = existing.clone();
            let error = replace_graph_assignments(
                101,
                assignments.clone(),
                |node_id| {
                    ready(Ok(existing
                        .iter()
                        .filter(|expert| expert.node_id == node_id)
                        .cloned()
                        .collect()))
                },
                |_, _| -> std::future::Ready<EKResult<usize>> {
                    panic!("foreign occupancy must never reach the database writer");
                },
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("assignments for instance 102"));
            assert_eq!(existing, before);
        }
        let error = replace_graph_assignments(
            101,
            assignments,
            |_| ready(Err(invalid("database read failed"))),
            |_, _| -> std::future::Ready<EKResult<usize>> {
                panic!("a failed occupancy query must never reach the database writer");
            },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("database read failed"));
    }

    #[tokio::test]
    async fn own_loaded_assignments_allow_idempotent_replacement() {
        let fixture = Fixture::new(2, 4, 8, 16, 2);
        let assignments = fixture.plan().unwrap();
        let existing: Vec<_> = assignments
            .iter()
            .enumerate()
            .map(|(index, assignment)| Expert {
                id: index as i32,
                instance_id: assignment.instance_id,
                node_id: assignment.node_id,
                expert_id: assignment.expert_id.clone(),
                replica: 1,
                state: json!({"status":"loaded"}),
            })
            .collect();
        let writes = Cell::new(0);
        let count = replace_graph_assignments(
            101,
            assignments.clone(),
            |node_id| {
                ready(Ok(existing
                    .iter()
                    .filter(|expert| expert.node_id == node_id)
                    .cloned()
                    .collect()))
            },
            |instance_id, replacement| {
                assert_eq!(instance_id, 101);
                assert_eq!(
                    assignment_targets(&replacement),
                    assignment_targets(&assignments)
                );
                writes.set(writes.get() + 1);
                ready(Ok(replacement.len()))
            },
        )
        .await
        .unwrap();
        assert_eq!(count, assignments.len());
        assert_eq!(writes.get(), 1);
        assert!(targets_ready(
            101,
            &assignment_targets(&assignments),
            &existing
        ));
    }

    #[tokio::test]
    async fn rejects_only_scheduled_assignments_that_would_be_preserved() {
        let fixture = Fixture::new(2, 4, 8, 16, 2);
        let assignments = fixture.plan().unwrap();
        for case in ["retained target", "old expert", "other node"] {
            let mut scheduled = Expert {
                id: 99,
                instance_id: 101,
                node_id: assignments[0].node_id,
                expert_id: assignments[0].expert_id.clone(),
                replica: 1,
                state: json!({"status":"scheduled"}),
            };
            match case {
                "old expert" => scheduled.expert_id = "model/l9-e0".into(),
                "other node" => scheduled.node_id = 2,
                _ => {}
            }
            let existing = vec![scheduled];
            let before = existing.clone();
            let writes = Cell::new(0);
            let result = replace_graph_assignments(
                101,
                assignments.clone(),
                |node_id| {
                    ready(Ok(existing
                        .iter()
                        .filter(|expert| expert.node_id == node_id)
                        .cloned()
                        .collect()))
                },
                |_, replacement| {
                    assert_ne!(
                        case, "retained target",
                        "retained scheduled state must never reach the writer"
                    );
                    writes.set(writes.get() + 1);
                    ready(Ok(replacement.len()))
                },
            )
            .await;
            if case == "retained target" {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("previous progressive scheduling")
                );
                assert_eq!(writes.get(), 0);
                assert_eq!(existing, before);
            } else {
                assert_eq!(result.unwrap(), assignments.len());
                assert_eq!(writes.get(), 1);
            }
        }
    }
}
