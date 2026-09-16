//! Build a small Qwen SafeTensors checkpoint for crate-local tests.

use std::{collections::BTreeMap, path::PathBuf, sync::OnceLock};

use safetensors::{Dtype, tensor::TensorView};

const MODEL_NAME: &str = "qwen-test";
const NUM_LAYERS: usize = 10;
const NUM_EXPERTS: usize = 256;
const HIDDEN_DIM: usize = 16;
const INTERMEDIATE_DIM: usize = 8;

pub(crate) fn synthetic_qwen_model() -> PathBuf {
    static MODEL_ROOT: OnceLock<PathBuf> = OnceLock::new();

    MODEL_ROOT
        .get_or_init(|| {
            let parent =
                std::env::temp_dir().join(format!("expert-kit-ek-db-tests-{}", std::process::id()));
            let model_root = parent.join(MODEL_NAME);
            if parent.exists() {
                std::fs::remove_dir_all(&parent).unwrap();
            }
            std::fs::create_dir_all(&model_root).unwrap();

            let config = serde_json::json!({
                "model_type": "qwen3_moe",
                "num_hidden_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "hidden_size": HIDDEN_DIM,
                "moe_intermediate_size": INTERMEDIATE_DIM,
            });
            std::fs::write(
                model_root.join("config.json"),
                serde_json::to_vec_pretty(&config).unwrap(),
            )
            .unwrap();

            let tensor_data = vec![0_u8; HIDDEN_DIM * INTERMEDIATE_DIM * 2];
            let mut tensors = BTreeMap::new();
            let mut weight_map = serde_json::Map::new();
            for layer_id in 0..NUM_LAYERS {
                for expert_id in 0..NUM_EXPERTS {
                    for projection in ["gate_proj", "up_proj", "down_proj"] {
                        let name = format!(
                            "model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}.weight"
                        );
                        let shape = if projection == "down_proj" {
                            vec![HIDDEN_DIM, INTERMEDIATE_DIM]
                        } else {
                            vec![INTERMEDIATE_DIM, HIDDEN_DIM]
                        };
                        tensors.insert(
                            name.clone(),
                            TensorView::new(Dtype::BF16, shape, &tensor_data).unwrap(),
                        );
                        weight_map.insert(
                            name,
                            serde_json::Value::String("model.safetensors".to_string()),
                        );
                    }
                }
            }

            safetensors::tensor::serialize_to_file(
                &tensors,
                None,
                &model_root.join("model.safetensors"),
            )
            .unwrap();
            let index = serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            });
            std::fs::write(
                model_root.join("model.safetensors.index.json"),
                serde_json::to_vec(&index).unwrap(),
            )
            .unwrap();

            model_root
        })
        .clone()
}

/// Build one small expert in the published V4 FP4 or compressed-tensors layout.
pub(crate) fn synthetic_v4_model(fp4: bool) -> PathBuf {
    static FP4_ROOT: OnceLock<PathBuf> = OnceLock::new();
    static INT8_ROOT: OnceLock<PathBuf> = OnceLock::new();
    let cell = if fp4 { &FP4_ROOT } else { &INT8_ROOT };
    cell.get_or_init(|| {
        let root =
            std::env::temp_dir().join(format!("expert-kit-v4-tests-{}-{fp4}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let raw = if fp4 {
            include_str!("../../tests/fixtures/deepseek_v4_fp4_config.json")
        } else {
            include_str!("../../tests/fixtures/deepseek_v4_w8a8_config.json")
        };
        let mut config: serde_json::Value = serde_json::from_str(raw).unwrap();
        config["hidden_size"] = 128.into();
        config["moe_intermediate_size"] = 256.into();
        config["num_hidden_layers"] = 1.into();
        config["n_routed_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut owned = Vec::new();
        for expert in 0..2 {
            for (role, rows, cols) in [("w1", 256, 128), ("w2", 128, 256), ("w3", 256, 128)] {
                let base = format!("layers.0.ffn.experts.{expert}.{role}");
                let width = if fp4 { cols / 2 } else { cols };
                owned.push((
                    format!("{base}.weight"),
                    Dtype::I8,
                    vec![rows, width],
                    vec![0x11; rows * width],
                ));
                if fp4 {
                    owned.push((
                        format!("{base}.scale"),
                        Dtype::F8_E8M0,
                        vec![rows, cols / 32],
                        vec![127; rows * cols / 32],
                    ));
                } else {
                    owned.push((
                        format!("{base}.weight_scale"),
                        Dtype::F32,
                        vec![rows, 1],
                        0.125_f32.to_le_bytes().repeat(rows),
                    ));
                }
            }
        }
        let views: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &root.join("model.safetensors")).unwrap();
        let weight_map: BTreeMap<_, _> = owned
            .iter()
            .map(|(name, _, _, _)| (name, "model.safetensors"))
            .collect();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        root
    })
    .clone()
}
