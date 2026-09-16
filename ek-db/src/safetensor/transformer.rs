use std::{collections::HashMap, path::PathBuf};

use actix_web::{HttpResponse, Responder, body::BoxBody, http::header::ContentType};
use ek_base::error::{EKError, EKResult};
use memmap2::MmapOptions;
use safetensors::{SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use tokio::{fs::File, sync::Mutex};

use super::memcache::{SafeTensorWithData, SafetensorCache};
#[derive(Debug, Clone)]
pub struct ModelConfig {
    map: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VitalMeta {
    pub moe_layers: (usize, usize),
    pub routed_experts: usize,
    pub hidden_dim: usize,
    pub inter_dim: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
/// Normalized storage and activation recipe for routed expert projections.
pub struct QuantizationMeta {
    pub method: String,
    pub bits: Option<u32>,
    pub group_size: Option<u32>,
    pub symmetric: Option<bool>,
    pub desc_act: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RuntimeMeta {
    pub schema_version: u32,
    pub model_type: String,
    pub num_layers: usize,
    pub moe_layer_start: usize,
    pub moe_layer_end: usize,
    pub experts_per_layer: usize,
    pub hidden_dim: usize,
    pub expert_intermediate_dim: usize,
    pub top_k: Option<usize>,
    pub activation_dtype: Option<String>,
    pub quantization: Option<QuantizationMeta>,
    /// Selects SwiGLU arithmetic and whether routing weights precede the down projection.
    pub expert_compute: String,
    /// V4 gate upper bound and symmetric up-projection bound; zero disables clipping.
    pub swiglu_limit: Option<f64>,
}

impl ModelConfig {
    fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        let path = desc.root.join(&desc.config_name);
        let file = std::fs::File::open(path.clone()).map_err(move |e| {
            log::error!("can not found model_config at {}", &path.to_string_lossy());
            EKError::IoError(e)
        })?;
        let map: HashMap<_, _> = serde_json::from_reader(file)?;
        Ok(Self { map })
    }
    pub fn model_type(&self) -> &str {
        self.map.get("model_type").unwrap().as_str().unwrap()
    }

    fn quantization_method(&self) -> Option<&str> {
        self.map
            .get("quantization_config")
            .and_then(serde_json::Value::as_object)
            .and_then(|config| {
                config
                    .get("quant_method")
                    .or_else(|| config.get("quantization_method"))
            })
            .and_then(serde_json::Value::as_str)
    }

    fn is_gptq(&self) -> bool {
        self.quantization_method() == Some("gptq")
    }

    fn expert_quantization(&self) -> EKResult<Option<QuantizationMeta>> {
        let Some(config) = self.map.get("quantization_config") else {
            return Ok(None);
        };
        let method = self.quantization_method().ok_or_else(|| {
            EKError::InvalidInput("quantization_config requires quant_method".into())
        })?;
        if method == "compressed-tensors" {
            let unsupported = || {
                EKError::InvalidInput(
                "only symmetric compressed-tensors int-quantized W8A8 token/channel experts are supported".into())
            };
            let groups = config
                .get("config_groups")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(unsupported)?;
            if config.get("format").and_then(serde_json::Value::as_str) != Some("int-quantized")
                || groups.len() != 1
                || config.get("transform_config").is_some_and(|v| !v.is_null())
            {
                return Err(unsupported());
            }
            let group = groups.values().next().unwrap();
            if group.get("targets") != Some(&serde_json::json!(["Linear"]))
                || group
                    .get("output_activations")
                    .is_some_and(|v| !v.is_null())
                || group
                    .get("activation_use_clip")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            {
                return Err(unsupported());
            }
            for (field, strategy, dynamic) in [
                ("weights", "channel", false),
                ("input_activations", "token", true),
            ] {
                let q = &group[field];
                if q["num_bits"] != 8
                    || q["type"] != "int"
                    || q["symmetric"] != true
                    || q["strategy"] != strategy
                    || q["dynamic"] != dynamic
                    || ["group_size", "block_structure", "actorder"]
                        .iter()
                        .any(|key| !q[key].is_null())
                {
                    return Err(unsupported());
                }
            }
            // Regex targets and partially quantized routed experts require a separate resolver.
            if let Some(ignore) = config.get("ignore") {
                let ignored = ignore.as_array().ok_or_else(unsupported)?;
                for value in ignored {
                    let name = value.as_str().ok_or_else(unsupported)?;
                    if name.contains("experts") || name.starts_with("re:") || name == "Linear" {
                        return Err(unsupported());
                    }
                }
            }
            return Ok(Some(QuantizationMeta {
                method: "w8a8".into(),
                bits: Some(8),
                group_size: None,
                symmetric: Some(true),
                desc_act: Some(false),
            }));
        }
        if self.model_type() == "deepseek_v4"
            && method == "fp8"
            && self
                .map
                .get("expert_dtype")
                .and_then(serde_json::Value::as_str)
                == Some("fp4")
        {
            if config["scale_fmt"] != "ue8m0"
                || config["activation_scheme"] != "dynamic"
                || config["fmt"] != "e4m3"
            {
                return Err(EKError::InvalidInput(
                    "unsupported V4 FP4 activation or scale recipe".into(),
                ));
            }
            return Ok(Some(QuantizationMeta {
                method: "mxfp4".into(),
                bits: Some(4),
                group_size: Some(32),
                symmetric: Some(true),
                desc_act: Some(false),
            }));
        }
        if matches!(method, "w8a8" | "mxfp4") {
            return Err(EKError::InvalidInput(
                "checkpoint requires a validated quantization format, not only a bit-width label"
                    .into(),
            ));
        }
        Ok(Some(QuantizationMeta {
            method: method.to_owned(),
            bits: config
                .get("bits")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32),
            group_size: config
                .get("group_size")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32),
            symmetric: config
                .get("sym")
                .or_else(|| config.get("symmetric"))
                .and_then(serde_json::Value::as_bool),
            desc_act: config.get("desc_act").and_then(serde_json::Value::as_bool),
        }))
    }

    pub fn moe_layers(&self) -> Option<(usize, usize)> {
        match self.model_type() {
            "deepseek_v2" | "deepseek_v3" => {
                let start = self
                    .map
                    .get("first_k_dense_replace")
                    .unwrap()
                    .as_u64()
                    .unwrap() as usize;
                let end = self.map.get("num_hidden_layers")?.as_u64()? as usize;
                Some((start, end))
            }
            "deepseek_v4" => {
                let end = self.map.get("num_hidden_layers")?.as_u64()? as usize;
                Some((0, end))
            }
            _ => {
                let end = self.map.get("num_hidden_layers")?.as_u64()? as usize;
                Some((0, end))
            }
        }
    }

    pub fn routed_experts(&self) -> Option<usize> {
        match self.model_type() {
            "deepseek_v2" | "deepseek_v3" | "deepseek_v4" => {
                Some(self.map.get("n_routed_experts")?.as_u64()? as usize)
            }
            "qwen2_moe" | "qwen3_moe" => Some(self.map.get("num_experts")?.as_u64()? as usize),
            "mixtral" => Some(self.map.get("num_local_experts")?.as_u64()? as usize),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn dim(&self) -> Option<(usize, usize)> {
        let hidden = self.map.get("hidden_size")?.as_u64()? as usize;
        let intermediate = match self.model_type() {
            "deepseek_v2" | "deepseek_v3" | "deepseek_v4" | "qwen2_moe" | "qwen3_moe" => {
                self.map.get("moe_intermediate_size")?.as_u64()? as usize
            }
            "mixtral" => self.map.get("intermediate_size")?.as_u64()? as usize,
            _ => unimplemented!(),
        };
        Some((hidden, intermediate))
    }
    pub fn normalized_vital(&self) -> EKResult<VitalMeta> {
        let dim = self.dim().ok_or(EKError::InvalidInput(
            "can not determine hidden_dim and inter_dim".to_string(),
        ))?;
        Ok(VitalMeta {
            moe_layers: self.moe_layers().ok_or(EKError::InvalidInput(
                "can not determine moe layers".to_string(),
            ))?,
            routed_experts: self.routed_experts().ok_or(EKError::InvalidInput(
                "can not determine routed_experts".to_string(),
            ))?,
            hidden_dim: dim.0,
            inter_dim: dim.1,
        })
    }

    pub fn runtime_meta(&self) -> EKResult<RuntimeMeta> {
        let (moe_start, moe_end) = self.moe_layers().ok_or(EKError::InvalidInput(
            "can not determine moe layers".to_string(),
        ))?;
        let (hidden_dim, expert_intermediate_dim) = self.dim().ok_or(EKError::InvalidInput(
            "can not determine hidden_dim and inter_dim".to_string(),
        ))?;
        let num_layers = moe_end;
        let experts_per_layer = self.routed_experts().ok_or(EKError::InvalidInput(
            "can not determine routed_experts".to_string(),
        ))?;
        let top_k = self
            .map
            .get("num_experts_per_tok")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .or_else(|| {
                self.map
                    .get("num_selected_experts")
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value as usize)
            });
        let activation_dtype = self
            .map
            .get("torch_dtype")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        let quantization = self.expert_quantization()?;
        Ok(RuntimeMeta {
            schema_version: 1,
            model_type: self.model_type().to_owned(),
            num_layers,
            moe_layer_start: moe_start,
            moe_layer_end: moe_end,
            experts_per_layer,
            hidden_dim,
            expert_intermediate_dim,
            top_k,
            activation_dtype,
            quantization,
            expert_compute: if self.model_type() == "deepseek_v4" {
                "deepseek_v4"
            } else {
                "swiglu"
            }
            .into(),
            swiglu_limit: self
                .map
                .get("swiglu_limit")
                .and_then(serde_json::Value::as_f64),
        })
    }
}

struct WeightMap {
    map: std::collections::HashMap<String, String>,
}

impl WeightMap {
    fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        let mut map = HashMap::new();
        let path = desc.root.join(&desc.weight_map_name);
        let file = std::fs::File::open(path.clone()).map_err(move |e| {
            log::error!(
                "can not found weight_map_file at {}",
                &path.to_string_lossy()
            );
            EKError::IoError(e)
        })?;
        let res: serde_json::Value = serde_json::from_reader(file)?;
        res.get("weight_map")
            .ok_or(EKError::NotFound("weight_map key not found".to_string()))?
            .as_object()
            .ok_or(EKError::NotFound(
                "weight_map is not a valid object".to_string(),
            ))?
            .iter()
            .for_each(|(k, v)| {
                let v = v.as_str().unwrap();
                map.insert(k.to_string(), v.to_string());
            });
        Ok(Self { map })
    }
    fn map_layer(&self, key: &String) -> Option<String> {
        self.map.get(key).cloned()
    }
}

#[derive(Debug, Clone)]
pub struct TransformerModelDesc {
    pub root: PathBuf,
    pub weight_map_name: String,
    pub config_name: String,
}

impl Default for TransformerModelDesc {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            weight_map_name: "model.safetensors.index.json".to_string(),
            config_name: "config.json".to_string(),
        }
    }
}

pub struct WrappedTensorView<'data> {
    data: &'data SafeTensorWithData<'data>,
    key: String,
}

impl Responder for WrappedTensorView<'_> {
    type Body = BoxBody;

    fn respond_to(self, _: &actix_web::HttpRequest) -> HttpResponse<BoxBody> {
        let body = self.inner().unwrap().data().to_vec();

        HttpResponse::Ok()
            .content_type(ContentType::octet_stream())
            .body(body)
    }
}

impl<'a> WrappedTensorView<'a> {
    pub fn inner(&self) -> EKResult<TensorView<'a>> {
        let st = self.data.safetensors();
        let tv = st.tensor(self.key.as_str())?;
        Ok(tv.clone())
    }
}

pub struct TransformerPretrained<'data> {
    desc: TransformerModelDesc,
    weight_map: WeightMap,
    model_config: ModelConfig,
    safetensors_cache: SafetensorCache<'data>, // SafeTensorCaCache<PathBuf, Arc<SafeTensorWithData<'data>>>,
    ser_lk: Mutex<()>,
}

impl<'data> TransformerPretrained<'data>
where
    'data: 'static,
{
    pub fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        // return Sel
        let weight_map = WeightMap::try_from_desc(desc)?;
        let model_config = ModelConfig::try_from_desc(desc)?;
        Ok(Self {
            desc: desc.clone(),
            weight_map,
            model_config,
            safetensors_cache: SafetensorCache::new(),
            ser_lk: Mutex::new(()),
        })
    }

    pub fn layer_names_except_experts(&self) -> Vec<String> {
        let mut names = vec![];
        for (k, _v) in self.weight_map.map.iter() {
            if !k.contains("mlp.experts") {
                names.push(k.to_string());
            }
        }
        names
    }

    pub fn config(&self) -> &ModelConfig {
        &self.model_config
    }

    fn has_layer(&self, key: &str) -> bool {
        self.weight_map.map.contains_key(key)
    }

    async fn get_safetensor(&self, key: &str) -> EKResult<&SafeTensors<'data>> {
        let _lg = self.ser_lk.lock().await;
        let fp = self
            .weight_map
            .map_layer(&key.to_string())
            .ok_or(EKError::NotFound(format!(
                "safetensor not found for layer: {key}"
            )))?;
        let fp = self.desc.root.join(fp);
        let fp_str = fp.to_str().unwrap();
        let hit = self.safetensors_cache.contains_key(fp_str);
        if !hit {
            let file = File::open(fp.clone()).await.unwrap();
            let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
            let st = SafeTensorWithData::new(buffer);
            // let st = Arc::new(st);
            self.safetensors_cache.insert(fp_str, st);
        }
        let res = self
            .safetensors_cache
            .get(fp_str)
            .ok_or(EKError::NotFound("safetensor not found".to_string()))?;

        Ok(res)
    }
    pub async fn get_tensor(&self, key: &str) -> EKResult<TensorView<'data>> {
        let st = self.get_safetensor(key).await?;
        let tv = st.tensor(key)?;
        Ok(tv)
    }

    pub async fn get_layer(&self, key: &str) -> EKResult<Vec<u8>> {
        let st = self.get_safetensor(key).await?;
        let serialized = {
            let tv = &st.tensor(key).unwrap();
            safetensors::tensor::serialize([("data", tv)].to_vec(), None)?
        };
        Ok(serialized)
    }

    async fn construct_expert_key(
        &self,
        layer_id: usize,
        expert_id: usize,
    ) -> EKResult<Vec<String>> {
        let is_v4 = self.model_config.model_type() == "deepseek_v4";
        let is_w8a8 = self
            .model_config
            .expert_quantization()?
            .is_some_and(|q| q.method == "w8a8");
        if is_v4 || is_w8a8 {
            let prefix = if is_v4 {
                format!("layers.{layer_id}.ffn.experts.{expert_id}.")
            } else {
                format!("model.layers.{layer_id}.mlp.experts.{expert_id}.")
            };
            // Preserve auxiliary tensors so adapters can reject unsupported formats explicitly.
            let keys: Vec<_> = self
                .weight_map
                .map
                .keys()
                .filter(|name| name.starts_with(&prefix))
                .cloned()
                .collect();
            if keys.is_empty() {
                return Err(EKError::NotFound(format!("no expert tensors for {prefix}")));
            }
            return Ok(keys);
        }
        match self.model_config.model_type() {
            "deepseek_v2" | "deepseek_v3" | "qwen2_moe" | "qwen3_moe" => {
                if self.model_config.is_gptq() {
                    let names = ["gate_proj", "up_proj", "down_proj"];
                    let mut keys = Vec::with_capacity(12);
                    for projection in names {
                        let base =
                            format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}");
                        keys.extend([
                            format!("{base}.qweight"),
                            format!("{base}.qzeros"),
                            format!("{base}.scales"),
                            format!("{base}.g_idx"),
                        ]);
                    }
                    return Ok(keys);
                }
                let key_up =
                    format!("model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight");
                let key_up_scale = format!("{key_up}_scale_inv");

                let key_gate =
                    format!("model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight");
                let key_gate_scale = format!("{key_gate}_scale_inv");

                let key_down =
                    format!("model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight");
                let key_down_scale = format!("{key_down}_scale_inv");

                Ok(vec![
                    key_down,
                    key_gate,
                    key_up,
                    key_up_scale,
                    key_gate_scale,
                    key_down_scale,
                ])
            }
            "mixtral" => {
                let key_up = format!(
                    "model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight"
                );
                let key_up_scale = String::new();

                let key_gate = format!(
                    "model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight"
                );
                let key_gate_scale = String::new();

                let key_down = format!(
                    "model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight"
                );
                let key_down_scale = String::new();

                Ok(vec![
                    key_down,
                    key_gate,
                    key_up,
                    key_up_scale,
                    key_gate_scale,
                    key_down_scale,
                ])
            }
            _ => unimplemented!(),
        }
    }

    pub async fn get_expert(&self, layer_id: usize, eid: usize) -> EKResult<Vec<u8>> {
        let keys = self.construct_expert_key(layer_id, eid).await?;
        let mut tensors: Vec<(String, TensorView<'data>)> = vec![];

        for key in keys.iter() {
            if !self.has_layer(key) {
                continue;
            }
            let st = self.get_safetensor(key).await?;
            let tensor = st.tensor(key)?;
            tensors.push((key.clone(), tensor));
        }
        let serialized = safetensors::tensor::serialize(tensors, None)?;
        Ok(serialized)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use tokio::task::JoinSet;

    use crate::safetensor::{
        test_fixture::synthetic_qwen_model,
        transformer::{TransformerModelDesc, TransformerPretrained},
    };

    #[tokio::test]
    async fn test_get_layer() {
        let desc = TransformerModelDesc {
            root: synthetic_qwen_model(),
            ..TransformerModelDesc::default()
        };
        let pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained
            .get_layer("model.layers.9.mlp.experts.94.down_proj.weight")
            .await
            .unwrap();

        let tv = safetensors::SafeTensors::deserialize(&tensor)
            .unwrap()
            .tensor("data")
            .unwrap();

        assert_eq!(tv.shape(), &[16, 8]);
    }

    #[tokio::test]
    async fn test_get_expert() {
        let desc = TransformerModelDesc {
            root: synthetic_qwen_model(),
            ..TransformerModelDesc::default()
        };
        let pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained.get_expert(9, 97).await.unwrap();
        let st = safetensors::SafeTensors::deserialize(&tensor).unwrap();
        let names = st.names();
        assert_eq!(names.len(), 3);
        let expected = vec![
            "model.layers.9.mlp.experts.97.gate_proj.weight",
            "model.layers.9.mlp.experts.97.down_proj.weight",
            "model.layers.9.mlp.experts.97.up_proj.weight",
        ];

        for name in expected {
            assert!(names.contains(&name));
        }
        let tensor = st
            .tensor("model.layers.9.mlp.experts.97.down_proj.weight")
            .unwrap();
        assert_eq!(tensor.shape(), &[16, 8]);
    }

    #[tokio::test]
    async fn pressure_test() {
        let desc = TransformerModelDesc {
            root: synthetic_qwen_model(),
            ..TransformerModelDesc::default()
        };
        let pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let pretrained = Arc::new(pretrained);

        let mut js = JoinSet::new();

        for layer in 3..9 {
            for expert in 1..10 {
                let p = pretrained.clone();
                js.spawn(async move { p.get_expert(layer, expert).await });
            }
        }
        js.join_all().await;
    }
    fn v4_config(fp4: bool) -> super::ModelConfig {
        let raw = if fp4 {
            include_str!("../../tests/fixtures/deepseek_v4_fp4_config.json")
        } else {
            include_str!("../../tests/fixtures/deepseek_v4_w8a8_config.json")
        };
        super::ModelConfig {
            map: serde_json::from_str(raw).unwrap(),
        }
    }

    #[test]
    fn v4_quantization_is_scoped_to_routed_experts() {
        for fp4 in [true, false] {
            let meta = v4_config(fp4).runtime_meta().unwrap();
            assert_eq!(
                (
                    meta.hidden_dim,
                    meta.expert_intermediate_dim,
                    meta.experts_per_layer,
                    meta.num_layers
                ),
                (4096, 2048, 256, 43)
            );
            assert_eq!((meta.moe_layer_start, meta.top_k), (0, Some(6)));
            assert_eq!(meta.expert_compute, "deepseek_v4");
            assert_eq!(meta.swiglu_limit, Some(10.0));
            let q = meta.quantization.unwrap();
            assert_eq!(q.method, if fp4 { "mxfp4" } else { "w8a8" });
            assert_eq!(q.bits, Some(if fp4 { 4 } else { 8 }));
        }
    }

    #[test]
    fn reject_unsupported_compressed_tensors_recipes() {
        for (field, key, value) in [
            ("weights", "symmetric", serde_json::json!(false)),
            ("weights", "strategy", serde_json::json!("group")),
            ("weights", "num_bits", serde_json::json!(4)),
            ("input_activations", "dynamic", serde_json::json!(false)),
            ("input_activations", "type", serde_json::json!("float")),
        ] {
            let mut config = v4_config(false);
            config.map.get_mut("quantization_config").unwrap()["config_groups"]["group_0"][field]
                [key] = value;
            assert!(config.runtime_meta().is_err());
        }
        for ignore in ["re:.*", "layers.0.ffn.experts.0.w1"] {
            let mut config = v4_config(false);
            config.map.get_mut("quantization_config").unwrap()["ignore"] =
                serde_json::json!([ignore]);
            assert!(config.runtime_meta().is_err());
        }
    }

    #[tokio::test]
    async fn v4_extraction_preserves_packed_weights_and_scales() {
        for fp4 in [true, false] {
            let desc = TransformerModelDesc {
                root: crate::safetensor::test_fixture::synthetic_v4_model(fp4),
                ..Default::default()
            };
            let pretrained = TransformerPretrained::try_from_desc(&desc).unwrap();
            let bytes = pretrained.get_expert(0, 0).await.unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            assert_eq!(tensors.names().len(), 6);
            let weight = tensors.tensor("layers.0.ffn.experts.0.w1.weight").unwrap();
            assert_eq!(weight.dtype(), safetensors::Dtype::I8);
            assert_eq!(weight.shape(), &[256, if fp4 { 64 } else { 128 }]);
            let suffix = if fp4 { "scale" } else { "weight_scale" };
            let scale = tensors
                .tensor(&format!("layers.0.ffn.experts.0.w1.{suffix}"))
                .unwrap();
            assert_eq!(
                scale.dtype(),
                if fp4 {
                    safetensors::Dtype::F8_E8M0
                } else {
                    safetensors::Dtype::F32
                }
            );
            assert_eq!(scale.shape(), &[256, if fp4 { 4 } else { 1 }]);
            assert!(
                tensors
                    .names()
                    .iter()
                    .all(|name| name.starts_with("layers.0.ffn.experts.0."))
            );
        }
    }
}
