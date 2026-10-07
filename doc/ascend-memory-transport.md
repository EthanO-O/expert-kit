# Mooncake Ascend transport

Expert Kit can move activation tensors through Mooncake Ascend Direct while
using gRPC for session and request control. The default transport remains gRPC.
Use the existing [Ascend deployment workflow](tutorial/ascend/README.md) for
model configuration, image preparation, and Controller setup.

## Build and install

Use Linux ARM64, CPython 3.12, CANN 9.0.1, and a matching PyTorch/torch-npu
environment on both Frontend and Worker hosts. Configure the NPU network and
make `/etc/hccn.conf` available. The generated Ascend Direct services use host
networking; advertised node addresses must be reachable by every peer.

Prepare the three source archives named in
[`build-manifest-ascend.toml`](../third_party/mooncake/build-manifest-ascend.toml).
Download the pinned GitHub source archives under those manifest filenames:

```bash
mkdir -p "$HOME/mooncake-inputs"
curl -fL https://codeload.github.com/kvcache-ai/Mooncake/tar.gz/731c4521dae71a78bc6fe4e218ad52cfc6ed3554 -o "$HOME/mooncake-inputs/mooncake-731c4521.tar.gz"
curl -fL https://codeload.github.com/pybind/pybind11/tar.gz/58c382a8e3d7081364d2f5c62e7f429f0412743b -o "$HOME/mooncake-inputs/pybind11-58c382a8.tar.gz"
curl -fL https://codeload.github.com/alibaba/yalantinglibs/tar.gz/6a0e067d9a43492cf8e4e280b531924fbd724dbd -o "$HOME/mooncake-inputs/yalantinglibs-6a0e067d.tar.gz"
```

The builder verifies their checksums and produces the pinned `ascend3` wheel
and build receipt. Keep its work and artifact directories outside the checkout:

```bash
python3.12 third_party/mooncake/build-ascend.py \
  --input-dir "$HOME/mooncake-inputs" \
  --work-dir "$HOME/mooncake-build" \
  --artifact-dir "$HOME/mooncake-artifacts"
```

Install the resulting `mooncake_transfer_engine-*.whl` in each environment
that runs a Frontend or Worker. For a local virtual environment:

```bash
uv pip install --python .venv/bin/python \
  "$HOME"/mooncake-artifacts/mooncake_transfer_engine-*.whl
```

Runtime images must also provide the builder runtime libraries
`libgflags.so.2.2`, `libglog.so.0`, and `libunwind.so.8` on the native library
search path; the wheel does not bundle them.

For Compose, install the same wheel in the Python environments of the attention
and expert images, then set `images.attention` and `images.expert` in the cluster
configuration to those image tags. The generator does not install native wheels.

## Select and start

Add this top-level section to `dev/ascend/configs/cluster.yaml`:

```yaml
transport:
  type: transfer_engine
  max_workers: 2
  client_max_in_flight: 1
  ascend_receive_fence: stream
  # max_registered_bytes: 1073741824
```

`max_registered_bytes` optionally limits registered staging arenas per process,
including retired arenas. Model weights are outside this limit.
`client_max_in_flight` limits frontend requests per Worker connection.
`ascend_receive_fence: stream` acquires incoming writes on the receiving stream;
`device` selects a device-wide fence. Devices come from the existing attention
and Worker placement configuration.

Generate the configuration, then use the deployment workflow to start the
Controller and Workers, rebalance placement, and start attention:

```bash
uv run python dev/ascend/cli.py generate
dev/ascend/run-compose.sh expert worker-pool1 up -d
dev/ascend/run-compose.sh attention up -d attention
```

Run each command on its assigned host, after the Controller is ready. Repeat
the Worker command for every configured pool. The generator emits Worker
`control_listen`, `control_advertise`, and `segment_advertise` settings and mounts
`a-transport.toml` into attention through `EK_TRANSPORT_CONFIG`. Segment addresses
use `<host>:0`: Mooncake chooses a free port, and Expert Kit exchanges the actual
endpoint. Control ports remain the configured Worker ports.

For a host-native Torch Frontend, set the same generated TOML path before
starting the existing benchmark:

```bash
export EK_TRANSPORT_CONFIG="$PWD/dev/ascend/generated/a-transport.toml"
uv run --package expertkit-torch ek-torch-benchmark \
  --config dev/ascend/generated/torch-bench.yaml run
```

Host-native processes also need `ASCEND_USE_ASYNC_TRANSFER=0`,
`ASCEND_BUFFER_POOL=0:0`, and `ASCEND_USE_SHORT_CONNECTION=0`; leave
`MC_FORCE_TCP`, `MC_USE_TENT`, and `MC_USE_TEV1` unset. Generated Compose services
set the Ascend options.

The TOML contains only `[transfer_engine]`, `segment_name`, and the resource and
fence settings above. Expert Kit fixes the native backend to Ascend Direct and
uses the native peer handshake internally. Omitting `transport`, or selecting
`type: grpc`, restores the existing gRPC configuration.

## Registered memory lifetime

Each connection reserves a registered arena for input and output staging.
Closing a frontend connection drains its requests and disconnects its Worker
session, then retires the arena logically. Its allocation and native registration
remain owned by the shared runtime and continue counting against
`max_registered_bytes`; the arena cannot be reused by another connection.

The frontend releases retired registrations only after all Worker connections
have closed and the shared runtime shuts down. ADXL registration is shared
across connections, so closing one connection cannot safely deregister its arena
while another Worker remains connected. Use the normal frontend shutdown to
complete this order. If device or native completion cannot be proven, Expert Kit
retains the affected storage until process exit and requires a process restart.
