# JR GB10 TensorFold Deployment

Baseline: JR-TensorFold-v1.0
Date: 2026-11-03

Hardware: Gigabyte NVIDIA GB10 / DGX Spark
Engine: MIA TensorFold v1.1.4
Image: tensorfold-qwen38:v1.1.4-eab8582

## 1. Production Configuration

Base Model:
nvidia/Qwen3.8-Flash-Next-FP4-Local

Model Revision:
b9ce18a6887e18c089fdea8509a33e63429a3300

Configuration:

- CHECKPOINT=nvfp4
- CONTEXT=262144
- PARALLEL=2
- KV_DTYPE=bf16
- DRAFTS=1
- VISION=1
- HOST=127.0.0.1
- PORT=8888
- SERVED_NAME=Qwen3.8-Flash-Next
- TENSORFOLD_MEMORY_RESERVE_GIB=7

DRAFTS=1 means MTP is enabled.

Observed MTP engine settings:

- TF_FLASHNEXT_DRAFTS=6
- TF_FLASHNEXT_CONFIDENCE=-0.4
- TF_FLASHNEXT_PRODUCT_STREAMS=5

IMPORTANT:
KV Cache uses BF16, NOT FP8.

The .env snapshot is stored in:

jr-config/JR-GB10.env.example

The actual running container configuration is stored in:

jr-config/JR-runtime-baseline.md


## 2. Model Modes

Three model modes are available.

All modes share the original NVIDIA FP4 checkpoint.

### Mode A: MIA Original Abliteration

Old Keys QSA-only overlay.

- 12 QSA layers
- 24 replacement tensors
- Approximately 190 MB sidecar
- Uses MIA ablit_view.py

Current .env contains ABLIT=1.

Start command:

cd ~/projects/gb10-tensorfold

ABLIT=1 ./start.sh

Requires the original MIA sidecar to be prepared.

### Mode B: Latest Keys QSA+GDN

Successfully tested on 2026-11-03.

- 12 QSA layers
- 36 GDN layers
- 96 replacement tensors
- Approximately 743 MB sidecar
- Uses graft.py
- Original NVIDIA checkpoint remains unchanged

Start command:

cd ~/projects/gb10-tensorfold

PREPARE=0 \
CHECKPOINT=nvfp4 \
ABLIT=0 \
MODEL_ID=local/Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47 \
MODEL_REVISION=house \
./start.sh

IMPORTANT:

ABLIT=0 disables MIA's built-in old overlay.

The latest Keys overlay remains active because it is
already included in the grafted checkpoint view.

New model cache:

/home/james/gb10-tensorfold-data/hf/hub/models--local--Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47

Snapshot revision:

house

Sidecar source:

drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47

File:

fp4-local/house-answer-qsa-gdn-mxfp8.safetensors

Graft script:

fp4-local/graft.py

Local download directory:

~/projects/gb10-tensorfold/keys-fp4-qad/

NOTE:
The model cache and sidecar are NOT stored in Git.
They must be downloaded or reconstructed separately.

### Mode C: NVIDIA Original FP4

No abliteration.

cd ~/projects/gb10-tensorfold

ABLIT=0 \
MODEL_ID= \
MODEL_REVISION= \
./start.sh

This mode uses the original NVIDIA FP4 QAD checkpoint.


## 3. IMPORTANT: Startup Behaviour

The current .env contains:

ABLIT=1

Therefore the default .env selects Mode A.

The last successfully tested running container used Mode B,
which was selected through temporary environment overrides.

Running container state and systemd default startup state
must be treated separately.

Configuration precedence:

1. Process environment
2. .env
3. Script defaults

If the TensorFold container is already running,
./start.sh without restart does not switch models.

The commands above are model selection examples,
not safe hot-switch commands.

When systemd is active, stop or change the managed
service through systemd before changing model modes.

Do not launch competing TensorFold instances.

Do not change Hindsight embedding or reranker services.


## 4. Systemd

Service:

gb10-tensorfold.service

Saved unit:

jr-config/systemd/gb10-tensorfold.unit.txt

Saved service metadata:

jr-config/systemd/gb10-tensorfold.status.txt

Preflight script, if installed:

jr-config/systemd/gb10-tensorfold-preflight.sh

Check service:

systemctl status gb10-tensorfold.service

Check complete service configuration:

systemctl cat gb10-tensorfold.service

Check automatic startup:

systemctl is-enabled gb10-tensorfold.service

IMPORTANT:

Temporary MODEL_ID and MODEL_REVISION overrides used
in manual commands are not automatically saved to systemd.

Verify the effective systemd startup configuration before
rebooting or making Mode B the permanent default.

The old SGLang deployment must not start automatically
alongside TensorFold.

The Hindsight embedding and reranker production services
must remain running and unchanged.


## 5. Tested MTP Performance

Test date: 2026-11-03

Engine: TensorFold v1.1.4
Model: Keys QSA+GDN
KV: BF16
Context: 262144
MTP: 6 Drafts / Confidence -0.4

Chinese Technical Answer:

MTP OFF: 24.67 tok/s
MTP ON:  45.35 tok/s
Improvement: +83.8%
Acceptance: 48.50%
Tokens/Round: 2.58

Python Code Generation:

MTP OFF: 24.82 tok/s
MTP ON:  63.91 tok/s
Improvement: +157.5%
Acceptance: 67.17%
Tokens/Round: 4.06

MTP ON/OFF output hashes matched in both tests.

Both tests were limited to 768 output tokens.
Results are single-run measurements, not a full benchmark.

Decision:

Keep current MTP settings unchanged.


## 6. Verification Commands

Check running model:

docker inspect qwen38-flash-next-tf \
  --format '{{json .Config.Cmd}}' \
  | python3 -m json.tool --no-ensure-ascii

Check API:

curl -fsS http://127.0.0.1:8888/health

Check models:

curl -fsS http://127.0.0.1:8888/v1/models \
  | python3 -m json.tool --no-ensure-ascii

Check memory:

free -h

Check GPU:

nvidia-smi

Check logs:

docker logs --tail 100 qwen38-flash-next-tf


## 7. Recovery Notes

If rebuilding this system:

1. Restore the MIA TensorFold repository and matching patch set.
2. Restore the documented .env configuration.
3. Ensure NVIDIA FP4 checkpoint exists in HF_CACHE.
4. Download the Keys QSA+GDN sidecar if using Mode B.
5. Run graft.py to reconstruct the local checkpoint view.
6. Restore and review the saved systemd unit and preflight.
7. Verify image, memory, model path and environment variables.
8. Start TensorFold through the configured service.
9. Verify /health and actual inference.
10. Confirm Hermes connectivity and Hindsight availability.

Do not copy the saved systemd unit blindly over an existing
working service. Review paths, dependencies and permissions.

Original checkpoint and existing services must be preserved.


## 8. Operational Rules

- Preserve NVIDIA original FP4 weights.
- Preserve working TensorFold Docker image.
- Preserve Hindsight production services.
- Do not run TensorFold and SGLang simultaneously.
- Do not commit model weights, tokens or credentials to Git.
- Do not change production settings without recording results.
- Diagnose inference problems using metrics and logs first.
- Preserve a rollback path before modifying systemd.

END OF JR BASELINE
