# JR GB10 TensorFold — NVIDIA FP4 + Keys Abliteration

> **JR verified baseline · 2026-11-03** · Gigabyte GB10 / DGX Spark · TensorFold v1.1.4 · **BF16 KV cache** · **262,144 context** · MTP enabled
>
> **JR 定制重点：** 保留 NVIDIA 原版 FP4 QAD 权重；在 MIA TensorFold 上运行 **Keys 最新 QSA+GDN（96 张量）消融外挂**，同时支持 MIA 自带的 **`ABLIT=1` QSA-only（24 张量）**。保留全部参数说明，并记录 BF16、MTP 和 systemd 运行基线。

This is a **JR-tested deployment profile**, not a new inference engine or a new set of ablation weights. [TensorFold](https://github.com/ashhart/TensorFold) is by Ash Hart and contributors; the GB10 runtime recipe and optimization patches are from [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold); the Keys weights and `graft.py` are from [drowzeys](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47). **JR's work is the integration, verification, runtime configuration and recovery documentation.**

## 1. NVIDIA FP4 QAD — the only checkpoint family covered here

| Component | Details |
| --- | --- |
| Base checkpoint | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
| Pinned stock revision | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Format | NVIDIA quantization-aware-distilled FP4: NVFP4 routed experts, MXFP8 attention/shared-expert projections, mixed precision for other components |
| Stock disk size | Approximately 96 GiB, reused by both overlay modes |
| Inference | TensorFold `tensorfold-native` v1.1.4 / CUDA on NVIDIA GB10 |
| Served API model name | `Qwen3.8-Flash-Next` |

Only **NVIDIA FP4 QAD** is documented here; other checkpoint families and historical MIA benchmark comparisons are intentionally omitted.

## 2. Abliteration: **JR QSA+GDN** versus **MIA `ABLIT=1`**

| Feature | **JR verified: Keys QSA+GDN** | MIA built-in: QSA-only |
| --- | --- | --- |
| Selection | **`ABLIT=0` + custom `MODEL_ID`** | **`ABLIT=1`** |
| Attention changes | **12 QSA + 36 GDN layers** | 12 QSA layers |
| Replacement tensors | **96** (48 weight/scale pairs) | 24 (12 weight/scale pairs) |
| Sidecar | **~743 MB** | ~190 MB |
| Loader | Keys author's **`graft.py`** checkpoint view | MIA `scripts/ablit_view.py` |
| Original NVIDIA weights | Unmodified; reused with hardlinks | Unmodified; reused through symlinks |
| JR GB10 validation | **96/96 index mappings; dtype/shape; model loading; `/health`; actual inference; MTP ON/OFF** | Not separately benchmarked on this JR baseline |

Both sidecars are published under the [Keys Hugging Face repository](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47). Access may require accepting the author's terms. Successful inference is not proof that every reasoning, safety or Agent workflow remains unchanged.

### **JR mode — latest Keys QSA+GDN (tested on GB10)**

```bash
cd ~/projects/gb10-tensorfold

PREPARE=0 \
CHECKPOINT=nvfp4 \
ABLIT=0 \
MODEL_ID=local/Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47 \
MODEL_REVISION=house \
./start.sh
```

**Why `ABLIT=0`?** It disables **only MIA's built-in 24-tensor QSA loader**. The new **96-tensor QSA+GDN** patch remains active because the already grafted checkpoint index redirects those tensors to the Keys sidecar. `PREPARE=0` does not download or generate the grafted view.

- Sidecar: `fp4-local/house-answer-qsa-gdn-mxfp8.safetensors`
- Graft tool: author's `fp4-local/graft.py`
- Local HF snapshot: `/home/james/gb10-tensorfold-data/hf/hub/models--local--Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47/snapshots/house`
- Stock revision must match the pinned NVIDIA checkpoint above.

See [JR-Start-README.md](JR-Start-README.md) for download, grafting, verification and recovery details. The model files themselves are **not committed to Git**.

### MIA mode — built-in `ABLIT=1` (QSA-only)

```bash
cd ~/projects/gb10-tensorfold
ABLIT=1 MODEL_ID= MODEL_REVISION= ./start.sh
```

MIA's older ~190 MB QSA-only sidecar must already be prepared. If not, follow the upstream `PREPARE=1` preparation instructions with authorized Hugging Face access. This is **not** the 96-tensor JR-tested configuration.

**Stock FP4 reference (no overlay):** `CHECKPOINT=nvfp4 ABLIT=0 MODEL_ID= MODEL_REVISION= ./start.sh`.

> **Boot and service warning / 开机注意：** These commands select models; they are **not hot-switch procedures**. Do not start another TensorFold instance while `gb10-tensorfold.service` owns the container. The [versioned `.env` snapshot](jr-config/JR-GB10.env.example) contains **`ABLIT=1`**, while the [captured running container](jr-config/JR-runtime-baseline.md) used the **JR Keys QSA+GDN** view via temporary overrides. The saved [systemd unit](jr-config/systemd/gb10-tensorfold.unit.txt) does **not** persist the custom `MODEL_ID` / `MODEL_REVISION`. A reboot could select the MIA old overlay until the service is explicitly updated. **Hindsight embedding and reranker services must remain unchanged.**

## 3. **JR verified GB10 configuration**

| Setting | **JR effective value** | Purpose |
| --- | --- | --- |
| Image | `tensorfold-qwen38:v1.1.4-eab8582` | MIA TensorFold 1.1.4 |
| Model | **Keys QSA+GDN, 96 tensor overrides** | Grafted NVIDIA FP4 view |
| `CHECKPOINT` | `nvfp4` | FP4 QAD only |
| `KV_DTYPE` | **`bf16`** | JR deliberately uses BF16 instead of MIA's FP8 default |
| `CONTEXT` | **`262144`** | Native 256K-token context |
| `PARALLEL` | **`2`** | Max concurrent requests |
| `DRAFTS` | `1` | Enable MTP |
| `TF_FLASHNEXT_DRAFTS` | `6` | Maximum MTP draft depth |
| `TF_FLASHNEXT_CONFIDENCE` | `-0.4` | Cumulative-confidence draft stopping |
| `TF_FLASHNEXT_PRODUCT_STREAMS` | `5` | Stream limit for cumulative-confidence rule |
| `VISION` | `1` | Image/video input enabled |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | **`7`** | Tighter memory reserve; monitor available memory |
| `HOST` / `PORT` | `127.0.0.1` / `8888` | Loopback-only API |

### JR MTP measurements (one run, 768 output tokens per test)

| Prompt | MTP OFF | **MTP ON** | Speed-up | Acceptance rate |
| --- | ---: | ---: | ---: | ---: |
| Chinese technical explanation | 24.67 tok/s | **45.35 tok/s** | **+83.8%** | 48.50% |
| Python code generation | 24.82 tok/s | **63.91 tok/s** | **+157.5%** | 67.17% |

Identical fixed-seed output hashes were observed for MTP ON/OFF on these two **truncated** test outputs. They are measurements of this one JR configuration, not general performance guarantees or official MIA benchmarks.

### BF16 KV versus FP8 KV (upstream per-sequence figures)

| KV format | Bytes per token | One 262,144-token sequence |
| --- | ---: | ---: |
| **BF16 — JR selected** | **30,784** | **7.52 GiB** |
| FP8 — MIA default | 17,888 | 4.37 GiB |

The values are KV cache sizes only. They are **not** the total GB10 process memory requirement. The effective available cache depends on other services, prompt cache, weights and workspace allocations.

## 4. Full configuration reference — **all MIA parameter descriptions retained**

MIA's original parameter list is preserved below, with a column identifying the **JR effective profile**. Environment variables override `.env`, which overrides script defaults. For implementation details, see [`scripts/config.sh`](scripts/config.sh).

| Variable | MIA default | **JR profile** | Meaning |
| --- | --- | --- | --- |
| `PARALLEL` | `5` | **`2`** | Requests decoded together. One that would not fit (prompt plus `max_tokens`) is refused. |
| `CONTEXT` | `262144` | **`262144`** | Prompt + reply window; 262,144 is the native maximum for this tested Flash Next v1.1.4 build. |
| `DRAFTS` | `1` | `1` | MTP drafts; `0` passes `--no-drafts`. |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Same | Default sampling; a request's own values win. |
| `THINKING` | `1` | `1` | Think before answering; `0` passes `--no-thinking`. |
| `MAX_TOKENS` | `32768` | `32768` | Reply budget when a request sets none. |
| `VISION` / `VISION_URLS` | `1` / `0` | `1` / `0` | Image and video (`--vision`); `VISION_URLS=1` also accepts public `https://` URLs. |
| `TENSORFOLD_MAX_IMAGES` / `TENSORFOLD_IMAGE_TOKENS` | `50` / `16384` | Same | Images a request may carry, and tokens shared by them (4,096 at most per image). |
| `MAX_VIDEOS` | `4` | `4` | Videos a request (`--vision-max-videos`). |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | `16384` | Video token budget per request. |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | `2048` | `2048` | Kept out of the KV budget for the vision helper. |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | Same | Model ID in `/v1/models`. |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | **`8888` / `127.0.0.1`** | API listen address and port. |
| `KV_DTYPE` | `fp8` | **`bf16`** | FP8 is lossy (~98.8% top-1 agreement with BF16); BF16 uses more memory but avoids FP8 KV quantization effects. Works with `VISION=1`. Adapted from MIA's GLM FP8 KV recipe. |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `9` | **`7`** | Memory floor while sizing the KV pool. JR uses a lower reserve; monitoring is essential. |
| `TF_FLASHNEXT_DRAFTS` | `6` | `6` | Maximum drafts per round. |
| `TF_FLASHNEXT_CONFIDENCE` / `TF_FLASHNEXT_PRODUCT_STREAMS` | `-0.4` / `5` | Same | Product of draft probabilities must stay ≥0.4 up to five drafting streams (0.5 per-draft floor above that). `0.6` restores v1.1.4 per-draft rule (MIA patch 0017). |
| `TF_FLASHNEXT_SAMPLED_HEADS` / `TF_FLASHNEXT_SAMPLE_CAND` | `1` / `1` | Same | Sampled requests join the batched MTP head pass; top-k candidates are picked on GPU (patches 0007, 0008). |
| `TF_FLASHNEXT_HOST_DRAW_THREADS` | `7` | `7` | Worker threads for host-side sampled draws (patch 0011); bit-identical to one thread. |
| `TF_FLASHNEXT_KEEP_LATE` | `1` | `1` | Partly rejected draft windows do not replay DeltaNet state; kept rows are applied next round (patch 0012). |
| `TF_FLASHNEXT_GATHER_EARLY` / `TF_FLASHNEXT_GATHER_READ` | `1` / `pread` | Same | Gather n-gram rows across lanes early, reading with `pread` thread pool instead of page-faulting `mmap` (patches 0010, 0013). |
| `TF_FLASHNEXT_DRAFT_AHEAD` / `TF_FLASHNEXT_WARM_AHEAD` | `1` / `1` | Same | Read ahead draft token n-gram rows and the next prompt chunk. |
| `TF_FLASHNEXT_PREFILL_TAIL` | `512` | `512` | Even out the last two prompt chunks if the final one would be under this many rows. |
| `TF_FLASHNEXT_PROMPT_HC2` / `TF_FLASHNEXT_PROMPT_SHARED_DIRECT` | `1` / `1` | Same | Fused hyper-connection readout and direct shared-expert output (patch 0014); bit-identical. |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS` / `TF_FLASHNEXT_HEAD_PIECE` | `256` / `1536` | Same | Shared-memory tiles for routed experts and prompt pieces for the MTP head (patch 0014). |
| `TF_FLASHNEXT_PROMPT_GEMM2` | `1` | `1` | Two-stage MX128 prompt tile (patch 0015), 15–25% faster per call in upstream measurements, same bits. |
| `TF_FLASHNEXT_SHARED_SIDE` / `TF_FLASHNEXT_GDN_LANES` | `1` / `1` | Same | Shared expert runs on a second stream and lane-wise DeltaNet rows use fewer launches (requires `KEEP_LATE=1`, patch 0016). |
| `CHECKPOINT` | `nvfp4` | `nvfp4` | NVIDIA FP4 QAD checkpoint for this profile. Custom `MODEL_ID` / `MODEL_REVISION` select the grafted FP4 view. |
| `MODEL_REVISION` | `b9ce18a6…` | **`house` in JR mode** | Stock checkpoint revision is pinned; JR's local graft snapshot revision is `house`. |
| `ABLIT` | `0` | **`0` JR / `1` MIA** | `1` enables MIA's gated QSA-only overlay; `0` disables MIA overlay loader. JR mode uses its own grafted checkpoint. |
| `MODEL_ID` | NVIDIA FP4 model | **`local/Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47`** | Custom checkpoint view selection. |
| `PREPARE` / `DRY_RUN` | Script-dependent / `0` | `0` / `1` for preview | `PREPARE=0` skips the preparation phase; `DRY_RUN=1` prints the launch command without starting or stopping a container. |
| `HF_CACHE` | Host-dependent | `/home/james/gb10-tensorfold-data/hf` | Hugging Face checkpoint cache, mounted read-only to the inference container. |
| `KERNEL_CACHE` | Host-dependent | `/home/james/gb10-tensorfold-data/kernel` | TensorFold kernel cache. |
| `STATE_DIR` | Host-dependent | `/home/james/gb10-tensorfold-data/state` | Service state directory. |
| `LOG_DIR` | Host-dependent | `/home/james/gb10-tensorfold-data/logs` | Service logs directory. |

## 5. Verification and recovery

```bash
# Preview only: no container starts or stops
DRY_RUN=1 PREPARE=0 CHECKPOINT=nvfp4 ABLIT=0 \
  MODEL_ID=local/Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47 \
  MODEL_REVISION=house ./start.sh

# Check actual model path (the served API name alone cannot distinguish modes)
docker inspect qwen38-flash-next-tf --format '{{json .Config.Cmd}}' \
  | python3 -m json.tool --no-ensure-ascii

curl -fsS http://127.0.0.1:8888/health
systemctl cat gb10-tensorfold.service
```

Detailed notes: [JR-Start-README.md](JR-Start-README.md) · [BF16 environment snapshot](jr-config/JR-GB10.env.example) · [Docker runtime snapshot](jr-config/JR-runtime-baseline.md) · [systemd unit](jr-config/systemd/gb10-tensorfold.unit.txt).

Keep Hindsight embedding and reranker running without changes. Local `.env`, `hf_token`, checkpoint shards and heavy sidecars are excluded from Git. This repository preserves configuration and verification data, not model binaries.

## Credits and license

**JR work:** GB10 deployment integration, **BF16/256K/2-concurrent** configuration, **96-tensor Keys checkpoint-view compatibility validation**, MTP experiments, systemd preflight and recovery documentation. **Not JR-authored:** TensorFold engine, MIA optimization patches, Keys weights or `graft.py`.

Upstream projects: [TensorFold](https://github.com/ashhart/TensorFold) · [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) · [Keys/drowzeys](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47). See [LICENSE](LICENSE), [NOTICE](NOTICE), [CREDITS.md](CREDITS.md), and the checkpoint / sidecar terms.
