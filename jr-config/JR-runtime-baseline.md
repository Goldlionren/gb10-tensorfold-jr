# JR GB10 TensorFold Runtime Baseline

Captured: 2026-11-03T15:42:30+11:00

## Git revision
20e15e41f2f514a0d287710f4665ee1b5009ba41

## Docker image
tensorfold-qwen38:v1.1.4-eab8582

## Actual model command
[
    "tensorfold-native",
    "serve",
    "/root/.cache/huggingface/hub/models--local--Qwen3.8-Flash-Next-FP4-Local-house-answer-L0-47/snapshots/house",
    "--name",
    "Qwen3.8-Flash-Next",
    "--host",
    "127.0.0.1",
    "--port",
    "8888",
    "--context",
    "262144",
    "--parallel",
    "2",
    "--max-tokens",
    "32768",
    "--temperature",
    "1.0",
    "--top-p",
    "0.95",
    "--top-k",
    "20",
    "--thinking",
    "--kv-cache",
    "bf16",
    "--vision",
    "--vision-max-images",
    "50",
    "--vision-max-videos",
    "4",
    "--vision-image-tokens",
    "16384"
]

## MTP / Memory environment
TF_FLASHNEXT_DRAFTS=6
TF_FLASHNEXT_CONFIDENCE=-0.4
TENSORFOLD_MEMORY_RESERVE_GIB=7
TF_FLASHNEXT_PRODUCT_STREAMS=5

## Service enablement
gb10-tensorfold.service: enabled
gb10-flashnext.service: disabled
gb10-flashnext-proxy.service: disabled
