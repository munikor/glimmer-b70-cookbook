# LoRA serving on B70 (vLLM 0.31 / llm-scaler) — and torch sidecars on RDNA3

Validated 2026-10-29 on 2x Arc Pro B70 (TP2, FP8) serving Muse-Glimmer-30B.

## Serve a LoRA adapter next to the base model

vLLM serves base + adapters on one engine, routed by the `model` field per request —
no restarts, no swaps. Cost: adapter-size VRAM per adapter + one flag set.

Recipe: **do not edit the launcher script** — make a variant that injects the flags:

```bash
#!/usr/bin/env bash
# Full-parameter Muse Glimmer 30B server for two Intel Arc Pro B70 GPUs.
#
# The FP8 mode preserves every parameter using eight-bit floating-point
# weights plus floating-point scales. It does not use GGUF, pruning,
# distillation, or 4/5/6-bit quantization.
set -euo pipefail

: # scaler-031 port: ZE_AFFINITY_MASK intentionally NOT set (0,1 hides GPU1 on new L0; 0.0,1.0 breaks vllm int() parse; unset shows both)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_PATH="${MODEL_PATH:-/models/muse-glimmer-30b-bf16}"
ASSISTANT_PATH="${ASSISTANT_PATH:-/models/muse-glimmer-30b-assistant-official}"
PARSER_PATH="${PARSER_PATH:-/opt/glimmer/muse_glimmer_vllm_parser.py}"
PORT="${PORT:-8000}"
WEIGHT_PRECISION="${WEIGHT_PRECISION:-fp8}"

case "${WEIGHT_PRECISION}" in
  fp8)
    model_dtype="float16"
    served_model_name="muse-glimmer-30b-fp8"
    quantization_args=(--quantization fp8)
    ;;
  bf16)
    model_dtype="bfloat16"
    served_model_name="muse-glimmer-30b-bf16"
    quantization_args=()
    ;;
  *)
    echo "Unsupported WEIGHT_PRECISION=${WEIGHT_PRECISION}; use fp8 or bf16." >&2
    exit 2
    ;;
esac

default_speculative_config=$(printf \
  '{"model":"%s","method":"dflash","num_speculative_tokens":15,"draft_tensor_parallel_size":2,"use_local_argmax_reduction":true}' \
  "${ASSISTANT_PATH}")
default_compilation_config='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[16],"max_cudagraph_capture_size":16}'

args=(
  vllm serve
  --host 0.0.0.0
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME:-${served_model_name}}"
  --model-impl "${MODEL_IMPL:-auto}"
  --dtype "${model_dtype}"
  --tensor-parallel-size "${TP_SIZE:-2}"
  --gpu-memory-utilization "${GPU_UTIL:-0.74}"
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
  --max-num-seqs "${MAX_NUM_SEQS:-1}"
  --block-size "${BLOCK_SIZE:-64}"
  --chat-template "${MODEL_PATH}/chat_template.jinja"
  --enable-auto-tool-choice
  --tool-call-parser muse_glimmer
  --tool-parser-plugin "${PARSER_PATH}"
  --reasoning-parser muse_glimmer
  --reasoning-parser-plugin "${PARSER_PATH}"
)

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "${API_KEY}")
fi

if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  args+=(--language-model-only)
fi

args+=("${quantization_args[@]}")

if [[ "${KV_CACHE_DTYPE:-auto}" != "auto" ]]; then
  args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${KV_CACHE_DTYPE_SKIP_LAYERS:-}" ]]; then
  args+=(--kv-cache-dtype-skip-layers "${KV_CACHE_DTYPE_SKIP_LAYERS}")
fi

if [[ -n "${ROPE_SCALING:-}" ]]; then
  args+=(--rope-scaling "${ROPE_SCALING}")
fi

if [[ -n "${LOAD_FORMAT:-}" ]]; then
  args+=(--load-format "${LOAD_FORMAT}")
fi

if [[ -n "${PROFILER_CONFIG:-}" ]]; then
  args+=(--profiler-config "${PROFILER_CONFIG}")
fi

compilation_config="${COMPILATION_CONFIG:-${default_compilation_config}}"
if [[ -n "${compilation_config}" ]]; then
  args+=(--compilation-config "${compilation_config}")
fi

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  args+=(--enforce-eager)
fi

case "${ASYNC_SCHEDULING:-auto}" in
  auto) ;;
  1) args+=(--async-scheduling) ;;
  0) args+=(--no-async-scheduling) ;;
  *)
    echo "Unsupported ASYNC_SCHEDULING=${ASYNC_SCHEDULING}; use auto, 1, or 0." >&2
    exit 2
    ;;
esac

speculative_config="${SPECULATIVE_CONFIG:-${default_speculative_config}}"
if [[ "${DISABLE_SPECULATION:-0}" != "1" && -n "${speculative_config}" ]]; then
  args+=(--speculative-config "${speculative_config}")
fi


# --- LoRA probe block (added 2026-10-29, trinity) ---
if [[ "${ENABLE_LORA:-0}" == "1" ]]; then
  args+=(--enable-lora)
  [[ -n "${LORA_MODEL_PATH:-}" ]] && args+=(--lora-modules "goku-lora=${LORA_MODEL_PATH}")
  args+=(--max-loras "${MAX_LORAS:-1}" --max-lora-rank "${MAX_LORA_RANK:-32}")
fi

exec "${args[@]}"
```

Key deltas vs the plain script: `--enable-lora --max-loras --max-lora-rank
--lora-modules name=path`, and pay for the adapter by dropping `GPU_UTIL` a notch
(0.95 → 0.92). Adapter used: `lemuralabs/Muse-Glimmer-30B-GUI-Grounding-Fast`
(838 MB peft, rank 32) — byte-verify `adapter_model.safetensors` against the hub API
before booting (resume downloads corrupt size silently).

Systemd unit env (bridged into the container as **name-only `-e`** — systemd
`Environment=` alone does not reach `docker run`):

```ini
[Unit]

Description=Muse Glimmer 30B on Intel llm-scaler vLLM 0.31 (B70 x2)

After=podman.socket docker.service



[Service]

Environment=KV_CACHE_DTYPE=fp8

Environment=MAX_NUM_SEQS=8

Environment=MAX_BATCHED_TOKENS=8448

Environment=GPU_UTIL=0.92

Environment=ENABLE_LORA=1

Environment=LORA_MODEL_PATH=/models/lora-gui

Environment=MAX_LORAS=1

Environment=MAX_LORA_RANK=32

Restart=on-failure

RestartSec=10

TimeoutStartSec=1800

ExecStartPre=-/usr/bin/docker rm -f glimmer-scaler

ExecStart=/usr/bin/docker run --rm --name glimmer-scaler \

  --device /dev/dri --group-add 105 \

  --security-opt seccomp=unconfined --security-opt label=disable \

  --ipc=host --shm-size 32g \

  -p 8210:8200 \

  -v /home/mikean501/glimmer-test/models:/models:ro \

  -v /home/mikean501/glimmer-port:/opt/port:ro \

  -e PORT=8200 \

  -e SERVED_MODEL_NAME=glimmer \

  -e PARSER_PATH=/opt/port/muse_glimmer_vllm_parser.py \

  -e MAX_NUM_SEQS \

  -e MAX_BATCHED_TOKENS \

  -e GPU_UTIL \

  -e ENABLE_LORA \

  -e LORA_MODEL_PATH \

  -e MAX_LORAS \

  -e MAX_LORA_RANK \

  -e CCL_SYCL_ALLREDUCE_LL=twoshots \

  -e CCL_SYCL_ALLREDUCE_SIMPLE_READ=1 \

  -e CCL_SYCL_COPY_ENGINE=1 \

  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \

  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \

  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \

  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \

  --entrypoint /opt/port/serve-glimmer-scaler-lora.sh \

  intel/llm-scaler-vllm:0.31.0-b1-experimental

ExecStop=/usr/bin/docker stop glimmer-scaler



[Install]

WantedBy=default.target





[Service]

TimeoutStopFailureMode=abort
```

Verify, in order: `docker exec <name> printenv ENABLE_LORA` (must not be empty —
a `-e NAME` bridge with no `Environment=` value injects empty string, and `set -u`
scripts with `${VAR:-default}` then skip LoRA **silently**: boot succeeds, health 200,
no adapter); then `/v1/models` must list the adapter id; then a chat probe with
`"model": "<adapter>"`.

## Torch sidecars: RDNA3 box > B70

Kronos (plain-PyTorch forecaster) moved from the B70 box (`torch+xpu`) to a 7900 XTX
(`torch 2.9.1+rocm6.3`, `KRONOS_DEVICE=cuda`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`):
**much faster** (user-verified). B70/XPU is excellent for vLLM, young for raw torch
(sync overhead, kernel coverage). Default embeddings/rerankers/ASR/forecast sidecars
to a ROCm box; keep the B70s on the vLLM fast path:

```ini
[Unit]
Description=Kronos webui on ROCm (migrated from 175 2026-10-29)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/kronos-smoke/Kronos/webui
ExecStart=%h/kronos-smoke/rocmvenv/bin/python app.py
Environment=KRONOS_DEVICE=cuda
Environment=PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

## Notes
- Prefix caching is ON by default in llm-scaler 0.31 vLLM (`enable_prefix_caching=True` in engine args) — check engine args before adding flags for it.
- llama.cpp/hipfire-style single-slot servers generally have **no** LoRA support; check the binary (`strings <bin> | grep -i lora`) before planning adapter lanes.
