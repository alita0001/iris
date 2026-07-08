#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the policy model (used by
# `python -m revact.cli eval-rollout`). Everything is overridable via env vars:
#   VLLM_PORT=8300 LORA_MODULES="iris-sft=outputs/sft_lora_p0" scripts/start_vllm.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen25-3b}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-/home/user/miniconda3/envs/qwen-vllm}"
PYTHON_BIN="${VLLM_PYTHON:-$VLLM_ENV_DIR/bin/python}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8300}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# "name=path name=path ..." pairs; when set, --enable-lora is added so the
# server exposes base and adapter as two selectable model names.
LORA_MODULES="${LORA_MODULES:-}"
LOG_DIR="${LOG_DIR:-outputs/vllm_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm-$PORT.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/vllm-$PORT.pid}"

mkdir -p "$LOG_DIR"
export PATH="$VLLM_ENV_DIR/bin:$PATH"

# vLLM JIT-compiles against the wheel-bundled CUDA toolkit inside the conda
# env; CUDA_HOME must point there or compilation picks up a mismatched nvcc.
NVIDIA_CUDA_HOME="${NVIDIA_CUDA_HOME:-$VLLM_ENV_DIR/lib/python3.11/site-packages/nvidia/cu13}"
if [ -d "$NVIDIA_CUDA_HOME" ]; then
  export CUDA_HOME="${CUDA_HOME:-$NVIDIA_CUDA_HOME}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# FlashInfer's sampling JIT path is fragile when the wheel-bundled CUDA
# headers and nvcc versions do not match exactly.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [ -z "${CC:-}" ] && [ -x "$VLLM_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc" ]; then
  export CC="$VLLM_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc"
fi
if [ -z "${CXX:-}" ] && [ -x "$VLLM_ENV_DIR/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CXX="$VLLM_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Model path not found or incomplete: $MODEL_PATH" >&2
  exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Existing process:" >&2
  lsof -i :"$PORT" >&2
  exit 1
fi

extra_args=()
if [ -n "$LORA_MODULES" ]; then
  # shellcheck disable=SC2206
  extra_args+=(--enable-lora --lora-modules $LORA_MODULES)
fi

echo "Starting vLLM:"
echo "  model path:  $MODEL_PATH"
echo "  served name: $MODEL_NAME"
echo "  endpoint:    http://127.0.0.1:$PORT/v1"
echo "  lora:        ${LORA_MODULES:-<none>}"
echo "  log:         $LOG_FILE"

setsid "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${extra_args[@]}" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")  (kill \$(cat $PID_FILE) to stop)"
echo "Use: tail -f $LOG_FILE"
