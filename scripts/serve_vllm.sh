#!/usr/bin/env bash
# Запуск vLLM с Qwen3.8-27B под рой агентов (OpenAI-совместимый API на порту 8000).
#
#   ./scripts/serve_vllm.sh <профиль> [дополнительные флаги vllm serve ...]
#
# Профили (ёмкость — оценка scripts/vram_calc.py с MTP; точная цифра будет в логе vLLM при старте):
#   h200        1× H200 141GB               FP8 веса + FP8 KV   ≈ 10 агентов × 256K (впритык)
#   b200        1× B200 180GB               FP8 веса + FP8 KV   ≈ 14 агентов × 256K
#   2xh100      2× H100 80GB (TP=2)         FP8 веса + FP8 KV   ≈ 12 агентов × 256K
#   2xpro6000   2× RTX PRO 6000 96GB (TP=2) FP8 веса + FP8 KV   ≈ 15 агентов × 256K
#   pro6000     1× RTX PRO 6000 96GB        NVFP4 + FP8 KV      ≈ 7 агентов × 256K (или 10 × ~180K)
#   h100        1× H100 80GB                FP8 веса + FP8 KV   ≈ 4 агента × 256K (или 10 × ~110K)
#   2x5090      2× RTX 5090 32GB (TP=2)     NVFP4 + FP8 KV      ≈ 3 агента × 256K (или 10 × ~85K)
#   5090        1× RTX 5090 32GB            NVFP4 + FP8 KV      контекст 32K — только для экспериментов
#
# Переменные окружения (все необязательные):
#   PORT=8000            порт API
#   API_KEY=...          ключ доступа к API (очень желательно, если порт виден из сети)
#   MTP=3                сколько токенов угадывает MTP (0 — выключить спекулятивное декодирование)
#   MAX_MODEL_LEN=...    максимальный контекст одного запроса (по умолчанию из профиля)
#   MAX_NUM_SEQS=16      максимум одновременных запросов
#   MODEL=...            свой чекпойнт вместо профильного
#   SERVED_NAME=qwen3.8-27b  имя модели в API
#   DOCKER=1             запустить в docker-образе (иначе — локально установленный vllm)
#   IMAGE=vllm/vllm-openai:v0.30.0   образ для DOCKER=1 (если не стартует — vllm/vllm-openai:qwen38 из рецепта)
#   DRY_RUN=1            только напечатать команду
set -euo pipefail

PROFILE="${1:-}"
if [[ -z "$PROFILE" || "$PROFILE" == "-h" || "$PROFILE" == "--help" ]]; then
  sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0
fi
shift

PORT="${PORT:-8000}"
MTP="${MTP:-3}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
BATCHED=8192
UTIL=0.92
TP=1
CTX=262144
SEQS=16
EXTRA=()
ENVS=()

case "$PROFILE" in
  h200)       M=Qwen/Qwen3.8-27B-FP8; BATCHED=16384 ;;
  b200)       M=Qwen/Qwen3.8-27B-FP8; BATCHED=16384 ;;
  h100)       M=Qwen/Qwen3.8-27B-FP8 ;;
  2xh100)     M=Qwen/Qwen3.8-27B-FP8; TP=2; BATCHED=16384 ;;
  # RTX PRO 6000 / RTX 5090 (sm120): внимание через FlashInfer. В замерах без него декод
  # на контексте 128K падал с ~88 до ~32 ток/с.
  2xpro6000)  M=Qwen/Qwen3.8-27B-FP8; TP=2; EXTRA+=(--attention-backend flashinfer) ;;
  pro6000)    M=nvidia/Qwen3.8-27B-NVFP4; EXTRA+=(--attention-backend flashinfer) ;;
  2x5090)     M=nvidia/Qwen3.8-27B-NVFP4; TP=2; UTIL=0.93; EXTRA+=(--attention-backend flashinfer) ;;
  5090)       M=nvidia/Qwen3.8-27B-NVFP4; CTX=32768; UTIL=0.93; SEQS=8
              # на одной 32-ГБ карте CUDA-графы не помещаются (см. рецепт vLLM)
              EXTRA+=(--attention-backend flashinfer --enforce-eager) ;;
  *) echo "Неизвестный профиль: $PROFILE (см. --help)"; exit 1 ;;
esac

MODEL="${MODEL:-$M}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$CTX}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$SEQS}"

ARGS=(
  "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$BATCHED"
  --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8               # KV-кэш в 8 битах: вдвое больше контекста в той же памяти
  --language-model-only              # без vision-энкодера: больше памяти под KV (агентам нужен только текст)
  --enable-prefix-caching            # кэш префиксов: повторный контекст не пересчитывается
  --enable-chunked-prefill           # длинные промпты кусками, не блокируя генерацию у других агентов
  --async-scheduling
  --enable-prompt-tokens-details     # сообщать cached_tokens в ответах (для bench_swarm.py)
  --reasoning-parser qwen3           # размышления отдельно от ответа
  --enable-auto-tool-choice --tool-call-parser qwen3_xml   # вызовы инструментов для агентов
)
if [[ "$MTP" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
fi
[[ -n "${API_KEY:-}" ]] && ARGS+=(--api-key "$API_KEY")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"} "$@")

if [[ "${DOCKER:-0}" == "1" ]]; then
  DENV=(); for e in ${ENVS[@]+"${ENVS[@]}"}; do DENV+=(-e "$e"); done
  [[ -n "${HF_TOKEN:-}" ]] && DENV+=(-e HF_TOKEN)
  CMD=(docker run --rm --gpus all --ipc=host -p "$PORT:$PORT"
       -v "$HOME/.cache/huggingface:/root/.cache/huggingface" ${DENV[@]+"${DENV[@]}"} "$IMAGE" "${ARGS[@]}")
else
  CMD=(env ${ENVS[@]+"${ENVS[@]}"} vllm serve "${ARGS[@]}")
fi

echo "Профиль: $PROFILE | модель: $MODEL | TP=$TP | контекст: $MAX_MODEL_LEN | MTP: $MTP"
printf '%q ' "${CMD[@]}"; echo
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
exec "${CMD[@]}"
