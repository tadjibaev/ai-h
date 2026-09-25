# 4. Запуск: пошагово

Движок — **vLLM 0.30.0** (вышел 22.09.2026). Альтернатива — SGLang (раздел 4.9).

---

## 4.0. Сначала — без GPU (5 минут, бесплатно)

Проверьте на своём компьютере, что клиентские скрипты работают, с мок-сервером:

```bash
git clone https://github.com/tadjibaev/ai-h && cd ai-h
pip install -r requirements.txt
python3 scripts/mock_server.py &
python3 scripts/bench_swarm.py --agents 10
kill %1
```

## 4.1. Арендуйте машину

1. Выберите GPU по [главе 3](03-zhelezo.md): для 10 × 256K — **1× H200**, **1× B200** или **2× RTX PRO 6000**.
2. Образ: Ubuntu + CUDA (любой шаблон вида «PyTorch/CUDA») или шаблон с Docker.
3. Диск: **от 100 ГБ** (модель FP8 — 31 ГБ, плюс кэш и образы). Если провайдер позволяет,
   храните модель на **постоянном томе**, чтобы не скачивать её при каждом запуске.
4. Доступ: SSH. Порт 8000 наружу **не открывайте** без ключа API (см. 4.6).

## 4.2. Установите vLLM

Вариант 1 — через `uv` (быстрый менеджер пакетов Python):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.12 ~/vllm && source ~/vllm/bin/activate
uv pip install "vllm==0.30.0" --torch-backend=auto
```

Вариант 2 — Docker (нужен NVIDIA Container Toolkit): `docker pull vllm/vllm-openai:v0.30.0`,
дальше запуск с `DOCKER=1` (см. 4.3).

> Рецепт vLLM для Qwen3.8 проверялся на специальном образе `vllm/vllm-openai:qwen38`.
> Все нужные для этой модели исправления вошли в релизы 0.28–0.30, поэтому 0.30.0 должно хватать.
> Если что-то не заводится, попробуйте `IMAGE=vllm/vllm-openai:qwen38 DOCKER=1 ...`.

## 4.3. Скачайте модель и запустите сервер

```bash
uv pip install -U huggingface_hub
hf download Qwen/Qwen3.8-27B-FP8          # или nvidia/Qwen3.8-27B-NVFP4 для профиля pro6000/5090

git clone https://github.com/tadjibaev/ai-h && cd ai-h
export API_KEY=$(openssl rand -hex 16); echo "API_KEY=$API_KEY"   # сохраните ключ
./scripts/serve_vllm.sh h200             # профили: h200, b200, 2xh100, 2xpro6000, pro6000, h100, 2x5090, 5090
# в Docker:  DOCKER=1 ./scripts/serve_vllm.sh h200
# посмотреть команду, не запуская:  DRY_RUN=1 ./scripts/serve_vllm.sh h200
```

Первый запуск занимает несколько минут: загрузка весов, компиляция, захват CUDA-графов.

## 4.4. Что проверить в логе при старте

| Строка лога | Что означает | Ожидаемо (H200, FP8, MTP-3) |
|---|---|---|
| `Setting attention block size to 1600 tokens ...` | шаг кэша префиксов для гибридной модели | 1600 (1568 без MTP; 800/784 при BF16 KV) |
| `GPU KV cache size: N tokens` | сколько токенов контекста помещается суммарно | ~3M |
| `Maximum concurrency for 262,144 tokens per request: X.XXx` | сколько агентов с полным контекстом поместится | ≥ 10 |
| предупреждение про FP8 KV scales = 1.0 | масштабы FP8 не откалиброваны | это нормально, см. [02-model.md, 2.4](02-model.md) |

Если `Maximum concurrency` меньше числа агентов — см. главу 7.2 (как поднять ёмкость).

## 4.5. Проверка, что всё работает

```bash
curl -s localhost:8000/v1/models -H "Authorization: Bearer $API_KEY"

curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Ответь одним предложением: кто ты?"}],
    "max_tokens": 200,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

python3 scripts/bench_swarm.py --api-key $API_KEY --agents 10     # быстрый нагрузочный тест
```

Целевой сценарий и как читать результаты — в [главе 7](07-zamery-i-tyuning.md).

## 4.6. Подключение с вашего компьютера (безопасно)

Самый безопасный путь — SSH-туннель, порт наружу не открывается:

```bash
ssh -N -L 8000:localhost:8000 user@ваш-сервер
# теперь на вашем компьютере: base_url = http://localhost:8000/v1, api_key = $API_KEY, model = qwen3.8-27b
```

Если порт всё-таки открываете в интернет, **обязательно** запускайте с `API_KEY` и по возможности
ограничьте доступ по IP в панели провайдера.

---

## 4.7. Что делает каждый флаг (scripts/serve_vllm.sh)

| Флаг | Зачем |
|---|---|
| `--kv-cache-dtype fp8` | KV-кэш в 8 битах: вдвое больше контекста |
| `--language-model-only` | не грузить vision-энкодер: +1–2 ГБ под KV (на RTX 5090 ёмкость росла с 91K до 136K токенов) |
| `--enable-prefix-caching` | кэш префиксов (в 0.30 включён по умолчанию, для гибрида — режим `align`) |
| `--enable-chunked-prefill` | длинные промпты считаются кусками и не останавливают генерацию у других агентов |
| `--max-num-batched-tokens 8192/16384` | размер куска префилла: больше — быстрее префилл, но заметнее пауза в декоде у остальных |
| `--max-num-seqs 16` | максимум одновременных запросов (по умолчанию 1024 — для роя слишком много) |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | MTP: ускорение декода в 1.5–2 раза |
| `--reasoning-parser qwen3` | размышления приходят отдельным полем `reasoning`, а не в тексте ответа |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | вызовы инструментов в формате OpenAI (для агентов) |
| `--enable-prompt-tokens-details` | сервер сообщает `cached_tokens` в каждом ответе |
| `--attention-backend flashinfer` | на RTX PRO 6000 / RTX 5090 — быстрое внимание на длинном контексте |
| `--gpu-memory-utilization 0.92` | доля памяти GPU под vLLM (значение по умолчанию) |
| `--api-key` | доступ только по ключу |

Полезные дополнительные флаги:
- `--kv-offloading-size 64` (ГиБ, суммарно на все GPU) — выгрузка KV-кэша в ОЗУ. Помогает,
  когда агентов больше, чем помещается на GPU: вернуть 250K контекста из ОЗУ — доли секунды,
  а пересчитать — около минуты. Режим `native` поддерживает гибридную модель.
- `--max-cudagraph-capture-size 64` — если при старте падает `assert num_cache_lines >= batch` (см. ниже).

---

## 4.8. Типичные проблемы

| Симптом | Причина и решение |
|---|---|
| OOM (нехватка памяти) при старте | уменьшите `MAX_MODEL_LEN` или `--gpu-memory-utilization 0.88`; проверьте, что нет других процессов на GPU (`nvidia-smi`) |
| `assert num_cache_lines >= batch` | размер захвата CUDA-графов больше кэша состояний: `--max-cudagraph-capture-size 64` |
| На RTX 5090 падает захват CUDA-графов | профиль `5090` уже включает `--enforce-eager` (так в рецепте vLLM) |
| Размышления попадают в текст ответа | нет `--reasoning-parser qwen3` |
| `Unexpected reasoning effort` | для Qwen3.8 допустимы только `low`, `medium`, `xhigh` |
| Вызовы инструментов приходят текстом | нужен `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (или `qwen3_coder`) |
| На длинном контексте декод резко падает (RTX PRO 6000/5090) | проверьте, что внимание идёт через FlashInfer (`--attention-backend flashinfer`) |
| Доля кэша на повторных ходах ~0% | агенты меняют начало промпта (дата, разные tools, разный `reasoning_effort`), см. [главу 5](05-roy-agentov.md). Известен баг vLLM #45238: при чередовании трафика кэш иногда промахивается |
| Низкая приёмка MTP (< 50%) | уменьшите `MTP=2`; на очень высокой нагрузке MTP можно выключить (`MTP=0`) |
| Долгий TTFT у всех агентов в начале | агенты стартовали одновременно без прогрева общего префикса (см. 5.2) |

---

## 4.9. Альтернатива: SGLang 0.5.20

SGLang тоже официально поддерживает Qwen3.8-27B (проверено на H200, RTX PRO 6000, RTX 5090, DGX Spark).
Плюсы: **HiCache** — многоуровневый кэш с выгрузкой KV и состояний в ОЗУ или на диск (удобно «парковать»
неактивных агентов), быстрый DFlash2. Минусы для нашей задачи: нельзя отключить vision-энкодер,
и по умолчанию очень много памяти уходит под состояния DeltaNet (см. ниже).

Команда для RTX PRO 6000 (из cookbook SGLang, с MTP):

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B-FP8 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-linear-replayssm-spec \
  --max-running-requests 16 --max-mamba-cache-size 80 \
  --host 0.0.0.0 --port 8000
```

- На H200: `--chunked-prefill-size 32768 --max-prefill-tokens 32768`, без `--enable-linear-replayssm-spec`
  (этот флаг только для RTX PRO 6000 / RTX 5090).
- **Состояния DeltaNet.** По умолчанию `--mamba-full-memory-ratio 0.9` отдаёт под них ~47% памяти
  после весов. Для длинных контекстов это расточительно. Задайте число слотов явно:
  `--max-mamba-cache-size` ≈ 5 × число одновременных агентов + запас. На каждый запрос с кэшем
  нужно ~5 слотов по 154 МБ.
- **HiCache:** `--enable-hierarchical-cache --hicache-size <ГБ ОЗУ> --page-size 64 --hicache-mem-layout page_first`.
- Не включайте `--enable-mixed-chunk`: есть сообщения, что он портит состояния DeltaNet.

Прямых сравнений SGLang и vLLM на этой модели в многоходовом агентном сценарии нет.
Если хотите выбрать по делу, прогоните `bench_swarm.py` на обоих: скрипт работает с любым
OpenAI-совместимым сервером.

## 4.10. Почему не llama.cpp / Ollama / LM Studio

Они поддерживают эту модель, но рассчитаны на одного пользователя. KV-кэш в f16 — ~64 КБ
на токен, у каждого параллельного «слота» свой лимит контекста. В одном сравнении на Qwen3.6-27B
vLLM был в 3–4 раза быстрее llama.cpp при большой конкуренции. Для 1–2 длинных сессий на домашнем
ПК — нормально, для роя из 10 агентов — нет.
