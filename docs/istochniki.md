# Источники

Дата сбора данных: **25 сентября 2026**. Рынок меняется быстро: перед покупкой или долгой арендой
перепроверяйте цены и версии.

## Проверено напрямую (первоисточники)

| Что | Источник |
|---|---|
| Рецепт vLLM для Qwen3.8-27B: варианты весов и их размеры, проверенные команды для RTX 5090 / RTX PRO 6000 / DGX Spark / GB300, объём KV-кэша на 2× RTX 5090, приёмка MTP | [vllm-project/recipes: models/Qwen/Qwen3.8-27B.yaml](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml) (обновлён 2026-09-14) |
| Рецепт vLLM для Qwen3.6-27B: FP8 на одной 40-ГБ GPU, NVFP4 на Blackwell, `VLLM_USE_DEEP_GEMM=0` для RTX PRO 6000, пометка «prefix caching (Mamba) — experimental, align mode» | [vllm-project/recipes: models/Qwen/Qwen3.6-27B.yaml](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-27B.yaml) (обновлён 2026-09-17) |
| Общий гайд vLLM по Qwen3.5/3.6: MTP снижает пропускную способность при высокой конкуренции, ошибка `num_cache_lines >= batch` и `--max-cudagraph-capture-size` | [vllm-project/recipes: Qwen/Qwen3.5.md](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md) |
| Версии движков: vLLM 0.30.0 (2026-09-22), SGLang 0.5.20 (2026-09-18) | [pypi.org/project/vllm](https://pypi.org/project/vllm/), [pypi.org/project/sglang](https://pypi.org/project/sglang/) |
| Подключение Qwen Code к своему OpenAI-совместимому серверу (`modelProviders`, `baseUrl`, `envKey`, `generationConfig`) | [QwenLM/qwen-code: docs/users/configuration/model-providers.md](https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/model-providers.md), [auth.md](https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/auth.md) |
