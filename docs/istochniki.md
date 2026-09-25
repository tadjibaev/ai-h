# Источники

Дата сбора данных: **25 сентября 2026**. Рынок меняется быстро: перед покупкой или долгой арендой
перепроверяйте цены и версии.

Как собиралось: исходный код и документация движков читались напрямую из GitHub (конкретные теги:
vLLM `v0.30.0`, SGLang `v0.5.20`); версии пакетов — с PyPI; цены облаков — из машинно-читаемых
каталогов (SkyPilot, снимки API Vast.ai), цены API — из прайс-карт LiteLLM и models.dev;
остальное — поиск. Сайты Hugging Face, документация vLLM/SGLang и сайты провайдеров из среды,
где готовился обзор, были недоступны, поэтому карточки моделей читались через зеркала на GitHub.

## Модель

| Что | Источник |
|---|---|
| Карточка Qwen3.8-27B: архитектура, бенчмарки, режимы размышлений, сэмплирование, YaRN | [huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (читалась через зеркала [HX-Ai-Platform](https://raw.githubusercontent.com/Hana-X-AI/HX-Ai-Platform/main/library/technologies/qwen/sources/qwen3.8-27b-card-1d4bf0f.md), [cardtrack](https://raw.githubusercontent.com/DouwMarx/cardtrack/main/site/docs/alibaba-qwen-qwen3-8-27b-model-card.html)) |
| config.json и chat template Qwen3.8-27B (4 KV-головы × 256, `mamba_ssm_dtype: float32`, `reasoning_effort`, `preserve_thinking`) | [pacell/llm-model-cards: Qwen__Qwen3.8-27B.json](https://raw.githubusercontent.com/pacell/llm-model-cards/main/data/model_cards/Qwen__Qwen3.8-27B.json) |
| Карточки Qwen3.6-27B и Qwen3.5 (сравнение) | [huggingface.co/Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [huggingface.co/Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Анонс Qwen3.8-27B от Alibaba Cloud | [alibabacloud.com/blog/…603463](https://www.alibabacloud.com/blog/alibaba-unveils-qwen3-8-27b-and-releases-weights-of-qwen3-8-flagship-model_603463) |
| Качество NVFP4 (NVIDIA) и INT4 (RedHat) против BF16 | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) |
| FP8 KV-кэш: точность на длинном контексте (Qwen3.5-27B) | [блог vLLM, 2026-04-22](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md) |
| Qwen 4: дорожная карта Alibaba | [alibabacloud.com/en/press-room/…](https://www.alibabacloud.com/en/press-room/alibaba-unveils-roadmap-on-full-stack-ai-strategy) |

## Движки

| Что | Источник |
|---|---|
| Рецепт vLLM для Qwen3.8-27B: варианты весов и размеры, проверенные команды, KV-ёмкость на RTX 5090, приёмка MTP | [vllm-project/recipes: Qwen3.8-27B.yaml](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml) |
| Рецепт vLLM для Qwen3.6-27B (NVFP4 на RTX PRO 6000 и DGX Spark, пометка про экспериментальный кэш) | [vllm-project/recipes: Qwen3.6-27B.yaml](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-27B.yaml), [Qwen3.5.md](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md) |
| vLLM 0.30.0: кэш префиксов для гибридных моделей (`mamba_cache_mode=align`), размер блока, учёт KV-ёмкости, выбор бэкенда внимания, выгрузка KV в ОЗУ, значения флагов по умолчанию | исходники на теге [v0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0): `vllm/config/cache.py`, `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/kv_cache_coordinator.py`, `vllm/platforms/cuda.py`, `vllm/engine/arg_utils.py`, `docs/features/automatic_prefix_caching.md` |
| vLLM: имена метрик Prometheus, поля `reasoning`/`reasoning_content` | `vllm/v1/metrics/loggers.py`, `vllm/entrypoints/chat_utils.py` (тег v0.30.0) |
| SGLang 0.5.20: cookbook Qwen3.8-27B, кэш состояний DeltaNet, HiCache | [sglang: docs/cookbook/…/Qwen3.8-27B.mdx](https://github.com/sgl-project/sglang/blob/v0.5.20/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx), `python/sglang/srt/mem_cache/…` |
| llama.cpp: поддержка qwen35, слоты, контрольные точки | [llama.cpp: tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) |
| Версии: vLLM 0.30.0 (2026-09-22), SGLang 0.5.20 (2026-09-18) | [pypi.org/project/vllm](https://pypi.org/project/vllm/), [pypi.org/project/sglang](https://pypi.org/project/sglang/) |
| Qwen Code: подключение своего OpenAI-совместимого сервера | [qwen-code: model-providers.md](https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/model-providers.md), [auth.md](https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/auth.md) |

## Замеры скорости (по ним откалиброван калькулятор)

| Что | Источник |
|---|---|
| 1× и 2× RTX PRO 6000, vLLM FP8 + MTP: декод, префилл 8K–128K, конкуренция, влияние FlashInfer | [local-inference-lab/rtx6kpro: qwen38-27b.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-27b.md) |
| 8× RTX PRO 6000: MTP при разной нагрузке, прогрев общего префикса (68.8 с → 0.65 с), SGLang NVFP4 196K | [helixml/ramjet: EXPERIMENTS.md](https://github.com/helixml/ramjet/blob/main/EXPERIMENTS.md) |
| 1× RTX PRO 6000, SGLang + DFlash2: реальный агентный трафик до 360K | [jpezzulli/sglang-rtxpro6000: RESULTS.md](https://github.com/jpezzulli/sglang-rtxpro6000/blob/master/RESULTS.md) |
| RTX 3090/4090/5090, 2× RTX 5090 | [noonghunna/club-3090: BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md), [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) |
| H100 | [avifenesh/memra: PERFORMANCE.md](https://github.com/avifenesh/memra/blob/main/docs/PERFORMANCE.md) |
| DGX Spark | [gitcommit90/qwen38-27b-dgx-spark](https://github.com/gitcommit90/qwen38-27b-dgx-spark), [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) |
| MI300X, Strix Halo, Mac | [arjhinety/brainwaves-longwriter-bench](https://github.com/arjhinety/brainwaves-longwriter-bench/blob/main/reports/FINAL_REPORT.md), [julianmb/q38rocm](https://github.com/julianmb/q38rocm), [Weschera/Qwen3.8-27B-oMLX-MTP-Mac](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac) |

## Цены

| Что | Источник |
|---|---|
| Аренда GPU: RunPod, Nebius, Lambda, Verda, Shadeform (API-цены, 25.09.2026) | [skypilot-org/skypilot-catalog](https://github.com/skypilot-org/skypilot-catalog/tree/master/catalogs/v8) |
| Vast.ai: медианы предложений (снимки API, 2.08–25.09.2026) | [guzus/ai-research-arm: gpu-spot.json](https://github.com/guzus/ai-research-arm/blob/main/research/market/gpu-spot.json) |
| Повышение цен Nebius с 1 октября 2026 | [investing.com](https://www.investing.com/news/stock-market-news/nebius-to-increase-prices-for-nvidia-gpu-resources-from-october-93CH-4905894) |
| API Qwen3.8-27B: OpenRouter, DeepInfra, Alibaba и др. | [openrouter.ai/qwen/qwen3.8-27b](https://openrouter.ai/qwen/qwen3.8-27b), [LiteLLM price map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), [models.dev](https://github.com/sst/models.dev/tree/dev/providers) |
| RTX PRO 6000: рост цены до $13 250 (июнь) и $16 000 (август 2026) | [tomshardware.com (август)](https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year), [tomshardware.com (июнь)](https://www.tomshardware.com/pc-components/gpus/nvidia-raises-rtx-pro-6000-blackwell-gpu-pricing-to-usd13-250-55-percent-increase-over-msrp-in-a-years-time) |
| RTX 5090: дефицит и цены $5–9.6K | [pcgamesn.com](https://www.pcgamesn.com/nvidia/rtx-5090-pricing-september-2026), [videocardz.com](https://videocardz.com/newz/geforce-rtx-5090-is-disappearing-from-stores-first-listing-hits-9600) |
| DGX Spark $4 699 | [techpowerup.com](https://www.techpowerup.com/346833/nvidia-raises-dgx-spark-pricing-to-usd-4-700) |
| Mac Studio: M3 Ultra 512GB снят, M5 Ultra | [macrumors.com](https://www.macrumors.com/2026/03/05/mac-studio-no-512gb-ram-upgrade/), [appleinsider.com](https://appleinsider.com/articles/26/08/25/you-can-spend-18299-on-a-mac-studio-today-or-more-in-october) |
| Рабочие станции с 2× RTX PRO 6000, цены на серверную DDR5 | [arsenalpc.com](https://arsenalpc.com/product/ai-workstations/enthoo-pro-2-server-edition-custom-ai-workstation-dual-rtx-pro-6000-blackwell-96gb-192gb-total-ddr5-256gb-ryzen-threadripper-pro-9985wx-64c-3-2ghz-8tb-nvme-ssd-2x4tb-raid/), [datacenterdisk.com](https://datacenterdisk.com/server-ram/ddr5/64gb) |
| Электричество: Узбекистан (с 1.06.2026), США | [tashkenttimes.uz](https://www.tashkenttimes.uz/national/17651-government-approves-june-1-electricity-and-gas-rate-increases), [electricchoice.com](https://www.electricchoice.com/electricity-prices-by-state/) |
