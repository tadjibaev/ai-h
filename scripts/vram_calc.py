#!/usr/bin/env python3
"""
Калькулятор памяти, скорости и стоимости для Qwen3.x-27B под «рой агентов» (vLLM).

Архитектура Qwen3.5/3.6/3.8-27B гибридная: из 64 слоёв только 16 — «полное внимание»
(им нужен KV-кэш, растущий с длиной контекста), остальные 48 — Gated DeltaNet
(линейное внимание с ФИКСИРОВАННЫМ состоянием ~154 МБ на агента, от контекста не зависит).
Поэтому KV-кэш у этой модели в ~4 раза меньше, чем у обычной 27–32B модели.

Считает:
  1. влезут ли веса + N агентов × контекст в выбранные GPU;
  2. сколько агентов с таким контекстом максимум;
  3. оценку скорости: декод (ток/с на агента и суммарно) и «холодный» префилл;
  4. стоимость аренды в час и за 1 млн сгенерированных токенов.

Скорости — «физическая» модель (пропускная способность памяти и TFLOPS), откалиброванная
по публичным замерам Qwen3.8-27B на RTX PRO 6000, RTX 5090, H100, H200 и DGX Spark
(см. docs/03-zhelezo.md). Точность ~±25% там, где есть замеры, и хуже там, где их нет (B200).
Реальные цифры меряйте scripts/bench_swarm.py, реальный объём KV-кэша — в логе vLLM
(«GPU KV cache size», «Maximum concurrency»).

Примеры:
    python3 scripts/vram_calc.py --compare
    python3 scripts/vram_calc.py --gpu rtx-pro-6000 --gpus 2 --weights fp8 --agents 10 --context 262144
    python3 scripts/vram_calc.py --gpu h200 --weights fp8 --kv bf16 --agents 10 --context 200000
"""
import argparse
import math
from dataclasses import dataclass

GiB = 1024 ** 3

# ---------------------------------------------------------------- модель
# Qwen3.8-27B (та же схема, что у Qwen3.5-27B и Qwen3.6-27B).
N_ATTN_LAYERS = 16          # слоёв полного внимания (full_attention_interval = 4)
N_GDN_LAYERS = 48           # слоёв Gated DeltaNet
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 24, 4, 256
GDN_V_HEADS, GDN_QK_HEADS, GDN_HEAD_DIM, CONV_K = 48, 16, 128, 4
CONV_DIM = 2 * GDN_QK_HEADS * GDN_HEAD_DIM + GDN_V_HEADS * GDN_HEAD_DIM
MATMUL_PARAMS = 24.4e9      # параметров в матричных умножениях на токен (без эмбеддингов)


def kv_bytes_per_token(kv_bytes: float, mtp: bool) -> float:
    # K и V для каждого слоя полного внимания; у MTP-головы есть ещё один такой слой.
    layers = N_ATTN_LAYERS + (1 if mtp else 0)
    return layers * 2 * N_KV_HEADS * HEAD_DIM * kv_bytes


def gdn_layer_state_bytes(spec_tokens: int = 0, ssm_bytes: float = 4.0) -> float:
    # SSM-состояние в fp32 (так требует config модели: mamba_ssm_dtype=float32) + conv в bf16.
    return GDN_V_HEADS * GDN_HEAD_DIM ** 2 * ssm_bytes + (CONV_K - 1 + spec_tokens) * CONV_DIM * 2


def gdn_state_bytes(ssm_bytes: float = 4.0) -> float:
    """Состояние DeltaNet одного агента (все 48 слоёв): ~154 МБ в fp32."""
    return N_GDN_LAYERS * gdn_layer_state_bytes(0, ssm_bytes)


def gdn_reserved_bytes(spec_tokens: int) -> float:
    """Сколько vLLM (режим кэша align) резервирует под DeltaNet на один запрос: (2 + k) состояний."""
    pad = 17 / 16 if spec_tokens else 1.0  # слой MTP делает группы по 17 слотов
    return (2 + spec_tokens) * N_GDN_LAYERS * gdn_layer_state_bytes(spec_tokens) * pad


def block_size(kv_bytes: float, spec_tokens: int) -> int:
    """Шаг кэша префиксов vLLM: страница внимания должна вместить одно состояние DeltaNet."""
    per_tok_layer = 2 * N_KV_HEADS * HEAD_DIM * kv_bytes
    return 16 * math.ceil(gdn_layer_state_bytes(spec_tokens) / (16 * per_tok_layer))


# Варианты весов. loaded — размер чекпойнта в GiB (рецепт vLLM для Qwen3.8-27B; включает
# vision-энкодер и MTP-голову); read — сколько читается на каждом шаге декода (без таблицы
# эмбеддингов, vision и MTP); gemm — точность матричных умножений на префилле.
@dataclass
class Weights:
    name: str
    loaded_gib: float
    read_gib: float
    gemm: str
    note: str


WEIGHTS = {
    "bf16": Weights("BF16 (оригинал)", 51.7, 47.7, "bf16", "Qwen/Qwen3.8-27B"),
    "fp8": Weights("FP8 (официальный)", 28.7, 25.1, "fp8", "Qwen/Qwen3.8-27B-FP8"),
    "nvfp4": Weights("NVFP4 (NVIDIA: MLP 4 бит, внимание FP8)", 20.4, 16.8, "mixed",
                     "nvidia/Qwen3.8-27B-NVFP4, только Blackwell"),
    "nvfp4-w4a4": Weights("NVFP4 W4A4 (Inferact)", 24.6, 21.0, "fp4", "Inferact/Qwen3.8-27B-NVFP4, только Blackwell"),
    "int4": Weights("INT4 W4A16 (RedHat)", 18.2, 14.6, "bf16", "RedHatAI/Qwen3.8-27B-INT4, в т.ч. Hopper"),
}
VISION_WEIGHTS_GIB = 0.85  # веса vision-энкодера: не грузятся с --language-model-only
VISION_PROFILE_GIB = 0.6   # + резерв под профилирование картинок (по замерам рецепта на RTX 5090)


# ---------------------------------------------------------------- GPU
@dataclass
class GPU:
    name: str
    mem_gib: float        # память одной GPU, GiB
    bw_tbs: float         # пропускная способность памяти, ТБ/с
    tflops: dict          # пиковые dense-TFLOPS тензорных ядер; "attn" — если внимание быстрее bf16
    nvlink: bool
    bw_eff: float         # доля пропускной способности на декоде (калибровка по замерам)
    price_h: float = None  # аренда $/час за GPU (RunPod Secure, 25.09.2026), None — только покупка
    util: float = 0.92    # --gpu-memory-utilization (по умолчанию в vLLM 0.30)
    overhead_gib: float = 1.5  # активации, буферы сэмплера и т.п. на одну GPU


GPUS = {
    "rtx3090": GPU("RTX 3090 24GB", 24.0, 0.936, {"bf16": 71}, False, 0.75, 0.50),
    "rtx4090": GPU("RTX 4090 24GB", 24.0, 1.008, {"bf16": 165, "fp8": 330}, False, 0.75, 0.74),
    "rtx5090": GPU("RTX 5090 32GB", 31.4, 1.792, {"bf16": 209, "fp8": 419, "fp4": 838, "attn": 419},
                   False, 0.80, 0.99),
    "rtx-pro-6000": GPU("RTX PRO 6000 Blackwell 96GB", 95.0, 1.792, {"bf16": 500, "fp8": 1000, "fp4": 2000},
                        False, 0.80, 2.09),
    "l40s": GPU("L40S 48GB", 44.4, 0.864, {"bf16": 362, "fp8": 733}, False, 0.75, 1.09),
    "a100": GPU("A100 80GB", 79.1, 2.039, {"bf16": 312}, True, 0.75, 1.59),
    "h100": GPU("H100 80GB SXM", 79.1, 3.35, {"bf16": 989, "fp8": 1979}, True, 0.70, 3.49),
    "h100-pcie": GPU("H100 80GB PCIe", 79.1, 2.0, {"bf16": 756, "fp8": 1513}, False, 0.70, 2.89),
    "h200": GPU("H200 141GB", 139.8, 4.8, {"bf16": 989, "fp8": 1979}, True, 0.78, 4.59),
    "b200": GPU("B200 180GB", 178.4, 8.0, {"bf16": 2250, "fp8": 4500, "fp4": 9000}, True, 0.75, 6.79),
    # ROCm-ядра для DeltaNet пока слабые: в замерах декод падал до 4 ток/с на 128K.
    "mi300x": GPU("MI300X 192GB", 191.5, 5.3, {"bf16": 1307, "fp8": 2615}, True, 0.45, 2.39),
    "dgx-spark": GPU("DGX Spark (GB10) 128GB", 119.0, 0.273, {"bf16": 100, "fp8": 200, "fp4": 400},
                     False, 0.75, None, util=0.70),
}

GEMM_EFF = 0.375  # доля пиковых TFLOPS на префилле (калибровка: RTX PRO 6000, FP8, 8K–128K)
ATTN_EFF = 0.54   # доля пиковых TFLOPS в ядрах внимания на длинном контексте (там же)
STEP_OVERHEAD_S = 0.0015  # планировщик и запуск ядер на каждом шаге (с CUDA graphs)


@dataclass
class Result:
    fits_weights: bool
    kv_pool_tokens: float
    per_agent_gib: float
    max_agents: int
    need_gib: float
    have_gib: float


def capacity(gpu, n_gpus, w, kv_bytes, context, agents, spec, vision, util=None, overhead=None) -> Result:
    util = gpu.util if util is None else util
    overhead = gpu.overhead_gib if overhead is None else overhead
    replicas = max(1, n_gpus // N_KV_HEADS)  # при TP > 4 KV-головы дублируются
    weights = w.loaded_gib - (0 if vision else VISION_WEIGHTS_GIB)
    overhead += VISION_PROFILE_GIB if vision else 0
    free_per_gpu = gpu.mem_gib * util - weights / n_gpus - overhead
    pool = max(0.0, free_per_gpu) * n_gpus * GiB
    per_tok = kv_bytes_per_token(kv_bytes, spec > 0) * replicas
    per_agent = context * per_tok + gdn_reserved_bytes(spec)
    return Result(
        fits_weights=free_per_gpu > 0,
        kv_pool_tokens=pool / per_tok,
        per_agent_gib=per_agent / GiB,
        max_agents=int(pool // per_agent) if pool > 0 else 0,
        need_gib=weights + agents * per_agent / GiB + (overhead + gpu.mem_gib * (1 - util)) * n_gpus,
        have_gib=gpu.mem_gib * n_gpus,
    )


def decode_speed(gpu, n_gpus, w, kv_bytes, agents, avg_ctx, spec, mtp_gain):
    """(ток/с на агента, ток/с суммарно).

    Декод упирается в чтение памяти: на каждом шаге читаются веса, KV-кэш ВСЕХ агентов
    и состояния DeltaNet (чтение + запись; с MTP ещё k промежуточных состояний).
    """
    state_traffic = gdn_state_bytes() * (2 + spec)
    step_bytes = (w.read_gib * GiB + agents * (avg_ctx * kv_bytes_per_token(kv_bytes, spec > 0) + state_traffic))
    t = step_bytes / n_gpus / (gpu.bw_tbs * 1e12 * gpu.bw_eff) + STEP_OVERHEAD_S
    if n_gpus > 1:
        t += 0.0006 if gpu.nvlink else 0.002  # all-reduce между GPU на каждом шаге
    per_agent = (mtp_gain if spec else 1.0) / t
    return per_agent, per_agent * agents


def gemm_tflops(gpu, w):
    t = gpu.tflops
    if w.gemm == "mixed" and "fp4" in t:  # MLP (~70% FLOPs) в FP4, остальное в FP8
        return 1 / (0.7 / t["fp4"] + 0.3 / t["fp8"])
    return t.get(w.gemm) or t.get("fp8") or t["bf16"]


def prefill_seconds(gpu, n_gpus, w, tokens) -> float:
    """Время «холодного» префилла одного контекста (без кэша). Упирается в TFLOPS."""
    linear = 2 * MATMUL_PARAMS * tokens / (gemm_tflops(gpu, w) * 1e12 * GEMM_EFF)
    attn_flops = 2 * tokens ** 2 * N_Q_HEADS * HEAD_DIM * N_ATTN_LAYERS  # causal: QK^T + AV
    attn = attn_flops / (gpu.tflops.get("attn", gpu.tflops["bf16"]) * 1e12 * ATTN_EFF)
    tp_eff = 1.0 if n_gpus == 1 else 0.85
    return (linear + attn) / (n_gpus * tp_eff)


def cost_per_mtok(gpu, n_gpus, total_tps):
    if gpu.price_h is None or total_tps <= 0:
        return None
    return gpu.price_h * n_gpus / (total_tps * 3600) * 1e6


def agents_word(n: int) -> str:
    n100, n10 = abs(n) % 100, abs(n) % 10
    if 11 <= n100 <= 19 or n10 == 0 or n10 >= 5:
        return f"{n} агентов"
    return f"{n} агент" if n10 == 1 else f"{n} агента"


def fmt_tokens(x: float) -> str:
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"


def single(args):
    gpu, w = GPUS[args.gpu], WEIGHTS[args.weights]
    kvb = 2.0 if args.kv == "bf16" else 1.0
    spec = args.mtp_tokens
    r = capacity(gpu, args.gpus, w, kvb, args.context, args.agents, spec, args.vision, args.util, args.overhead)
    print(f"Конфигурация: {args.gpus}× {gpu.name}, веса {w.name} ({w.note}), KV-кэш {args.kv.upper()}, "
          f"MTP {'k=%d' % spec if spec else 'выкл'}")
    print(f"  веса в памяти:            {w.loaded_gib - (0 if args.vision else VISION_WEIGHTS_GIB):.1f} GiB"
          f"{'' if args.vision else '  (текстовый режим, без vision-энкодера)'}")
    print(f"  KV на токен:              {kv_bytes_per_token(kvb, spec > 0) / 1024:.0f} KiB"
          f"  (у обычной 32B-модели без гибрида было бы в ~4 раза больше)")
    print(f"  DeltaNet на агента:       {gdn_state_bytes() / 1e6:.0f} МБ состояние, "
          f"{gdn_reserved_bytes(spec) / GiB:.2f} GiB резерв vLLM (от контекста не зависит)")
    print(f"  шаг кэша префиксов:       {block_size(kvb, spec)} токенов (размер блока vLLM)")
    print(f"  один агент × {args.context:,} ток.: {r.per_agent_gib:.1f} GiB")
    if not r.fits_weights:
        print("  ✗ Веса не помещаются. Нужна более сильная квантизация или больше GPU.")
        return
    print(f"  ёмкость KV-кэша:          ≈{fmt_tokens(r.kv_pool_tokens)} токенов "
          f"→ максимум агентов с таким контекстом: {r.max_agents}")
    verdict = "✓ помещается" if r.max_agents >= args.agents else "✗ НЕ помещается"
    print(f"  {agents_word(args.agents)} × {args.context:,}: {verdict} "
          f"(нужно ≈{r.need_gib:.0f} GiB из {r.have_gib:.0f} GiB)")
    per_tok = kv_bytes_per_token(kvb, spec > 0) * max(1, args.gpus // N_KV_HEADS)
    ctx_each = (r.kv_pool_tokens * per_tok - args.agents * gdn_reserved_bytes(spec)) / (args.agents * per_tok)
    print(f"  контекст на агента при N={args.agents}: до ≈{fmt_tokens(max(0, ctx_each))} токенов")

    print("\nОценка скорости (±25%, проверяйте bench_swarm.py):")
    n = min(args.agents, max(1, r.max_agents))
    total_half = 0
    for label, ctx in (("контексты заполнены наполовину", args.context / 2), ("контексты заполнены целиком", args.context)):
        pa, total = decode_speed(gpu, args.gpus, w, kvb, n, ctx, spec, args.mtp_gain)
        total_half = total_half or total
        print(f"  декод, {agents_word(n)}, {label}: ~{pa:.0f} ток/с на агента, ~{total:.0f} ток/с суммарно")
    pa1, _ = decode_speed(gpu, args.gpus, w, kvb, 1, 8000, spec, args.mtp_gain)
    print(f"  декод, 1 агент, короткий контекст: ~{pa1:.0f} ток/с")
    for toks in (32_768, 131_072, args.context):
        s = prefill_seconds(gpu, args.gpus, w, toks)
        print(f"  холодный префилл {toks:>7,} ток.: ~{s:.0f} с (~{toks / s:,.0f} ток/с)")
    print("  (повторные ходы с кэшем префиксов пересчитывают только новые токены + 1–2 блока)")
    c = cost_per_mtok(gpu, args.gpus, total_half)
    if c is not None:
        print(f"\nАренда: ~${gpu.price_h * args.gpus:.2f}/час → ~${c:.2f} за 1 млн сгенерированных токенов "
              "при полной загрузке (весь прочитанный контекст — бесплатно)")


COMPARE = [
    ("rtx5090", 1, "nvfp4"), ("rtx5090", 2, "nvfp4"), ("rtx-pro-6000", 1, "nvfp4"),
    ("rtx-pro-6000", 1, "fp8"), ("rtx-pro-6000", 2, "fp8"), ("h100", 1, "fp8"), ("h100", 2, "fp8"),
    ("h200", 1, "fp8"), ("b200", 1, "fp8"), ("b200", 1, "nvfp4-w4a4"), ("dgx-spark", 1, "nvfp4"),
]


def compare(args):
    kvb = 2.0 if args.kv == "bf16" else 1.0
    spec = args.mtp_tokens
    print(f"Сценарий: {agents_word(args.agents)} × контекст {args.context:,} ток., KV-кэш {args.kv.upper()}, "
          f"MTP {'k=%d (×%.1f)' % (spec, args.mtp_gain) if spec else 'выкл'}\n")
    hdr = (f"{'конфигурация':<31}|{'веса':>6} |{'KV-ёмк.':>8} |{'агентов':>8} |{'ток/с/аг':>9} |"
           f"{'ток/с всего':>12} |{'префилл':>8} |{'$/час':>6} |{'$/1М ток':>9}")
    print(hdr)
    print("-" * len(hdr))
    for key, n, wkey in COMPARE:
        gpu, w = GPUS[key], WEIGHTS[wkey]
        r = capacity(gpu, n, w, kvb, args.context, args.agents, spec, args.vision)
        name = f"{n}× {gpu.name}"[:30]
        wlabel = wkey.replace("-w4a4", "*")
        if not r.fits_weights:
            print(f"{name:<31}|{wlabel:>6} | веса не помещаются")
            continue
        agents = min(args.agents, r.max_agents)
        price = f"{gpu.price_h * n:.2f}" if gpu.price_h else "покуп."
        if agents == 0:
            print(f"{name:<31}|{wlabel:>6} |{fmt_tokens(r.kv_pool_tokens):>8} |{0:>8} |{'—':>9} |{'—':>12} |"
                  f"{'—':>8} |{price:>6} |{'—':>9}")
            continue
        pa, total = decode_speed(gpu, n, w, kvb, agents, args.context * 0.6, spec, args.mtp_gain)
        pf = prefill_seconds(gpu, n, w, args.context)
        c = cost_per_mtok(gpu, n, total)
        mark = "" if r.max_agents >= args.agents else "✗"
        print(f"{name:<31}|{wlabel:>6} |{fmt_tokens(r.kv_pool_tokens):>8} |{r.max_agents:>6}{mark:<2} |{pa:>9.0f} |"
              f"{total:>12.0f} |{pf:>7.0f}с |{price:>6} |{('$%.2f' % c) if c else '—':>9}")
    print("\nток/с — при заполнении контекстов на ~60% и числе агентов min(заданное, влезающее); ✗ — все "
          "не помещаются.\nпрефилл — холодный расчёт всего контекста одного агента (потом работает кэш префиксов)."
          "\n$/час — RunPod Secure, 25.09.2026; $/1М ток — цена 1 млн сгенерированных токенов при полной "
          "загрузке.\nnvfp4* — Inferact W4A4. Точность оценок ±25% (для B200 замеров нет).")


def main():
    p = argparse.ArgumentParser(description="Калькулятор памяти/скорости/стоимости Qwen3.x-27B для роя агентов")
    p.add_argument("--compare", action="store_true", help="таблица по типовым конфигурациям")
    p.add_argument("--gpu", choices=sorted(GPUS), default="rtx-pro-6000")
    p.add_argument("--gpus", type=int, default=1, help="число GPU (tensor parallel)")
    p.add_argument("--weights", choices=sorted(WEIGHTS), default="fp8")
    p.add_argument("--kv", choices=["fp8", "bf16"], default="fp8", help="тип KV-кэша (--kv-cache-dtype)")
    p.add_argument("--context", type=int, default=262_144, help="контекст одного агента, токенов")
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--mtp-tokens", type=int, default=3, help="num_speculative_tokens для MTP (0 — выключить)")
    p.add_argument("--mtp-gain", type=float, default=1.6, help="ускорение декода от MTP (в замерах 1.5–2.2)")
    p.add_argument("--vision", action="store_true", help="грузить vision-энкодер (без --language-model-only)")
    p.add_argument("--util", type=float, default=None, help="--gpu-memory-utilization (по умолчанию 0.92)")
    p.add_argument("--overhead", type=float, default=None, help="накладные расходы на GPU, GiB (по умолч. 1.5)")
    args = p.parse_args()
    (compare if args.compare else single)(args)


if __name__ == "__main__":
    main()
