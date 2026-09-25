#!/usr/bin/env python3
"""
Калькулятор памяти и скорости для Qwen3.x-27B под «рой агентов».

Архитектура Qwen3.5/3.6/3.8-27B гибридная: из 64 слоёв только 16 — «полное внимание»
(им нужен KV-кэш, растущий с длиной контекста), остальные 48 — Gated DeltaNet
(линейное внимание с ФИКСИРОВАННЫМ состоянием ~75 МБ на агента, от контекста не зависит).
Поэтому KV-кэш у этой модели в ~4 раза меньше, чем у обычной 27–32B модели.

Считает:
  1. влезут ли веса + N агентов × контекст в выбранные GPU;
  2. сколько агентов с таким контекстом максимум;
  3. ГРУБУЮ оценку скорости: декод (ток/с на агента и суммарно) и «холодный» префилл.

Оценки скорости — по «физике» (пропускная способность памяти и TFLOPS) с поправочными
коэффициентами; точность ±30–50%. Реальные цифры меряйте scripts/bench_swarm.py,
а реальный объём KV-кэша смотрите в логе vLLM при старте («GPU KV cache size: … tokens»).

Примеры:
    python3 scripts/vram_calc.py --compare                    # сравнить типовые конфигурации
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
MATMUL_PARAMS = 24.4e9      # параметров в матричных умножениях на токен (без эмбеддингов)


def kv_bytes_per_token(kv_bytes: float, mtp: bool) -> float:
    # K и V для каждого слоя полного внимания; у MTP-головы есть ещё один такой слой.
    layers = N_ATTN_LAYERS + (1 if mtp else 0)
    return layers * 2 * N_KV_HEADS * HEAD_DIM * kv_bytes


def gdn_state_bytes(state_bytes: float = 2.0) -> float:
    # Рекуррентное состояние (V-головы × d_k × d_v) + состояние свёртки (kernel-1) × conv_dim.
    ssm = GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * state_bytes
    conv = (CONV_K - 1) * (2 * GDN_QK_HEADS * GDN_HEAD_DIM + GDN_V_HEADS * GDN_HEAD_DIM) * 2
    return N_GDN_LAYERS * (ssm + conv)


# Варианты весов. loaded — размер чекпойнта (из рецепта vLLM для Qwen3.8-27B; включает
# vision-энкодер и MTP-голову), read — сколько читается на каждом шаге декода
# (без таблицы эмбеддингов, vision и MTP),
# gemm — какой точностью считаются матричные умножения при префилле.
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
    "nvfp4": Weights("NVFP4 (NVIDIA, MLP в 4 бит)", 20.4, 16.8, "fp8", "nvidia/Qwen3.8-27B-NVFP4, только Blackwell"),
    "nvfp4-w4a4": Weights("NVFP4 W4A4 (Inferact)", 24.6, 21.0, "fp4", "Inferact/Qwen3.8-27B-NVFP4, только Blackwell"),
    "int4": Weights("INT4 W4A16 (RedHat)", 18.2, 14.6, "bf16", "RedHatAI/Qwen3.8-27B-INT4, в т.ч. Hopper"),
}
VISION_WEIGHTS_GIB = 0.85  # веса vision-энкодера: не грузятся с --language-model-only
VISION_PROFILE_GIB = 0.6   # + резерв под профилирование картинок (по замерам рецепта на RTX 5090)


# ---------------------------------------------------------------- GPU
@dataclass
class GPU:
    name: str
    mem_gib: float        # доступно под vLLM (с учётом --gpu-memory-utilization по умолчанию ниже)
    bw_tbs: float         # пропускная способность памяти, ТБ/с
    tflops: dict          # пиковые dense-TFLOPS тензорных ядер по точностям
    nvlink: bool
    bw_eff: float = 0.70  # какая доля пропускной способности реально достигается на декоде
    util: float = 0.90    # --gpu-memory-utilization по умолчанию
    overhead_gib: float = 1.5  # активации, буферы сэмплера и т.п. на одну GPU


GPUS = {
    "rtx3090": GPU("RTX 3090 24GB", 23.6, 0.936, {"bf16": 71}, False, 0.65),
    "rtx4090": GPU("RTX 4090 24GB", 23.6, 1.008, {"bf16": 165, "fp8": 330}, False, 0.65),
    "rtx5090": GPU("RTX 5090 32GB", 31.4, 1.792, {"bf16": 209, "fp8": 419, "fp4": 838}, False, 0.65),
    "rtx-pro-6000": GPU("RTX PRO 6000 Blackwell 96GB", 94.5, 1.792, {"bf16": 500, "fp8": 1000, "fp4": 2000},
                        False, 0.65),
    "l40s": GPU("L40S 48GB", 44.4, 0.864, {"bf16": 362, "fp8": 733}, False, 0.65),
    "a100": GPU("A100 80GB SXM", 79.1, 2.039, {"bf16": 312}, True),
    "h100": GPU("H100 80GB SXM", 79.1, 3.35, {"bf16": 989, "fp8": 1979}, True),
    "h100-pcie": GPU("H100 80GB PCIe", 79.1, 2.0, {"bf16": 756, "fp8": 1513}, False),
    "h200": GPU("H200 141GB", 139.8, 4.8, {"bf16": 989, "fp8": 1979}, True),
    "b200": GPU("B200 180GB", 178.4, 8.0, {"bf16": 2250, "fp8": 4500, "fp4": 9000}, True),
    "mi300x": GPU("MI300X 192GB", 191.5, 5.3, {"bf16": 1307, "fp8": 2615}, True, 0.60),
    "dgx-spark": GPU("DGX Spark (GB10) 128GB общей памяти", 119.0, 0.273, {"bf16": 100, "fp8": 200, "fp4": 400},
                     False, 0.75, util=0.70),
    "mac-m3-ultra": GPU("Mac Studio M3 Ultra 512GB (MLX)", 476.0, 0.819, {"bf16": 28}, False, 0.70, util=0.80),
}

GEMM_EFF = 0.55   # доля пиковых TFLOPS на больших матричных умножениях (префилл)
ATTN_EFF = 0.35   # доля пиковых BF16 TFLOPS в ядрах внимания на длинном контексте
STEP_OVERHEAD_S = 0.0015  # планировщик, запуск ядер (с CUDA graphs)


@dataclass
class Result:
    fits_weights: bool
    kv_pool_tokens: float
    per_agent_gib: float
    max_agents: int
    need_gib: float
    have_gib: float


def capacity(gpu: GPU, n_gpus: int, w: Weights, kv_bytes: float, context: int, agents: int,
             mtp: bool, vision: bool, util=None, overhead=None) -> Result:
    util = gpu.util if util is None else util
    overhead = gpu.overhead_gib if overhead is None else overhead
    replicas = max(1, n_gpus // N_KV_HEADS)  # при TP > 4 KV-головы дублируются
    weights = w.loaded_gib - (0 if vision else VISION_WEIGHTS_GIB)
    overhead += VISION_PROFILE_GIB if vision else 0
    free_per_gpu = gpu.mem_gib * util - weights / n_gpus - overhead
    pool = max(0.0, free_per_gpu) * n_gpus * GiB
    per_tok = kv_bytes_per_token(kv_bytes, mtp) * replicas
    per_agent = context * per_tok + gdn_state_bytes()
    return Result(
        fits_weights=free_per_gpu > 0,
        kv_pool_tokens=pool / per_tok,
        per_agent_gib=per_agent / GiB,
        max_agents=int(pool // per_agent) if pool > 0 else 0,
        need_gib=(weights + agents * per_agent / GiB) + (overhead + gpu.mem_gib * (1 - util)) * n_gpus,
        have_gib=gpu.mem_gib * n_gpus,
    )


def decode_speed(gpu: GPU, n_gpus: int, w: Weights, kv_bytes: float, agents: int, avg_ctx: float,
                 mtp: bool, mtp_gain: float):
    """(ток/с на агента, ток/с суммарно). Декод упирается в чтение памяти: веса + KV всех агентов."""
    step_bytes = (w.read_gib * GiB
                  + agents * avg_ctx * kv_bytes_per_token(kv_bytes, mtp)
                  + agents * 2 * gdn_state_bytes()) / n_gpus
    t = step_bytes / (gpu.bw_tbs * 1e12 * gpu.bw_eff) + STEP_OVERHEAD_S
    if n_gpus > 1:
        t += 0.0006 if gpu.nvlink else 0.002  # all-reduce между GPU на каждом шаге
    per_agent = (mtp_gain if mtp else 1.0) / t
    return per_agent, per_agent * agents


def prefill_seconds(gpu: GPU, n_gpus: int, w: Weights, tokens: int) -> float:
    """Время «холодного» префилла одного контекста (без кэша). Упирается в TFLOPS."""
    peak_gemm = gpu.tflops.get(w.gemm) or gpu.tflops.get("fp8") or gpu.tflops["bf16"]
    linear = 2 * MATMUL_PARAMS * tokens / (peak_gemm * 1e12 * GEMM_EFF)
    attn_flops = 2 * tokens ** 2 * N_Q_HEADS * HEAD_DIM * N_ATTN_LAYERS  # causal: QK^T + AV
    attn = attn_flops / (gpu.tflops["bf16"] * 1e12 * ATTN_EFF)
    tp_eff = 1.0 if n_gpus == 1 else 0.85
    return (linear + attn) / (n_gpus * tp_eff)


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
    r = capacity(gpu, args.gpus, w, kvb, args.context, args.agents, args.mtp, args.vision, args.util, args.overhead)
    print(f"Конфигурация: {args.gpus}× {gpu.name}, веса {w.name} ({w.note}), KV-кэш {args.kv.upper()}")
    print(f"  веса в памяти:            {w.loaded_gib - (0 if args.vision else VISION_WEIGHTS_GIB):.1f} GiB"
          f"{'' if args.vision else '  (текстовый режим, без vision-энкодера)'}")
    print(f"  KV на токен:              {kv_bytes_per_token(kvb, args.mtp) / 1024:.0f} KiB"
          f"  (у обычной 32B-модели без гибрида было бы в ~4 раза больше)")
    print(f"  состояние DeltaNet/агент: {gdn_state_bytes() / 2**20:.0f} MiB (не зависит от контекста)")
    print(f"  один агент × {args.context:,} ток.: {r.per_agent_gib:.1f} GiB")
    if not r.fits_weights:
        print("  ✗ Веса не помещаются. Нужна более сильная квантизация или больше GPU.")
        return
    print(f"  ёмкость KV-кэша:          ≈{fmt_tokens(r.kv_pool_tokens)} токенов "
          f"→ максимум агентов с таким контекстом: {r.max_agents}")
    verdict = "✓ помещается" if r.max_agents >= args.agents else "✗ НЕ помещается"
    print(f"  {agents_word(args.agents)} × {args.context:,}: {verdict} "
          f"(нужно ≈{r.need_gib:.0f} GiB из {r.have_gib:.0f} GiB)")

    print("\nГрубая оценка скорости (±30–50%, проверяйте bench_swarm.py):")
    for label, ctx in (("контекст заполнен наполовину", args.context / 2), ("контекст заполнен целиком", args.context)):
        n = min(args.agents, max(1, r.max_agents))
        pa, total = decode_speed(gpu, args.gpus, w, kvb, n, ctx, args.mtp, args.mtp_gain)
        print(f"  декод, {agents_word(n)}, {label}: ~{pa:.0f} ток/с на агента, ~{total:.0f} ток/с суммарно")
    pa1, _ = decode_speed(gpu, args.gpus, w, kvb, 1, 8000, args.mtp, args.mtp_gain)
    print(f"  декод, 1 агент, короткий контекст: ~{pa1:.0f} ток/с")
    for toks in (32_768, 131_072, args.context):
        print(f"  холодный префилл {toks:>7,} ток.: ~{prefill_seconds(gpu, args.gpus, w, toks):.0f} с "
              "(повторные ходы с кэшем префиксов — только новые токены)")
    if args.mtp:
        print(f"  (MTP включён: множитель декода ×{args.mtp_gain} — уточните по метрикам приёмки)")


COMPARE = [
    ("rtx5090", 1, "nvfp4-w4a4"), ("rtx5090", 2, "nvfp4"), ("rtx-pro-6000", 1, "nvfp4"),
    ("rtx-pro-6000", 1, "fp8"), ("rtx-pro-6000", 2, "fp8"), ("h100", 1, "fp8"), ("h100", 2, "fp8"),
    ("h200", 1, "fp8"), ("b200", 1, "fp8"), ("b200", 1, "nvfp4-w4a4"), ("mi300x", 1, "fp8"),
    ("dgx-spark", 1, "nvfp4"), ("mac-m3-ultra", 1, "int4"),
]


def compare(args):
    kvb = 2.0 if args.kv == "bf16" else 1.0
    print(f"Сценарий: {agents_word(args.agents)} × контекст {args.context:,} ток., KV-кэш {args.kv.upper()}, "
          f"MTP {'вкл (×%.1f)' % args.mtp_gain if args.mtp else 'выкл'}\n")
    hdr = (f"{'конфигурация':<34}|{'веса':>11} |{'KV-ёмкость':>11} |{'макс.агентов':>13} |"
           f"{'ток/с/агент':>12} |{'ток/с всего':>12} |{'префилл':>9}")
    print(hdr)
    print("-" * len(hdr))
    for key, n, wkey in COMPARE:
        gpu, w = GPUS[key], WEIGHTS[wkey]
        r = capacity(gpu, n, w, kvb, args.context, args.agents, args.mtp, args.vision)
        name = f"{n}× {gpu.name}"[:33]
        if not r.fits_weights:
            print(f"{name:<34}|{wkey:>11} | веса не помещаются")
            continue
        agents = min(args.agents, r.max_agents)
        if agents == 0:
            print(f"{name:<34}|{wkey:>11} |{fmt_tokens(r.kv_pool_tokens):>11} |{0:>13} |{'—':>12} |{'—':>12} |")
            continue
        pa, total = decode_speed(gpu, n, w, kvb, agents, args.context * 0.6, args.mtp, args.mtp_gain)
        pf = prefill_seconds(gpu, n, w, args.context)
        mark = "" if r.max_agents >= args.agents else " ✗"
        print(f"{name:<34}|{wkey:>11} |{fmt_tokens(r.kv_pool_tokens):>11} |{r.max_agents:>11}{mark:<2} |"
              f"{pa:>12.0f} |{total:>12.0f} |{pf:>8.0f}с")
    print("\nток/с — при заполнении контекстов на ~60% (агенты «растут» по ходу работы) и числе агентов "
          "min(заданное, влезающее).\nпрефилл — холодный расчёт всего контекста одного агента; "
          "с кэшем префиксов на следующих ходах считаются только новые токены.")


def main():
    p = argparse.ArgumentParser(description="Калькулятор памяти/скорости Qwen3.x-27B для роя агентов")
    p.add_argument("--compare", action="store_true", help="таблица по типовым конфигурациям")
    p.add_argument("--gpu", choices=sorted(GPUS), default="rtx-pro-6000")
    p.add_argument("--gpus", type=int, default=1, help="число GPU (tensor parallel)")
    p.add_argument("--weights", choices=sorted(WEIGHTS), default="fp8")
    p.add_argument("--kv", choices=["fp8", "bf16"], default="fp8", help="тип KV-кэша (--kv-cache-dtype)")
    p.add_argument("--context", type=int, default=262_144, help="контекст одного агента, токенов")
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--no-mtp", dest="mtp", action="store_false", help="без спекулятивного декодирования MTP")
    p.add_argument("--mtp-gain", type=float, default=1.8, help="ускорение декода от MTP (реально 1.5–2.5)")
    p.add_argument("--vision", action="store_true", help="грузить vision-энкодер (без --language-model-only)")
    p.add_argument("--util", type=float, default=None, help="--gpu-memory-utilization (по умолчанию 0.9)")
    p.add_argument("--overhead", type=float, default=None, help="накладные расходы на GPU, GiB (по умолч. 1.5)")
    args = p.parse_args()
    (compare if args.compare else single)(args)


if __name__ == "__main__":
    main()
