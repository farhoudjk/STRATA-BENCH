#!/usr/bin/env python3
"""Supplementary: default policies vs. aligned constraints (L40S fairness experiments).

    python scripts/plot_supp_fairness.py --metrics <InferenceBench>/results/l40s_fairness/metrics.csv \
        --out docs/figures/paper_supp

Uses only the November reruns (source = rerun_nov2026): every variant is compared with a default run from the
same environment and harness (which applies each model's chat template; the July RQ1 runs did not).
(a) TTFT p95 from arrival: each engine's default, and TensorRT-LLM with MAX_UTILIZATION (E1).
(b) Strict goodput: each engine with its default KV pool (solid) and with an equal pool (hatched, E2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

GROUPS = [("llama31_8b", 2), ("llama31_8b", 8), ("mistral_nemo_12b", 2), ("mistral_nemo_12b", 8)]
GLAB = ["Llama\n2 req/s", "Llama\n8 req/s", "Mistral\n2 req/s", "Mistral\n8 req/s"]


def val(m, model, engine, variant, rps, col):
    r = m[(m.model == model) & (m.engine == engine) & (m.variant == variant) & np.isclose(m.rate_rps, rps)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(a.metrics)
    m = m[m.source == "rerun_nov2026"]
    F.style(1.15)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.22, width_ratios=[1, 1.25]))
    x = np.arange(len(GROUPS))

    # (a) E1: TTFT p95
    bars = [("vllm", "default", "", "vLLM"), ("sglang", "default", "", "SGLang"),
            ("tensorrt-llm", "default", "", "TensorRT-LLM (default)"),
            ("tensorrt-llm", "maxutil", "////", "TensorRT-LLM (MAX_UTILIZATION)")]
    w = 0.2
    for k, (e, v, hatch, lab) in enumerate(bars):
        y = [val(m, mo, e, v, r, "ttft_user_p95_s") for mo, r in GROUPS]
        c = F.ENG[e]["color"]
        ax.bar(x + (k - 1.5) * w, y, width=w * 0.92, color="white" if hatch else c, edgecolor=c, hatch=hatch,
               linewidth=0.8, label=lab)
    ax.set_yscale("log"); ax.set_ylim(0.05, 1000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axhline(2, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_xticks(x, GLAB, fontsize=F.fz(6)); ax.grid(axis="x", visible=False)
    ax.set_ylabel("TTFT p95 (s)"); ax.set_title("(a) Admission policy (E1)", loc="left", fontsize=F.fz(6.8))

    # (b) E2: goodput, default vs equal KV pool
    w = 0.13
    k = 0
    for e, st in F.ENG.items():
        for v, hatch in (("default", ""), ("eqkv", "////")):
            y = [val(m, mo, e, v, r, "strict_goodput_pct") for mo, r in GROUPS]
            c = st["color"]
            bx.bar(x + (k - 2.5) * w, y, width=w * 0.92, color="white" if hatch else c, edgecolor=c, hatch=hatch,
                   linewidth=0.8)
            k += 1
    bx.set_ylim(0, 105); bx.set_xticks(x, GLAB, fontsize=F.fz(6)); bx.grid(axis="x", visible=False)
    bx.set_ylabel("Strict goodput (%)"); bx.set_title("(b) Equal KV pools (E2)", loc="left", fontsize=F.fz(6.8))
    handles = [plt.Rectangle((0, 0), 1, 1, color=v["color"], label=v["label"]) for v in F.ENG.values()] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=F.INK2, label="default"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=F.INK2, hatch="////",
                      label="aligned: MAX_UTILIZATION in (a), equal KV pool in (b)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=F.fz(6),
               handlelength=1.4, columnspacing=1.0)
    fig.savefig(a.out / "fig_supp_fairness.pdf")
    plt.close(fig)
    cols = ["model", "engine", "variant", "rate_rps", "ttft_user_p50_s", "ttft_user_p95_s", "tpot_p50_ms",
            "tpot_p95_ms", "throughput_tok_s", "strict_goodput_pct"]
    m[cols].sort_values(["model", "rate_rps", "engine", "variant"]).to_csv(a.out / "supp_fairness.csv", index=False)
    print(m[cols].sort_values(["model", "rate_rps", "engine", "variant"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
