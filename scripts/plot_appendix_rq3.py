#!/usr/bin/env python3
"""Appendix figure for RQ3: where n-gram speculation pays, and why per-chunk latency misleads.

    python scripts/plot_appendix_rq3.py --rq3 results/Aggregated_results/rq3_ngram_speculative_decoding.csv \
        --out docs/figures/paper_appendix

(a) Per-request TPOT change with n-gram speculation, for the same request (same id, engine,
    max_num_seqs, and arrival process) with and without speculation, binned by how repetitive the
    request's baseline response is: the share of its word-level 3-grams that already appeared
    earlier in the response. N-gram drafting can only propose text already in the context, so
    more repetition should mean more accepted drafts.
(b) Change in the per-request median gap between streamed chunks (ITL p50) vs. change in TPOT.
(c) Median tokens per streamed chunk with and without speculation.
mixed_real, finished requests with > 20 output tokens.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENG = {
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^"),
}
BASE, SPEC = "baseline_full_fp16_tp1", "spec_ngram3_fp16_tp1"
BINS, LABELS = [-0.01, 0.05, 0.15, 0.3, 0.5, 1.0], ["<.05", ".05-.15", ".15-.3", ".3-.5", ">.5"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def repetition(text) -> float:
    w = re.findall(r"\w+|[^\w\s]", str(text))
    if len(w) < 4:
        return np.nan
    seen, hit = set(), 0
    for i in range(len(w) - 2):
        g = tuple(w[i:i + 3])
        hit += g in seen
        seen.add(g)
    return hit / (len(w) - 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq3", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_appendix"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    d = pd.read_csv(a.rq3, usecols=["request_id", "engine", "workload_name", "task_type", "strategy_name",
                                    "max_num_seqs", "rate_label", "status", "output_tokens", "e2e_s", "ttft_s",
                                    "itl_mean_ms", "itl_p50_ms", "response_text"], low_memory=False)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[(d.status == "ok") & (d.output_tokens > 20) & (d.workload_name == "mixed_real")].copy()
    d["tpot"] = (d.e2e_s - d.ttft_s) / (d.output_tokens - 1) * 1000
    d["tok_per_chunk"] = (d.output_tokens - 1) / ((d.e2e_s - d.ttft_s) * 1000 / d.itl_mean_ms)

    key = ["request_id", "engine", "max_num_seqs", "rate_label"]
    b, s = d[d.strategy_name == BASE], d[d.strategy_name == SPEC]
    m = b.merge(s[key + ["tpot"]], on=key, suffixes=("_b", "_s"))
    m["rep"] = m.response_text.map(repetition)
    m["change"] = 100 * (m.tpot_s / m.tpot_b - 1)
    m["bin"] = pd.cut(m.rep, BINS, labels=LABELS)
    pa = m.groupby(["engine", "bin"], observed=True).change.agg(["median", "count"]).reset_index()
    pa.to_csv(a.out / "appendix_a5_repetition.csv", index=False)
    m.groupby("task_type").rep.median().rename("median_repetition").reset_index().to_csv(
        a.out / "appendix_a5_repetition_by_task.csv", index=False)

    q = d[d.max_num_seqs == 64]
    summ = q.groupby(["engine", "strategy_name"]).agg(itl_p50=("itl_p50_ms", "median"), tpot_p50=("tpot", "median"),
                                                      tok_per_chunk=("tok_per_chunk", "median")).reset_index()
    summ.to_csv(a.out / "appendix_a5_trap.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.75], wspace=0.42))
    ax = axes[0]
    for e, st in ENG.items():
        t = pa[pa.engine == e].set_index("bin").reindex(LABELS)
        ax.plot(range(len(LABELS)), t["median"].values, color=st["color"], marker=st["marker"], markersize=4,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Response repetition (share of repeated 3-grams)")
    ax.set_ylabel("TPOT change, same request (%)")
    ax.set_title("(a) Repetitive output pays", loc="left")

    ax = axes[1]
    x = np.arange(len(ENG))
    for k, (col, lab, alpha) in enumerate((("itl_p50", "gap between chunks", 0.35), ("tpot_p50", "time per token", 1.0))):
        vals = []
        for e in ENG:
            r = summ[summ.engine == e].set_index("strategy_name")[col]
            vals.append(100 * (r[SPEC] / r[BASE] - 1))
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=[ENG[e]["color"] for e in ENG], alpha=alpha,
               edgecolor=[ENG[e]["color"] for e in ENG], linewidth=0.8, label=lab)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + (2 if v >= 0 else -2), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=5.8)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, [v["label"] for v in ENG.values()], fontsize=6)
    ax.set_ylabel("Change with n-gram (%)"); ax.set_ylim(-30, 95); ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.35, label="gap between chunks"),
                       plt.Rectangle((0, 0), 1, 1, facecolor=INK2, label="time per token")],
              loc="upper right", fontsize=5.8, handlelength=1.2)
    ax.set_title("(b) Opposite verdicts", loc="left")

    ax = axes[2]
    for e, st in ENG.items():
        r = summ[summ.engine == e].set_index("strategy_name").tok_per_chunk
        ax.plot([0, 1], [r[BASE], r[SPEC]], color=st["color"], marker=st["marker"], markersize=4,
                markeredgecolor="white")
    ax.set_xticks([0, 1], ["off", "n-gram"]); ax.set_xlim(-0.3, 1.3); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Tokens per streamed chunk")
    ax.set_title("(c) Why", loc="left")
    fig.legend(handles=[plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4, label=v["label"])
                        for v in ENG.values()], loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(a.out / "fig_appendix_a5_speculation.pdf")
    print(pa.round(1).to_string()); print(summ.round(2).to_string())
    print(m.groupby("task_type").rep.median().round(2))


if __name__ == "__main__":
    main()
