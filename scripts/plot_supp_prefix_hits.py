#!/usr/bin/env python3
"""Supplementary: reported prefix-cache hit rate by conversation turn and by workload.

    python scripts/plot_supp_prefix_hits.py --rq1-dir results/Aggregated_results \
        --rq2 results/Aggregated_results/rq2_prefill_handling.csv --out docs/figures/paper_supp

(a-c) chat_multiturn, C = S = 256, L40S: mean reported hit rate of the requests in each turn, per
      model, for vLLM and SGLang; TensorRT-LLM reports one value per run (scraped after the run),
      drawn as a single marker at the right.
(d)   Mistral-NeMo-12B, prefix caching on: mean hit rate per workload (mixed_real and chat_multiturn
      from RQ1, chat_real_long from RQ2).
The engines report the metric differently: vLLM a cumulative token hit rate since server start,
SGLang the hit rate of the most recent batch, TensorRT-LLM one post-run value. Values are therefore
comparable within an engine, not across engines; vLLM's cumulative value also rises with time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402  (style, colours, models)

import matplotlib.pyplot as plt  # noqa: E402

COLS = ["engine", "workload_name", "strategy_name", "max_concurrent", "max_num_seqs", "rate_label", "turn_index",
        "prefix_cache_hit_rate"]
FS = 1.15


def read(p: Path, model: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    if model:
        d["model"] = model
    d["hit"] = 100 * d.prefix_cache_hit_rate
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--rq2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    F.style(FS)
    r1 = pd.concat([read(a.rq1_dir / f"rq1_{m}.csv", m) for m in F.MODEL], ignore_index=True)
    r1 = r1[(r1.max_concurrent == 256) & (r1.max_num_seqs == 256)]
    r2 = read(a.rq2, "mistral_nemo_12b")
    r2 = r2[r2.strategy_name == "baseline_full_fp16_tp1"]  # prefix caching on, chunking off

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15], wspace=0.12))
    recs = []
    mt = r1[r1.workload_name == "chat_multiturn"]
    for ax, (m, mlab) in zip(axes[:3], F.MODEL.items()):
        g = mt[mt.model == m]
        for e in ("vllm", "sglang"):
            t = g[g.engine == e].groupby("turn_index").hit.agg(["mean", "size"])
            t = t[t["size"] >= 30]
            ax.plot(t.index + 1, t["mean"], color=F.ENG[e]["color"], marker=F.ENG[e]["marker"], markersize=3.4,
                    linewidth=1.2, markeredgecolor="white", markeredgewidth=0.4)
            recs += [dict(panel="turn", model=m, engine=e, turn=int(i) + 1, hit_pct=v, n=n)
                     for i, (v, n) in t.iterrows()]
        trt = g[g.engine == "tensorrt-llm"].hit.mean()
        ax.scatter([7.1], [trt], s=26, color=F.ENG["tensorrt-llm"]["color"], marker="^", edgecolor="white",
                   linewidth=0.4, zorder=3, clip_on=False)
        ax.axvline(6.55, color=F.GRID, linewidth=0.8)
        recs.append(dict(panel="turn", model=m, engine="tensorrt-llm", turn="post-run", hit_pct=trt))
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7.1], ["1", "2", "3", "4", "5", "6", "run"])
        ax.set_xlim(0.6, 7.6); ax.set_ylim(0, 70)
        ax.set_title(mlab, loc="center", fontsize=F.fz(6.2), pad=3)
        ax.grid(axis="x", visible=False)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Reported prefix-cache\nhit rate (%)")
    axes[1].set_xlabel("Conversation turn (chat_multiturn)")

    ax = axes[3]
    wl = [("mixed_real", r1[(r1.model == "mistral_nemo_12b") & (r1.workload_name == "mixed_real")]),
          ("multi-turn", r1[(r1.model == "mistral_nemo_12b") & (r1.workload_name == "chat_multiturn")]),
          ("long prompts", r2[r2.workload_name == "chat_real_long"])]
    x = np.arange(len(wl))
    for k, e in enumerate(F.ENG):
        vals = [g[g.engine == e].hit.mean() for _, g in wl]
        ax.bar(x + (k - 1) * 0.26, vals, width=0.24, color=F.ENG[e]["color"], linewidth=0)
        for xi, v in zip(x + (k - 1) * 0.26, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=F.fz(5.2))
        recs += [dict(panel="workload", model="mistral_nemo_12b", engine=e, workload=w, hit_pct=v)
                 for (w, _), v in zip(wl, vals)]
    ax.set_xticks(x, ["mixed_\nreal", "multi-\nturn", "long\nprompts"], fontsize=F.fz(5.8))
    ax.set_ylim(0, 70); ax.tick_params(labelleft=False); ax.grid(axis="x", visible=False)
    ax.set_title("Mistral-NeMo-12B, by workload", loc="center", fontsize=F.fz(6.2), pad=3)

    handles = [plt.Line2D([], [], color=F.ENG["vllm"]["color"], marker="o", markersize=4,
                          label="vLLM (cumulative since start)"),
               plt.Line2D([], [], color=F.ENG["sglang"]["color"], marker="s", markersize=4,
                          label="SGLang (last batch)"),
               plt.Line2D([], [], color=F.ENG["tensorrt-llm"]["color"], marker="^", linestyle="", markersize=5,
                          label="TensorRT-LLM (one value per run)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.05), handlelength=1.8)
    fig.savefig(a.out / "fig_supp_prefix_hits.pdf")
    plt.close(fig)
    pd.DataFrame(recs).to_csv(a.out / "supp_prefix_hits.csv", index=False)
    print(pd.DataFrame(recs).round(1).to_string())


if __name__ == "__main__":
    main()
