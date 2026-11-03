#!/usr/bin/env python3
"""chat_multiturn prefix-cache hits per engine, against what the workload makes reusable.

(a) RQ1: reported prefix-cache hit rate per engine and model, next to the workload's
    reusable-prefix ceiling. (b) RQ2: TTFT by conversation turn with prefix caching
    on vs off, per engine — what the hits buy.

The engines report the metric differently, and the figure says so:
  vLLM          cumulative token hit rate since server start (one server per rate
                config); we take its final value per rate config.
  SGLang        hit rate of the most recent batch (instantaneous); we average the
                snapshots, which is not token-weighted.
  TensorRT-LLM  one value scraped after each run; we take that value.
Workload ceiling (token-weighted over all requests): tokens of the previous turn's
prompt, which every later turn resends, over all prompt tokens. Upper bound adds the
previous turn's output, reusable only if the harness resends the generated reply.

    python scripts/plot_prefix_hits.py --rq1-dir results/Aggregated_results \
        --rq2 results/Aggregated_results/rq2_prefill_handling.csv --out docs/figures/prefix
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENG = {
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o", how="cumulative"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s", how="last batch, averaged"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^", how="post-run"),
}
MODEL = {"llama31_8b": "Llama-3.1-8B", "mistral_nemo_12b": "Mistral-NeMo-12B", "qwen15_14b": "Qwen1.5-14B"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLS = ["engine", "model", "workload_name", "max_concurrent", "max_num_seqs", "rate_label", "run_id", "status",
        "send_time", "e2e_s", "ttft_s", "prefix_cache_hit_rate", "conversation_id", "turn_index",
        "prompt_token_len", "output_tokens"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read(p, extra=()):
    d = pd.read_csv(p, usecols=lambda c: c in set(COLS) | set(extra), low_memory=False)
    d["engine"] = d["engine"].replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    return d


def ceiling(d: pd.DataFrame) -> tuple[float, float]:
    """Token-weighted reusable share of prompt tokens (lower, upper bound)."""
    keys = ["engine", "max_concurrent", "max_num_seqs", "rate_label", "conversation_id"]
    d = d.sort_values(keys + ["turn_index"])
    g = d.groupby(keys)
    prev_prompt = g.prompt_token_len.shift(1).fillna(0)
    prev_out = g.output_tokens.shift(1).fillna(0)
    lo = np.minimum(prev_prompt, d.prompt_token_len)
    hi = np.minimum(prev_prompt + prev_out, d.prompt_token_len)
    tot = d.prompt_token_len.sum()
    return lo.sum() / tot, hi.sum() / tot


def measured(d: pd.DataFrame, engine: str) -> float:
    g = d[(d.engine == engine) & d.prefix_cache_hit_rate.notna()]
    if g.empty:
        return np.nan
    if engine == "sglang":
        return g.prefix_cache_hit_rate.mean()
    # vLLM: final value of the cumulative ratio; TRT-LLM: the single post-run value
    g = g.assign(t=g.send_time + g.e2e_s).sort_values("t")
    return g.groupby("rate_label").prefix_cache_hit_rate.last().median()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--rq2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/prefix"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()

    rows = []
    for m in MODEL:
        d = read(a.rq1_dir / f"rq1_{m}.csv")
        d = d[(d.workload_name == "chat_multiturn") & (d.max_concurrent == 256) & (d.max_num_seqs == 256)]
        lo, hi = ceiling(d)
        for e in ENG:
            rows.append(dict(model=m, engine=e, hit_rate=measured(d, e), ceiling_lo=lo, ceiling_hi=hi))
    hits = pd.DataFrame(rows)
    hits.to_csv(a.out / "prefix_hits_rq1.csv", index=False)

    r2 = read(a.rq2, extra=("strategy_name",))
    r2 = r2[(r2.workload_name == "chat_multiturn") & (r2.status == "ok")
            & r2.strategy_name.isin(["baseline_full_fp16_tp1", "baseline_full_fp16_tp1_nopfx"])]
    turn = (r2.groupby(["engine", "strategy_name", "turn_index"])
              .agg(ttft_p50=("ttft_s", "median"), prompt=("prompt_token_len", "median"), n=("ttft_s", "size"))
              .reset_index())
    turn.to_csv(a.out / "prefix_ttft_by_turn_rq2.csv", index=False)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.28))

    # (a) hit rate vs ceiling
    x = np.arange(len(MODEL))
    w = 0.22
    for i, m in enumerate(MODEL):
        h = hits[hits.model == m].iloc[0]
        ax.plot([i - 0.4, i + 0.4], [h.ceiling_lo * 100] * 2, color=INK, linewidth=1, linestyle=(0, (3, 1.5)), zorder=4)
        ax.text(i + 0.4, h.ceiling_lo * 100 + 1.5, f"reusable {h.ceiling_lo * 100:.0f}%", fontsize=5.8,
                color=INK2, ha="right")
    for j, (e, s) in enumerate(ENG.items()):
        v = hits[hits.engine == e].set_index("model").reindex(list(MODEL)).hit_rate.values * 100
        xs = x + (j - 1) * w
        ax.bar(xs, v, width=w * 0.9, color=s["color"], zorder=3, edgecolor="white", linewidth=0.6)
        for xi, vi in zip(xs, v):
            if np.isfinite(vi):
                inside = vi >= 12
                ax.text(xi, vi - 1.5 if inside else vi + 1.2, f"{vi:.0f}", ha="center", va="top" if inside else "bottom",
                        fontsize=6, color="white" if inside else INK, fontweight="bold" if inside else "normal")
    ax.set_xticks(x, list(MODEL.values())); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 80); ax.set_ylabel("Prefix-cache hit rate (%)")
    ax.set_title("(a) Reported hit rate vs. what the workload makes reusable\n"
                 "RQ1 chat_multiturn, conc. 256, max_num_seqs 256, L40S", loc="left")
    handles = [matplotlib.patches.Patch(color=s["color"], label=f"{s['label']} ({s['how']})") for s in ENG.values()]
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (3, 1.5)), label="reusable: previous turn's prompt"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=5.8)

    # (b) TTFT by turn, prefix on vs off
    for e, s in ENG.items():
        for strat, ls, mk in (("baseline_full_fp16_tp1", "-", s["marker"]), ("baseline_full_fp16_tp1_nopfx", (0, (3, 1.5)), None)):
            t = turn[(turn.engine == e) & (turn.strategy_name == strat) & (turn.n >= 30)]
            bx.plot(t.turn_index + 1, t.ttft_p50 * 1000, color=s["color"], linestyle=ls, marker=mk, markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.5)
    bx.set_xlabel("Conversation turn"); bx.set_ylabel("TTFT p50 (ms)")
    bx.set_title("(b) What the hits buy: TTFT by turn\nRQ2 Mistral-NeMo, chat_multiturn, conc. 256", loc="left")
    pr = turn[(turn.engine == "vllm") & (turn.strategy_name == "baseline_full_fp16_tp1") & (turn.n >= 30)]
    sec = bx.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    sec.set_xticks(pr.turn_index + 1, [f"{p / 1000:.1f}k" if p >= 1000 else f"{p:.0f}" for p in pr.prompt])
    sec.set_xlabel("median prompt tokens", fontsize=6, color=INK2); sec.tick_params(labelsize=5.8, colors=INK2)
    h2 = [plt.Line2D([], [], color=s["color"], marker=s["marker"], markersize=3.5, label=s["label"]) for s in ENG.values()]
    h2 += [plt.Line2D([], [], color=INK2, label="prefix caching on"),
           plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="off")]
    bx.legend(handles=h2, loc="upper left", fontsize=6)

    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"fig_prefix_hits_chat_multiturn.{ext}", dpi=300 if ext == "png" else None)
    print(hits.round(3).to_string())
    print(turn.round(3).to_string())


if __name__ == "__main__":
    import matplotlib.patches  # noqa: F401
    main()
