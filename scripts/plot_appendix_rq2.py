#!/usr/bin/env python3
"""Appendix figure for RQ2: why chunked prefill removes stalls on some engines and not others.

    python scripts/plot_appendix_rq2.py --rq2 results/Aggregated_results/rq2_prefill_handling.csv \
        --out docs/figures/paper_appendix

For each finished request on chat_real_long (prefix caching on), count the other requests whose
prefill completed (first token emitted) while it was streaming, i.e. between its own first and
last token, and relate that count to its longest gap between tokens. Every prompt in this
workload has at least 500 tokens, so each such event is a long prefill that ran while this
request was decoding. Without chunking a prefill pauses running decodes; chunking should cap the
pause at one chunk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

ENG = {
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^"),
}
STRATS = {"baseline_full_fp16_tp1": ("no chunking", (0, (3, 1.5))), "chunked_512_fp16_tp1": ("chunked (512)", "-")}
BINS, LABELS = [-1, 0, 5, 15, 40, 1000], ["0", "1-5", "6-15", "16-40", "41+"]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_appendix"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    d = pd.read_csv(a.rq2, usecols=["engine", "workload_name", "strategy_name", "rate_label", "status", "send_time",
                                    "ttft_s", "e2e_s", "output_tokens", "itl_max_ms"], low_memory=False)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[(d.send_time > 0) & (d.workload_name == "chat_real_long") & d.strategy_name.isin(STRATS)]
    rows = []
    for (e, s, r), g in d.groupby(["engine", "strategy_name", "rate_label"]):
        started = g[g.ttft_s > 0]
        prefills_done = np.sort((started.send_time + started.ttft_s).values)
        ok = g[(g.status == "ok") & (g.output_tokens > 20)]
        t_first, t_end = (ok.send_time + ok.ttft_s).values, (ok.send_time + ok.e2e_s).values
        n = np.searchsorted(prefills_done, t_end, side="left") - np.searchsorted(prefills_done, t_first, side="right")
        rows.append(pd.DataFrame(dict(engine=e, strategy=s, rate=r, prefills=n, gap_s=ok.itl_max_ms.values / 1000)))
    x = pd.concat(rows, ignore_index=True)
    x["bin"] = pd.cut(x.prefills, BINS, labels=LABELS)
    stats = x.groupby(["engine", "strategy", "bin"], observed=True).gap_s.agg(["median", "count"]).reset_index()
    stats.to_csv(a.out / "appendix_a3_chunking_mechanism.csv", index=False)
    corr = x.groupby(["engine", "strategy"]).apply(lambda t: np.corrcoef(t.prefills, t.gap_s)[0, 1])
    corr.rename("pearson_r").reset_index().to_csv(a.out / "appendix_a3_chunking_correlation.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (e, st) in zip(axes, ENG.items()):
        for s, (lab, ls) in STRATS.items():
            t = stats[(stats.engine == e) & (stats.strategy == s)].set_index("bin").reindex(LABELS)
            ax.plot(range(len(LABELS)), t["median"].values, color=st["color"], linestyle=ls,
                    marker=st["marker"] if s == "chunked_512_fp16_tp1" else None, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yscale("log"); ax.set_ylim(0.04, 3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(st["label"], loc="left")
        ax.set_xlabel("Other prefills during the stream")
    axes[0].set_ylabel("Longest token gap,\nmedian (s)")
    fig.legend(handles=[plt.Line2D([], [], color=INK2, label="chunked prefill (512 tokens)"),
                        plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="no chunking")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(a.out / "fig_appendix_a3_chunking.pdf")
    print(stats.round(2).to_string()); print(corr.round(2))


if __name__ == "__main__":
    main()
