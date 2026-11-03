#!/usr/bin/env python3
"""RQ1: TTFT vs. position within a rate config, per engine.

Grid: rows = max_num_seqs (64 top, 256 bottom); columns = arrival-rate configs
(2, 4, 8, 16 req/s Poisson, then bursty). x = the request's position in send
order within its rate config; y = TTFT (log). Colour = engine.

Requests that timed out before their first token are recorded with ttft_s = 0;
they are drawn at the 300 s ceiling as hollow markers (censored), not dropped.
Zero-token errors (status "error") are a model/prompt issue, not latency, and
are excluded.

    python scripts/plot_rq1_ttft_position.py --rq1 results/Aggregated_results/rq1_trimmed.parquet \
        --out docs/figures/rq1_ttft
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
MODEL = {"llama31_8b": "Llama-3.1-8B", "mistral_nemo_12b": "Mistral-NeMo-12B", "qwen15_14b": "Qwen1.5-14B"}
RATES = [("low_poisson", "2 req/s"), ("med_poisson", "4 req/s"), ("high_poisson", "8 req/s"),
         ("vhigh_poisson", "16 req/s"), ("bursty", "bursty")]
CEIL = 300.0
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


def prepare(r1: pd.DataFrame, model: str, workload: str) -> pd.DataFrame:
    d = r1[(r1.model == model) & (r1.workload_name == workload) & (r1.max_concurrent == 256)
           & (r1.status != "error")].copy()
    d["censored"] = (d.status == "timeout") & (d.ttft_s <= 0)
    d["y"] = np.where(d.censored, CEIL, d.ttft_s.clip(lower=0.02))
    d = d.sort_values("send_time")
    d["pos"] = d.groupby(["engine", "max_num_seqs", "rate_label"]).cumcount()
    return d


def plot(d: pd.DataFrame, model: str, workload: str, out: Path) -> pd.DataFrame:
    seqs = sorted(d.max_num_seqs.unique())
    fig, axes = plt.subplots(len(seqs), len(RATES), figsize=(7.0, 1.55 * len(seqs) + 0.5),
                             sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.18, wspace=0.08))
    stats = []
    for i, s in enumerate(seqs):
        for j, (rate, rlab) in enumerate(RATES):
            ax = axes[i, j]
            for e, st in ENG.items():
                g = d[(d.engine == e) & (d.max_num_seqs == s) & (d.rate_label == rate)]
                if g.empty:
                    continue
                ok = g[~g.censored]
                ax.scatter(ok.pos, ok.y, s=1.6, color=st["color"], linewidth=0, alpha=0.55, rasterized=True)
                c = g[g.censored]
                if len(c):
                    ax.scatter(c.pos, c.y, s=5, facecolor="none", edgecolor=st["color"], linewidth=0.4,
                               marker=st["marker"], rasterized=True)
                # rolling median shows the shape through the scatter
                ax.plot(g.pos, g.y.rolling(25, center=True, min_periods=5).median(),
                        color=st["color"], linewidth=1.1)
                slow = g.y > 10
                stats.append(dict(model=model, workload=workload, max_num_seqs=s, rate=rate, engine=e,
                                  n=len(g), ttft_p50=g.y.median(), ttft_p95=g.y.quantile(.95),
                                  share_over_10s=slow.mean(), first_pos_over_10s=int(g.pos[slow].min()) if slow.any() else None,
                                  censored=int(g.censored.sum())))
            ax.axhline(CEIL, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
            ax.set_yscale("log"); ax.set_ylim(0.02, 600)
            ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 1, 10, 100]))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            if i == 0:
                ax.set_title(rlab, loc="left")
            if i == len(seqs) - 1 and j == len(RATES) // 2:
                ax.set_xlabel("Request position within the rate config (send order)")
            if j == 0:
                ax.set_ylabel(f"max_num_seqs {s}\nTTFT (s)")
        axes[i, -1].text(1.02, CEIL, "300 s\ntimeout", transform=axes[i, -1].get_yaxis_transform(),
                         fontsize=5.5, color=INK2, va="center")
    handles = [plt.Line2D([], [], color=v["color"], linewidth=1.4, label=v["label"]) for v in ENG.values()]
    handles.append(plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="none", linestyle="",
                              markersize=3.5, label="timed out before first token"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"TTFT vs. position within a rate config — {MODEL[model]}, {workload}, concurrency 256, L40S\n"
                 "dots: requests · line: rolling median of 25", x=0.06, ha="left", y=1.03, fontsize=7.5)
    name = f"fig_rq1_ttft_position_{model}_{workload}"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}")
    return pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1", type=Path, required=True, help="trimmed RQ1 parquet from plot_rq1.py")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq1_ttft"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    r1 = pd.read_parquet(a.rq1)
    allstats = []
    for m in MODEL:
        for w in ("mixed_real", "chat_multiturn"):
            d = prepare(r1, m, w)
            if not d.empty:
                allstats.append(plot(d, m, w, a.out))
    pd.concat(allstats).to_csv(a.out / "rq1_ttft_position_stats.csv", index=False)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
