#!/usr/bin/env python3
"""RQ4: does the RQ1 picture hold on different hardware? L40S vs GB10. Paper-ready PDFs.

    python scripts/plot_rq4.py --rq1-dir results/Aggregated_results --gb10-report rq5_gb10_report.html \
        --out docs/figures/paper_rq4 --appendix-out docs/figures/paper_appendix

The GB10 data exist only as per-rate-configuration summaries (the report's embedded table), so every
comparison uses quantities computed identically on both platforms from per-configuration values:
  median TTFT        median over rate configs of the per-config TTFT p50 (finished requests, from send)
  median TPOT        1000 / median per-request decode rate (tok/s); on the L40S, tokens_per_second
                     equals 1000/TPOT to within 2%
  peak KV            maximum KV-cache use over rate configs (not live for TensorRT-LLM on the L40S)
  completion         share of requests that finished
Both platforms: concurrency 256, max_num_seqs 256, mixed_real and chat_multiturn, five arrival processes.

Figure (main):    fig_rq4_platforms     peak KV, median TTFT, median TPOT per model x engine, L40S -> GB10
Figure (appendix): fig_appendix_a6_gb10 (a) vLLM preemptions vs peak KV on both platforms;
                                        (b, c) GB10 request outcomes, reasoning vs other tasks, with
                                        the L40S finished share for reference.
"""
from __future__ import annotations

import argparse
import json
import re
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
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


FS = 1.0  # font scale; the paper versions set it per figure (see PAPER_FS)


def fz(x: float) -> float:
    return x * FS


def style(fs: float = 1.0):
    global FS
    FS = fs
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7 * fs, "axes.titlesize": 7.5 * fs, "axes.labelsize": 7 * fs,
        "xtick.labelsize": 6.5 * fs, "ytick.labelsize": 6.5 * fs, "legend.fontsize": 6.3 * fs, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plain_log(axis):
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def load_gb10(path: Path):
    s = path.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const D = (\{.*?\});\n", s, re.S).group(1))
    rows = pd.DataFrame(data["rows"]).rename(columns={"workload": "workload_name", "seqs": "max_num_seqs",
                                                      "rate": "rate_label", "pre": "preemptions"})
    return rows, pd.DataFrame(data["tasks"])


def load_l40s(rq1_dir: Path) -> pd.DataFrame:
    cols = ["engine", "workload_name", "max_concurrent", "max_num_seqs", "rate_label", "status", "ttft_s", "e2e_s",
            "output_tokens", "tokens_per_second", "kv_cache_usage_pct", "num_preemptions"]
    d = pd.concat([pd.read_csv(rq1_dir / f"rq1_{m}.csv", usecols=cols, low_memory=False).assign(model=m)
                   for m in MODEL], ignore_index=True)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[d.max_concurrent == 256].copy()
    trt = d.engine == "tensorrt-llm"
    d.loc[~trt, "kv_cache_usage_pct"] *= 100
    d.loc[trt, ["kv_cache_usage_pct", "num_preemptions"]] = np.nan  # not live on the L40S harness
    k = ["model", "engine", "workload_name", "max_num_seqs", "rate_label"]
    fin = d[(d.status == "ok") & (d.output_tokens > 1)]
    pc = fin.groupby(k).agg(ttft_p50=("ttft_s", "median"), tps=("tokens_per_second", "median"))
    pc = pc.join(d.groupby(k).agg(kv_peak=("kv_cache_usage_pct", "max"), preemptions=("num_preemptions", "max"),
                                  ok_pct=("status", lambda s: 100 * (s == "ok").mean()), n=("status", "size")))
    return pc.reset_index()


def summarise(pc: pd.DataFrame, platform: str) -> pd.DataFrame:
    s = pc[pc.max_num_seqs == 256]

    def agg(x):
        return pd.Series(dict(ttft_med=x.ttft_p50.median(), tpot_med=1000 / x.tps.median(), kv_peak=x.kv_peak.max(),
                              preempt_max=x.preemptions.max(), completion=(x.ok_pct * x.n).sum() / x.n.sum()))
    out = s.groupby(["model", "engine"]).apply(agg).reset_index()
    out["platform"] = platform
    return out


def fig_main(summ: pd.DataFrame, out: Path):
    rows = [(m, e) for m in MODEL for e in ENG]
    panels = [("kv_peak", "Peak KV-cache use (%)", False, (0, 105)),
              ("ttft_med", "Median TTFT (s)", True, (0.05, 400)),
              ("tpot_med", "Median TPOT (ms)", True, (30, 700))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (col, lab, logx, lim) in zip(axes, panels):
        for i, (m, e) in enumerate(rows):
            if i % 3 == 0 and (i // 3) % 2 == 0:
                ax.axhspan(i - 0.5, i + 2.5, color="#f4f3f0", linewidth=0, zorder=0)
            st = ENG[e]
            a = summ[(summ.model == m) & (summ.engine == e) & (summ.platform == "L40S")][col]
            b = summ[(summ.model == m) & (summ.engine == e) & (summ.platform == "GB10")][col]
            a = a.iloc[0] if len(a) else np.nan
            b = b.iloc[0] if len(b) else np.nan
            if np.isfinite(a) and np.isfinite(b):
                ax.annotate("", xy=(b, i), xytext=(a, i),
                            arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=1.0, mutation_scale=7))
            if np.isfinite(a):
                ax.scatter(a, i, s=20, facecolor="white", edgecolor=st["color"], marker=st["marker"], linewidth=0.9,
                           zorder=3)
            if np.isfinite(b):
                ax.scatter(b, i, s=22, color=st["color"], marker=st["marker"], edgecolor="white", linewidth=0.4,
                           zorder=3)
        if logx:
            ax.set_xscale("log"); plain_log(ax.xaxis)
        ax.set_xlim(*lim); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        if col == "tpot_med":
            ax.axvline(200, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
            ax.text(208, -0.75, "200 ms SLO", fontsize=fz(5.8), color=INK2, va="center")
        if col == "ttft_med":
            ax.axvline(2, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
            ax.text(2.15, -0.75, "2 s SLO", fontsize=fz(5.8), color=INK2, va="center")
    axes[0].set_yticks(range(len(rows)), [f"{MODEL[m]} · {ENG[e]['label']}" for m, e in rows], fontsize=fz(6))
    axes[0].set_ylim(len(rows) - 0.5, -0.9)
    handles = [plt.Line2D([], [], color=v["color"], marker=v["marker"], linestyle="", markersize=4, label=v["label"])
               for v in ENG.values()]
    handles += [plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="white", linestyle="", label="L40S"),
                plt.Line2D([], [], color=INK2, marker="o", linestyle="", label="GB10 (arrow head)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out / "fig_rq4_platforms.pdf")
    plt.close(fig)


def l40s_task_outcomes(rq1_dir: Path) -> pd.DataFrame:
    """Finished share per task on the L40S (mixed_real, C = 256, both max_num_seqs)."""
    d = pd.concat([pd.read_csv(rq1_dir / f"rq1_{m}.csv", usecols=["engine", "workload_name", "max_concurrent",
                                                                    "task_type", "status"], low_memory=False)
                   .assign(model=m) for m in MODEL], ignore_index=True)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[(d.max_concurrent == 256) & (d.workload_name == "mixed_real")]
    return (d.groupby(["model", "engine", "task_type"]).status.apply(lambda s: 100 * (s == "ok").mean())
            .rename("ok_pct").reset_index().rename(columns={"task_type": "task"}))


def fig_appendix(l40: pd.DataFrame, gb: pd.DataFrame, tasks: pd.DataFrame, l40_tasks: pd.DataFrame, out: Path):
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw=dict(width_ratios=[0.95, 0.36, 1, 1], wspace=0.1))
    axs[1].set_visible(False)
    axes = [axs[0], axs[2], axs[3]]
    ax = axes[0]
    shades = {"llama31_8b": ("#1f1f1f", "o"), "mistral_nemo_12b": ("#8a8984", "s"), "qwen15_14b": ("#2a78d6", "^")}
    for src, filled in ((l40, False), (gb, True)):
        v = src[src.engine == "vllm"]
        for m, (c, mk) in shades.items():
            x = v[v.model == m]
            off = {"llama31_8b": -2.2, "mistral_nemo_12b": 0.0, "qwen15_14b": 2.2}[m]
            xs = np.where(x.kv_peak >= 99, x.kv_peak + off, x.kv_peak)  # spread saturated runs for visibility
            ax.scatter(xs, x.preemptions, s=16, marker=mk, facecolor=c if filled else "white", edgecolor=c,
                       linewidth=0.8, zorder=3, alpha=0.9)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlim(0, 106); ax.set_ylim(-0.3, 2000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
    ax.axvline(100, color=INK2, linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_xlabel("Peak KV-cache use (%)"); ax.set_ylabel("Preemptions per run (vLLM)")
    ax.set_title("(a) Preemption needs a full cache", loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=4, label=MODEL[m])
                       for m, (c, mk) in shades.items()] +
              [plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="white", linestyle="", label="L40S"),
               plt.Line2D([], [], color=INK2, marker="o", linestyle="", label="GB10")],
              loc="upper left", fontsize=fz(5.6), handletextpad=0.2)

    rows = [(m, e) for m in MODEL for e in ENG]
    other = ["chat", "code", "summarization"]
    for ax, sel, title in ((axes[1], ["reasoning"], "(b) Reasoning requests"),
                           (axes[2], other, "(c) Chat, code, summarisation")):
        for i, (m, e) in enumerate(rows):
            t = tasks[(tasks.model == m) & (tasks.engine == e) & tasks.task.isin(sel)]
            w = t.n / t.n.sum()
            fin, nf, nd = (t.ok_pct * w).sum(), (t.ttft_to_pct * w).sum(), (t.e2e_to_pct * w).sum()
            ax.barh(i, fin, color=ENG[e]["color"], height=0.72, linewidth=0)
            ax.barh(i, nf, left=fin, color=INK, height=0.72, linewidth=0)
            ax.barh(i, nd, left=fin + nf, color="white", hatch="//////", edgecolor=INK2, height=0.72, linewidth=0.3)
            ref = l40_tasks[(l40_tasks.model == m) & (l40_tasks.engine == e) & l40_tasks.task.isin(sel)].ok_pct.mean()
            ax.plot(ref, i, marker="D", markerfacecolor="white", markeredgecolor=INK, markersize=3.6,
                    markeredgewidth=0.9, zorder=4, clip_on=False)
        ax.set_xlim(0, 100); ax.set_xlabel("Share of GB10 requests (%)"); ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left"); ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)), [f"{MODEL[m].split('-')[0]} · {ENG[e]['label']}" for m, e in rows]
                      if ax is axes[1] else [], fontsize=fz(5.8))
        if ax is axes[2]:
            ax.tick_params(axis="y", length=0)
    axes[1].yaxis.set_tick_params(pad=1)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, label="finished (engine colour)"),
                        plt.Rectangle((0, 0), 1, 1, color=INK, label="no first token in 120 s"),
                        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="//////",
                                      label="not finished in 300 s"),
                        plt.Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor=INK, linestyle="",
                                   markersize=3.6, label="finished on the L40S")],
               loc="upper center", bbox_to_anchor=(0.62, -0.02), ncol=4, fontsize=fz(5.9))
    fig.savefig(out / "fig_appendix_a6_gb10.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--gb10-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_rq4"))
    ap.add_argument("--appendix-out", type=Path, default=Path("docs/figures/paper_appendix"))
    a = ap.parse_args()
    for p in (a.out, a.appendix_out):
        p.mkdir(parents=True, exist_ok=True)
    gb, tasks = load_gb10(a.gb10_report)
    l40 = load_l40s(a.rq1_dir)
    summ = pd.concat([summarise(l40, "L40S"), summarise(gb, "GB10")], ignore_index=True)
    summ.to_csv(a.out / "rq4_platform_summary.csv", index=False)
    tasks.to_csv(a.appendix_out / "appendix_a6_gb10_tasks.csv", index=False)
    style(1.2)  # main figure: paper font scale
    fig_main(summ, a.out)
    style()
    fig_appendix(l40, gb, tasks, l40s_task_outcomes(a.rq1_dir), a.appendix_out)
    print(summ.pivot_table(index=["model", "engine"], columns="platform",
                           values=["ttft_med", "tpot_med", "kv_peak", "completion"]).round(2).to_string())


if __name__ == "__main__":
    main()
