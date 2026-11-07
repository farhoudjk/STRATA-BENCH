#!/usr/bin/env python3
"""Appendix figures for RQ1: the mechanisms behind the main findings. PDF only.

    python scripts/plot_appendix_rq1.py --rq1-dir results/Aggregated_results --out docs/figures/paper_appendix

A1  Admission behaviour over a run: TTFT over arrival time (Llama, Qwen), running requests over
    time (Qwen), and the position at which TTFT first exceeds 10 s vs. arrival rate.
A2  Preemption and its cost (vLLM only: the one engine with a live preemption counter), and the
    output-length difference between models.
TTFT is measured from arrival (open loop); requests with no first token within 300 s are drawn at
the ceiling. Colour = engine; in A2, where only vLLM is shown, colour = model (a palette distinct from the
engine colours).
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
MODEL = {
    "llama31_8b": dict(label="Llama-3.1-8B", grey="#1f1f1f", marker="o", ls="-"),
    "mistral_nemo_12b": dict(label="Mistral-NeMo-12B", grey="#7a7975", marker="s", ls=(0, (4, 1.5))),
    "qwen15_14b": dict(label="Qwen1.5-14B", grey="#b5b3ae", marker="^", ls=(0, (1.2, 1.2))),
}
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
RPS = {"low_poisson": 2, "med_poisson": 4, "high_poisson": 8, "vhigh_poisson": 16}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CEIL = 300.0
COLS = ["engine", "workload_name", "task_type", "max_concurrent", "max_num_seqs", "rate_label", "status", "error",
        "arrival_time", "send_time", "ttft_s", "e2e_s", "output_tokens", "itl_max_ms", "kv_cache_usage_pct",
        "num_running_seqs", "num_preemptions", "num_preemptions_delta"]


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


def plain_log(axis):
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def load(rq1_dir: Path) -> pd.DataFrame:
    fr = []
    for m in MODEL:
        x = pd.read_csv(rq1_dir / f"rq1_{m}.csv", usecols=lambda c: c in COLS, low_memory=False)
        x["model"] = m
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[(d.max_concurrent == 256) & (d.workload_name == "mixed_real") & (d.send_time > 0)].copy()
    k = ["model", "engine", "max_num_seqs", "rate_label"]
    d["t_arr"] = d.arrival_time - d.groupby(k).arrival_time.transform("min")
    d["t_send"] = d.send_time - d.groupby(k).send_time.transform("min")
    wait = (d.t_send - d.t_arr).clip(lower=0)
    no_first = (d.status == "timeout") & (d.ttft_s <= 0)
    d["ttft_user"] = np.where(no_first, CEIL, d.ttft_s) + wait
    d["t_done"] = d.t_send + np.where(d.status == "ok", d.e2e_s, np.nan)
    d.loc[d.engine == "tensorrt-llm", ["kv_cache_usage_pct", "num_running_seqs", "num_preemptions",
                                       "num_preemptions_delta"]] = np.nan  # never live in this harness
    d.loc[d.engine != "tensorrt-llm", "kv_cache_usage_pct"] *= 100
    return d[d.status != "error"]


def turn_point(g: pd.DataFrame, thr=10.0, win=9) -> float:
    """Arrival position at which the rolling-median TTFT first exceeds thr seconds."""
    g = g.sort_values("t_arr").reset_index(drop=True)
    r = g.ttft_user.rolling(win, center=True, min_periods=3).median()
    hit = np.flatnonzero(r.values > thr)
    return float(hit[0]) if len(hit) else np.nan


# ── A1: admission behaviour ──────────────────────────────────────────────────
def a1(d, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1, 0.95]))
    rate, seqs = "high_poisson", 256
    stats = []
    for ax, m, tag in ((axes[0], "llama31_8b", "(a)"), (axes[1], "qwen15_14b", "(b)")):
        for e, st in ENG.items():
            g = d[(d.model == m) & (d.engine == e) & (d.max_num_seqs == seqs) & (d.rate_label == rate)].sort_values("t_arr")
            ax.scatter(g.t_arr, g.ttft_user.clip(lower=0.03), s=1.2, color=st["color"], alpha=0.35, linewidth=0,
                       rasterized=True)
            ax.plot(g.t_arr, g.ttft_user.rolling(15, center=True, min_periods=5).median().clip(lower=0.03),
                    color=st["color"], linewidth=1.2)
            stats.append(dict(panel=tag, model=m, engine=e, turn_position=turn_point(g)))
        ax.axhline(CEIL, color=INK2, linestyle=(0, (3, 2)), linewidth=0.6)
        ax.set_yscale("log"); ax.set_ylim(0.03, 800); plain_log(ax.yaxis)
        ax.set_xlabel("Arrival time in run (s)")
        ax.set_title(f"{tag} {MODEL[m]['label']}: TTFT", loc="left")
    axes[0].set_ylabel("TTFT from arrival (s)")

    ax = axes[2]
    # Requests streaming at time t, from client timestamps only (works for every engine):
    # first token received and not yet finished. Started-but-unfinished requests end at the 600 s timeout.
    for e, st in ENG.items():
        g = d[(d.model == "qwen15_14b") & (d.engine == e) & (d.max_num_seqs == seqs) & (d.rate_label == rate)]
        started = g.ttft_s > 0
        t_first = g.t_send + g.ttft_s
        t_end = np.where(g.status == "ok", g.t_send + g.e2e_s, np.where(started, g.t_send + 600.0, np.nan))
        grid = np.arange(0, np.nanmax(t_end), 2.0)
        cnt = np.array([int(((t_first <= t) & (t_end > t) & started).sum()) for t in grid])
        ax.plot(grid, cnt, color=st["color"], linewidth=1.1)
        stats.append(dict(panel="(c)", model="qwen15_14b", engine=e, streaming_p50=float(np.median(cnt[cnt > 0])),
                          streaming_max=int(cnt.max()), started=int(started.sum()), requests=len(g)))
    ax.set_ylim(0, None); ax.set_xlabel("Time in run (s)"); ax.set_ylabel("Requests streaming")
    ax.set_title("(c) Qwen1.5: in service", loc="left")

    ax = axes[3]
    for e, st in ENG.items():
        pts = []
        for r in RATES:
            g = d[(d.model == "llama31_8b") & (d.engine == e) & (d.max_num_seqs == 64) & (d.rate_label == r)]
            pts.append(turn_point(g))
            stats.append(dict(panel="(d)", model="llama31_8b", engine=e, rps=RPS[r], turn_position=pts[-1]))
        ax.plot([RPS[r] for r in RATES], pts, color=st["color"], marker=st["marker"], markersize=4,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); plain_log(ax.xaxis)
    ax.set_ylim(0, 500); ax.set_xlabel("Arrival rate (req/s)"); ax.set_ylabel("Request position")
    ax.set_title("(d) Llama: backlog from", loc="left")
    handles = [plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4, label=v["label"])
               for v in ENG.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(out / "fig_appendix_a1_admission.pdf")
    plt.close(fig)
    return pd.DataFrame(stats)


# ── A2: preemption, stalls, output length ────────────────────────────────────
# Model colours for A2, where only vLLM is shown; chosen to be distinct from the engine palette.
MCOL = {"llama31_8b": "#7b4fd6", "mistral_nemo_12b": "#d6488a", "qwen15_14b": "#b07d00"}
A2_RC = {"font.size": 8, "axes.titlesize": 8.4, "axes.labelsize": 8, "xtick.labelsize": 7.4,
         "ytick.labelsize": 7.4, "legend.fontsize": 7.4}


def a2(d, out):
    with plt.rc_context(A2_RC):
        return _a2(d, out)


def _a2(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw=dict(wspace=0.42))
    stats = []
    v = d[d.engine == "vllm"]

    # (a) peak KV vs preemptions, one point per run
    ax = axes[0]
    for m, st in MODEL.items():
        rows = []
        for (q, r), g in v[v.model == m].groupby(["max_num_seqs", "rate_label"]):
            rows.append((g.kv_cache_usage_pct.max(), g.num_preemptions.max()))
        a = np.array(rows)
        off = {"llama31_8b": -1.6, "mistral_nemo_12b": 0.0, "qwen15_14b": 1.6}[m]
        xs = np.where(a[:, 0] >= 99, a[:, 0] + off, a[:, 0])  # spread runs at a full cache
        ax.scatter(xs, a[:, 1], s=20, marker=st["marker"], color=MCOL[m], edgecolor="white", linewidth=0.4,
                   alpha=0.9, zorder=3)
        stats += [dict(panel="(a)", model=m, kv_peak=k_, preemptions=p_) for k_, p_ in rows]
    ax.set_yscale("symlog", linthresh=1); ax.set_xlim(0, 106)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
    ax.axvline(100, color=INK2, linestyle=(0, (3, 2)), linewidth=0.6)
    ax.text(98, 0.25, "cache\nfull", ha="right", va="bottom", fontsize=6.6, color=INK2)
    ax.set_xlabel("Peak KV-cache use (%)"); ax.set_ylabel("Preemptions per run")
    ax.set_title("(a) Only a full cache preempts", loc="left")

    # (b) longest gap, requests that overlapped a preemption (solid) or not (dashed)
    ax = axes[1]
    s = v[(v.max_num_seqs == 256) & (v.status == "ok") & v.itl_max_ms.notna()]
    for m in ("llama31_8b", "qwen15_14b"):
        for hit, ls, lw in ((False, (0, (3, 1.6)), 1.1), (True, "-", 1.5)):
            x = np.sort(s[(s.model == m) & ((s.num_preemptions_delta > 0) == hit)].itl_max_ms.values / 1000)
            if len(x) < 5:
                continue
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MCOL[m], linestyle=ls, linewidth=lw)
            stats.append(dict(panel="(b)", model=m, overlaps_preemption=hit, n=len(x), gap_p50_s=np.median(x),
                              gap_p90_s=np.quantile(x, .9)))
    ax.set_xscale("log"); plain_log(ax.xaxis); ax.set_xticks([0.1, 1, 10, 100])
    ax.axvline(1, color=INK2, linestyle=(0, (1, 1.5)), linewidth=0.6)
    ax.set_xlabel("Longest token gap (s)"); ax.set_ylabel("Share of requests")
    ax.set_title("(b) Preemption causes stalls", loc="left")

    # (c) reasoning output length per model
    ax = axes[2]
    r = d[(d.status == "ok") & (d.task_type == "reasoning") & (d.output_tokens > 0)]
    for m, st in MODEL.items():
        x = np.sort(r[r.model == m].output_tokens.values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MCOL[m], linewidth=1.4)
        stats.append(dict(panel="(c)", model=m, mean_tokens=x.mean(), median_tokens=np.median(x)))
    ax.set_xscale("log"); plain_log(ax.xaxis)
    ax.set_xlabel("Reasoning output tokens"); ax.set_ylabel("Share of requests")
    ax.set_title("(c) Llama answers 6× longer", loc="left")

    handles = [plt.Line2D([], [], color=MCOL[m], marker=st["marker"], linestyle="", markersize=6,
                          markeredgecolor="white", markeredgewidth=0.4, label=st["label"]) for m, st in MODEL.items()]
    leg = fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.04, -0.05),
                     title="Model (all panels; vLLM only)", title_fontsize=7.4, handlelength=2.0, columnspacing=1.4)
    fig.add_artist(leg)
    fig.legend(handles=[plt.Line2D([], [], color=INK2, linewidth=1.5, label="preempted during the request"),
                        plt.Line2D([], [], color=INK2, linewidth=1.1, linestyle=(0, (3, 1.6)),
                                   label="not preempted")],
               loc="upper right", ncol=1, bbox_to_anchor=(0.98, -0.05), title="Line style in (b)",
               title_fontsize=7.4, handlelength=2.0)
    fig.savefig(out / "fig_appendix_a2_preemption.pdf")
    plt.close(fig)
    return pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_appendix"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    d = load(a.rq1_dir)
    a1(d, a.out).to_csv(a.out / "appendix_a1_admission.csv", index=False)
    a2(d, a.out).to_csv(a.out / "appendix_a2_preemption.csv", index=False)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
