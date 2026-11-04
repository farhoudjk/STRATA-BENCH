#!/usr/bin/env python3
"""RQ3: n-gram speculative decoding per engine. PDF figures.

    python scripts/plot_rq3.py --rq3 results/Aggregated_results/rq3_ngram_speculative_decoding.csv \
        --out docs/figures/rq3 [--preview-dir /tmp/previews]

Design (Mistral-NeMo-12B, L40S, client concurrency 256, five arrival processes):
  baseline vs n-gram speculation (k = 3) x max_num_seqs {32, 64} x {mixed_real, chat_multiturn}.
  max_num_seqs is kept low because SGLang caps running requests at 48 under speculation.
  TensorRT-LLM runs speculation with its overlap scheduler disabled (required); its baseline
  keeps the overlap scheduler on, which confounds its result.
Definitions match scripts/plot_flowc.py:
  TTFT    first token - arrival (mixed_real, open loop), first token - send (chat_multiturn).
  TPOT    (E2E - TTFT) / (tokens - 1): per-token speed, robust to multi-token chunks.
  ITL p50 per-request median gap between streamed chunks (shown only to expose the trap).
  Goodput % of all requests finishing with TTFT <= 2 s and TPOT <= 200 ms (strict).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

ENG = {
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^"),
}
BASE, SPEC = "baseline_full_fp16_tp1", "spec_ngram3_fp16_tp1"
WL = {"mixed_real": "mixed_real", "chat_multiturn": "chat_multiturn"}
TASKS = ["reasoning", "code", "chat", "summarization"]
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
RPS = {"low_poisson": 2, "med_poisson": 4, "high_poisson": 8, "vhigh_poisson": 16}
STRICT, RELAXED = (2.0, 200.0), (4.0, 500.0)
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


class Saver:
    def __init__(self, out: Path, preview: Path | None, paper: bool = False):
        self.out, self.preview, self.paper = out, preview, paper

    def __call__(self, fig, name):
        if self.paper:  # captions live in LaTeX: drop figure and panel-level context titles
            if fig._suptitle is not None:
                fig._suptitle.set_visible(False)
            for ax in fig.axes:
                t = ax.get_title(loc="left")
                if "\n(" in t:
                    ax.set_title(t.split("\n")[0], loc="left")
        fig.savefig(self.out / f"{name}.pdf")
        if self.preview:
            fig.savefig(self.preview / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {name}.pdf")


def eng_handles(line=False):
    return [plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4,
                       linestyle="-" if line else "", label=v["label"]) for v in ENG.values()]


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, usecols=lambda c: c != "response_text", low_memory=False)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[d.send_time > 0].copy()
    k = ["engine", "workload_name", "strategy_name", "max_num_seqs", "rate_label"]
    so = d.send_time - d.groupby(k).send_time.transform("min")
    ao = d.arrival_time - d.groupby(k).arrival_time.transform("min")
    d["client_wait"] = np.where(d.workload_name == "mixed_real", (so - ao).clip(lower=0), 0.0)
    d["finished"] = (d.status == "ok") & (d.output_tokens > 1)
    ttft = np.where((d.status == "timeout") & (d.ttft_s <= 0), 300.0, d.ttft_s)
    d["ttft_user"] = ttft + d.client_wait
    d["tpot_ms"] = np.where(d.finished, (d.e2e_s - d.ttft_s) / (d.output_tokens - 1).clip(lower=1) * 1000, np.nan)
    d["tok_per_chunk"] = np.where(d.finished & (d.itl_mean_ms > 0),
                                  (d.output_tokens - 1) / ((d.e2e_s - d.ttft_s) * 1000 / d.itl_mean_ms), np.nan)
    return d


def summarise(g: pd.DataFrame) -> pd.Series:
    f = g[g.finished]
    t0, t1 = (g.send_time - g.client_wait).min(), (f.send_time + f.e2e_s).max()
    gp = lambda a, b: 100 * (g.finished & (g.ttft_user <= a) & (g.tpot_ms <= b)).mean()  # noqa: E731
    return pd.Series(dict(ttft_p50=g.ttft_user.median(), ttft_p95=g.ttft_user.quantile(.95),
                          tpot_p50=f.tpot_ms.median(), tpot_p95=f.tpot_ms.quantile(.95),
                          gap_p95_ms=f.itl_max_ms.quantile(.95), itl_p50_ms=f.itl_p50_ms.median(),
                          e2e_p50=f.e2e_s.median(), out_tok_s=f.output_tokens.sum() / (t1 - t0),
                          tok_per_chunk=f.tok_per_chunk.median(),
                          goodput_strict=gp(*STRICT), goodput_relaxed=gp(*RELAXED)))


# ── F1: payoff matrix ────────────────────────────────────────────────────────
METRICS = [("ttft_p50", "TTFT p50", "pct"), ("tpot_p50", "TPOT p50", "pct"), ("tpot_p95", "TPOT p95", "pct"),
           ("gap_p95_ms", "Longest gap p95", "pct"), ("e2e_p50", "E2E p50", "pct"),
           ("out_tok_s", "Throughput", "pct_inv"), ("goodput_strict", "Goodput (pp)", "pp")]


def f1_payoff(s, save):
    cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#6250d6", "#b9b1ee", "#f2f1ee", "#f2a7a6", "#e34948"])
    norm = mcolors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    rows, labels, recs = [], [], []
    for w in WL:
        for q in (32, 64):
            for e, st in ENG.items():
                b = s.loc[(w, q, e, BASE)]; x = s.loc[(w, q, e, SPEC)]
                vals, colour = [], []
                for m, _, kind in METRICS:
                    if kind == "pp":
                        v = x[m] - b[m]; c = -v
                    else:
                        v = 100 * (x[m] - b[m]) / b[m]; c = -v if kind == "pct_inv" else v
                    vals.append(v); colour.append(c)
                    recs.append(dict(workload=w, max_num_seqs=q, engine=e, metric=m, base=b[m], spec=x[m], change=v))
                rows.append((vals, colour)); labels.append(f"{w} · seqs {q} · {st['label']}")
    V = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.imshow(C, cmap=cmap, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(V):
        unit = " pp" if METRICS[c][2] == "pp" else "%"
        ax.text(c, r, f"{v:+.0f}{unit}", ha="center", va="center", fontsize=5.8,
                color="white" if abs(C[r, c]) >= 40 else INK)
    ax.set_xticks(range(len(METRICS)), [m[1] for m in METRICS], fontsize=6.2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for k in (3, 6, 9):
        ax.axhline(k - 0.5, color="white", linewidth=2 if k != 6 else 3.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=50)
    cb.set_label("Colour: purple = better, red = worse (throughput and goodput: higher is better)", fontsize=6.5)
    cb.outline.set_visible(False)
    fig.suptitle("RQ3 · What n-gram speculative decoding (k = 3) changes, per engine "
                 "(Mistral-NeMo-12B, conc. 256, L40S; vs. no speculation)", x=0.02, ha="left", y=1.02, fontsize=7.5)
    save(fig, "fig_rq3_1_payoff_matrix")
    return pd.DataFrame(recs)


# ── F2: where it pays (task type) ────────────────────────────────────────────
def f2_tasks(d, save):
    m = d[(d.workload_name == "mixed_real") & (d.max_num_seqs == 64) & d.finished]
    fig, ax = plt.subplots(figsize=(3.33, 2.3))
    recs = []
    for i, task in enumerate(TASKS):
        ax.axhspan(i - 0.42, i + 0.42, color="#f4f3f0" if i % 2 == 0 else "white", linewidth=0, zorder=0)
        for j, (e, st) in enumerate(ENG.items()):
            g = m[(m.task_type == task) & (m.engine == e)]
            b, x = g[g.strategy_name == BASE].tpot_ms.median(), g[g.strategy_name == SPEC].tpot_ms.median()
            v = 100 * (x - b) / b
            recs.append(dict(task=task, engine=e, tpot_base=b, tpot_spec=x, change=v))
            y = i + (j - 1) * 0.24
            ax.plot([0, v], [y, y], color=st["color"], linewidth=1.2, zorder=2)
            ax.scatter(v, y, s=20, color=st["color"], marker=st["marker"], edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=5.6, color=INK2)
    out = m[m.strategy_name == BASE].groupby("task_type").output_tokens.median()
    ax.set_yticks(range(len(TASKS)), [f"{t}\n({out[t]:.0f} tok)" for t in TASKS], fontsize=6)
    ax.invert_yaxis(); ax.axvline(0, color=INK, linewidth=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlim(-48, 40)
    ax.set_xlabel("TPOT p50 change with n-gram (%)   ←  faster   |   slower  →")
    ax.legend(handles=eng_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6)
    ax.set_title("Where it pays: by task type\n(mixed_real, max_num_seqs 64; median output tokens)", loc="left")
    save(fig, "fig_rq3_2_by_task")
    return pd.DataFrame(recs)


# ── F3: when it pays (load) ──────────────────────────────────────────────────
def f3_load(d, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    recs = []
    for ax, w in zip(axes, WL):
        for e, st in ENG.items():
            for q, ls in ((64, "-"), (32, (0, (3, 1.5)))):
                g = d[(d.workload_name == w) & (d.engine == e) & (d.max_num_seqs == q) & d.finished
                      & d.rate_label.isin(RATES)]
                p = g.groupby(["rate_label", "strategy_name"]).tpot_ms.median().unstack().reindex(RATES)
                ch = 100 * (p[SPEC] - p[BASE]) / p[BASE]
                ax.plot([RPS[r] for r in RATES], ch.values, color=st["color"], linestyle=ls,
                        marker=st["marker"] if q == 64 else None, markersize=4, markeredgecolor="white",
                        markeredgewidth=0.5)
                recs += [dict(workload=w, engine=e, max_num_seqs=q, rps=RPS[r], tpot_change=v) for r, v in ch.items()]
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(WL[w], loc="left"); ax.set_xlabel("Arrival rate (req/s, Poisson)")
    axes[0].set_ylabel("TPOT p50 change with n-gram (%)\n←  faster   |   slower  →")
    handles = eng_handles(True) + [plt.Line2D([], [], color=INK2, label="max_num_seqs 64"),
                                   plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="32")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("RQ3 · When it pays: the gain shrinks or reverses as load grows (TensorRT-LLM: overlap "
                 "scheduler off under n-gram)", x=0.06, ha="left", y=1.05, fontsize=7.5)
    save(fig, "fig_rq3_3_by_load")
    return pd.DataFrame(recs)


# ── F4: the measurement trap ─────────────────────────────────────────────────
def f4_trap(s, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.3))
    ax = axes[0]
    names = [("itl_p50_ms", "Gap between chunks (ITL p50)"), ("tpot_p50", "Time per output token (TPOT p50)")]
    recs = []
    x = np.arange(len(ENG))
    for k, (m, lab) in enumerate(names):
        vals = [100 * (s.loc[("mixed_real", 64, e, SPEC)][m] - s.loc[("mixed_real", 64, e, BASE)][m])
                / s.loc[("mixed_real", 64, e, BASE)][m] for e in ENG]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34,
               color=[ENG[e]["color"] for e in ENG], alpha=1.0 if k == 1 else 0.35,
               edgecolor=[ENG[e]["color"] for e in ENG], linewidth=0.8)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + (2 if v >= 0 else -2), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6, color=INK)
        recs += [dict(engine=e, metric=m, change=v) for e, v in zip(ENG, vals)]
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, [v["label"] for v in ENG.values()]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Change with n-gram (%)"); ax.set_ylim(-30, 75)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.35, label=names[0][1]),
                       plt.Rectangle((0, 0), 1, 1, facecolor=INK2, label=names[1][1])],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6)
    ax.set_title("Same runs, opposite verdicts (mixed_real, max_num_seqs 64)", loc="left")
    bx = axes[1]
    tp = [[s.loc[("mixed_real", 64, e, st_)].tok_per_chunk for st_ in (BASE, SPEC)] for e in ENG]
    for i, (e, st) in enumerate(ENG.items()):
        bx.plot([0, 1], tp[i], color=st["color"], marker=st["marker"], markersize=4.5, markeredgecolor="white")
        bx.text(1.06, tp[i][1], f"{tp[i][1]:.2f}", va="center", fontsize=6, color=INK2)
    bx.set_xticks([0, 1], ["no speculation", "n-gram"]); bx.set_xlim(-0.2, 1.3)
    bx.grid(axis="x", visible=False); bx.set_ylabel("Tokens per streamed chunk (median)")
    bx.set_title("Why: speculation streams\nseveral tokens per chunk", loc="left")
    fig.suptitle("RQ3 · Measurement trap: per-chunk gaps make speculation look slower than it is",
                 x=0.06, ha="left", y=1.14, fontsize=7.5)
    save(fig, "fig_rq3_4_measurement_trap")
    return pd.DataFrame(recs)


# ── S1: goodput (supplementary) ──────────────────────────────────────────────
def s1_goodput(s, save):
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    cells = [(w, q) for w in WL for q in (32, 64)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw=dict(wspace=0.05))
    for k, (ax, (col, lab)) in enumerate(zip(axes, [("goodput_strict", "strict: TTFT ≤ 2 s, TPOT ≤ 200 ms"),
                                                    ("goodput_relaxed", "relaxed: TTFT ≤ 4 s, TPOT ≤ 500 ms")])):
        M = np.array([[s.loc[(w, q, e, st_)][col] for e in ENG for st_ in (BASE, SPEC)] for w, q in cells])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if v >= 55 else INK)
        ax.set_xticks(range(6), ["base", "n-gram"] * 3, fontsize=5.8)
        for j, e in enumerate(ENG):
            ax.text(2 * j + 0.5, len(cells) - 0.5 + 0.95, ENG[e]["label"], ha="center", va="top", fontsize=6.2)
        ax.set_yticks(range(len(cells)) if k == 0 else [],
                      [f"{w} · seqs {q}" for w, q in cells] if k == 0 else [], fontsize=6)
        for c in (1.5, 3.5):
            ax.axvline(c, color="white", linewidth=2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(lab, loc="left")
    fig.suptitle("RQ3 (supplementary) · SLO goodput with and without n-gram speculation (% of requests)",
                 x=0.06, ha="left", y=1.1, fontsize=7.5)
    save(fig, "fig_rq3_s1_goodput")


# ── S2: TTFT under load (supplementary) ──────────────────────────────────────
def s2_ttft(d, save):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    g0 = d[(d.workload_name == "mixed_real") & (d.max_num_seqs == 64) & d.rate_label.isin(RATES)]
    for ax, (e, st) in zip(axes, ENG.items()):
        for strat, ls, mk in ((BASE, (0, (3, 1.5)), None), (SPEC, "-", st["marker"])):
            p = g0[(g0.engine == e) & (g0.strategy_name == strat)].groupby("rate_label").ttft_user.quantile(.95).reindex(RATES)
            ax.plot([RPS[r] for r in RATES], p.values, color=st["color"], linestyle=ls, marker=mk, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(st["label"], loc="left"); ax.set_xlabel("Arrival rate (req/s)")
    axes[0].set_ylabel("TTFT p95 (s)")
    fig.legend(handles=[plt.Line2D([], [], color=INK2, label="n-gram"),
                        plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="no speculation")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("RQ3 (supplementary) · TTFT p95 under load with and without n-gram (mixed_real, max_num_seqs 64)",
                 x=0.06, ha="left", y=1.06, fontsize=7.5)
    save(fig, "fig_rq3_s2_ttft_load")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq3", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq3"))
    ap.add_argument("--preview-dir", type=Path)
    ap.add_argument("--paper", action="store_true", help="drop in-figure titles (captions go in LaTeX)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.preview_dir:
        a.preview_dir.mkdir(parents=True, exist_ok=True)
    style()
    save = Saver(a.out, a.preview_dir, a.paper)
    d = load(a.rq3)
    s = d.groupby(["workload_name", "max_num_seqs", "engine", "strategy_name"]).apply(summarise)
    s.reset_index().to_csv(a.out / "rq3_summary.csv", index=False)
    f1_payoff(s, save).to_csv(a.out / "rq3_1_payoff_matrix.csv", index=False)
    f2_tasks(d, save).to_csv(a.out / "rq3_2_by_task.csv", index=False)
    f3_load(d, save).to_csv(a.out / "rq3_3_by_load.csv", index=False)
    f4_trap(s, save).to_csv(a.out / "rq3_4_measurement_trap.csv", index=False)
    s1_goodput(s, save)
    s2_ttft(d, save)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
