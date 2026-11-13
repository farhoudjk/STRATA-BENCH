#!/usr/bin/env python3
"""RQ1, Flow C ("which engine should I deploy?"): engine comparison across all metrics.

    python scripts/plot_flowc.py --rq1-dir results/Aggregated_results --out docs/figures/flowc

Definitions (shared by every figure):
  TTFT (user)   mixed_real (open loop): first token - arrival, i.e. TTFT + time queued in
                the client (the harness starts TTFT at send). chat_multiturn: TTFT from send,
                because later turns are deliberately held until the previous turn finishes.
  TPOT          (E2E - TTFT) / (output_tokens - 1), finished requests with > 1 token.
  TBT           gap between streamed chunks (the harness's itl_* columns), per request.
  E2E           finished: e2e_s. Censored, not dropped:
                  first-token timeout (no token in 300 s)   -> TTFT = 300 s, E2E = 300 s (lower bound)
                  total timeout (no finish in 600 s)        -> E2E = 600 s
                Zero-token replies (status "error") are a model/prompt failure: excluded from
                latency, counted as misses in goodput.
  Goodput       share of ALL requests that finish with TTFT(user) <= a and TPOT <= b.
                Tiers: strict 2 s / 200 ms, relaxed 4 s / 500 ms; supplementary tight-TPOT
                4 s / 50 ms.
Main figures use concurrency 256, max_num_seqs 256 unless stated.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

ENG = {
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^"),
}
MODEL = {"llama31_8b": "Llama-3.1-8B", "mistral_nemo_12b": "Mistral-NeMo-12B", "qwen15_14b": "Qwen1.5-14B"}
WL = {"mixed_real": "mixed_real", "chat_multiturn": "chat_multiturn"}
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
RPS = {"low_poisson": 2, "med_poisson": 4, "high_poisson": 8, "vhigh_poisson": 16}
TIERS = {"strict": (2.0, 200.0), "relaxed": (4.0, 500.0)}
TIGHT = (4.0, 50.0)
TTFT_CAP, E2E_CAP = 300.0, 600.0
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLS = ["engine", "model", "workload_name", "task_type", "max_concurrent", "max_num_seqs", "rate_label", "status",
        "error", "arrival_time", "send_time", "ttft_s", "e2e_s", "output_tokens", "itl_p50_ms", "itl_p95_ms",
        "itl_max_ms"]


FS = 1.0  # font scale; the paper versions set it per figure (see PAPER_FS)


def fz(x: float) -> float:
    return x * FS


def style(fs: float = 1.0):
    global FS
    FS = fs
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7 * fs, "axes.titlesize": 7.5 * fs, "axes.labelsize": 7 * fs,
        "xtick.labelsize": 6.5 * fs, "ytick.labelsize": 6.5 * fs, "legend.fontsize": 6.5 * fs, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# Paper font scale per figure: C2 and C4 are set at column width, the others at text width.
PAPER_FS = {"c2": 1.12, "c3": 1.22, "c4": 1.22, "c56": 1.22, "c6": 1.12, "c8": 1.12}
PAPER = False  # set by --paper: drop in-figure titles that duplicate LaTeX captions


def save(fig, out, name):
    if PAPER:
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        for ax in fig.axes:
            t = ax.get_title(loc="left")
            if "conc." in t:            # context line: belongs in the caption
                ax.set_title("", loc="left")
            elif t:
                ax._left_title.set_text(re.sub(r"^[CS]\d+ · ", "", t))  # keeps the title's font size
    for ext in ("pdf", "png"):
        if PAPER and ext == "png":
            continue
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}")


def plain_log(axis):
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def eng_handles(line=False):
    return [plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4,
                       linestyle="-" if line else "", label=v["label"]) for v in ENG.values()]


# ── data ─────────────────────────────────────────────────────────────────────
def load(rq1_dir: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    fr = []
    for m in MODEL:
        x = pd.read_csv(rq1_dir / f"rq1_{m}.csv", usecols=lambda c: c in COLS, low_memory=False)
        x["model"] = m
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[d.max_concurrent == 256].copy()
    k = ["model", "engine", "workload_name", "max_num_seqs", "rate_label"]
    send_off = d.send_time - d.groupby(k).send_time.transform("min")
    arr_off = d.arrival_time - d.groupby(k).arrival_time.transform("min")
    d["client_wait"] = np.where(d.workload_name == "mixed_real", (send_off - arr_off).clip(lower=0), 0.0)

    err = d.error.fillna("")
    d["ttft_timeout"] = (d.status == "timeout") & err.str.startswith("ttft_timeout")
    d["e2e_timeout"] = (d.status == "timeout") & ~d.ttft_timeout
    d["finished"] = (d.status == "ok") & (d.output_tokens > 0)
    d["latency_valid"] = d.finished | (d.status == "timeout")  # zero-token errors excluded

    ttft = np.where(d.ttft_timeout, TTFT_CAP, d.ttft_s)
    d["ttft_user"] = ttft + d.client_wait
    e2e = np.select([d.ttft_timeout, d.e2e_timeout], [TTFT_CAP, E2E_CAP], d.e2e_s)
    d["e2e_user"] = e2e + d.client_wait
    d["tpot_ms"] = np.where(d.finished & (d.output_tokens > 1),
                            (d.e2e_s - d.ttft_s) / (d.output_tokens - 1).clip(lower=1) * 1000, np.nan)
    d.to_parquet(cache, index=False)
    return d


def goodput(d, ttft_s, tpot_ms):
    meet = d.finished & (d.ttft_user <= ttft_s) & (d.tpot_ms <= tpot_ms)
    return 100 * meet.mean()


# ── C2: latency vs throughput ────────────────────────────────────────────────
def c2(d, out):
    s = d[(d.workload_name == "mixed_real") & (d.max_num_seqs == 256) & d.rate_label.isin(RATES)]
    fig, axes = plt.subplots(3, 3, figsize=(3.45, 3.7), sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.16, wspace=0.12))
    rows = []
    for j, m in enumerate(MODEL):
        for e, st in ENG.items():
            pts = []
            for r in RATES:
                g = s[(s.model == m) & (s.engine == e) & (s.rate_label == r)]
                if g.empty:
                    continue
                f = g[g.finished]
                t0 = (g.send_time - g.client_wait).min()
                t1 = (f.send_time + f.e2e_s).max()
                v = g[g.latency_valid]
                pts.append(dict(model=m, engine=e, rps=RPS[r], ttft_p95=v.ttft_user.quantile(.95),
                                tpot_p95=f.tpot_ms.quantile(.95), out_tok_s=f.output_tokens.sum() / (t1 - t0)))
            rows += pts
            p = pd.DataFrame(pts)
            for i, col in enumerate(["ttft_p95", "tpot_p95", "out_tok_s"]):
                axes[i, j].plot(p.rps, p[col], color=st["color"], marker=st["marker"], markersize=3.2,
                                linewidth=1.1, markeredgecolor="white", markeredgewidth=0.4)
        axes[0, j].set_title(MODEL[m], loc="center", pad=3, fontsize=fz(6.4))
        axes[0, j].axhline(2, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
        axes[0, j].axhline(4, color=INK2, linestyle=(0, (1, 1.5)), linewidth=0.7)
        axes[1, j].axhline(200, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
        axes[2, j].set_xscale("log", base=2); axes[2, j].set_xticks([2, 4, 8, 16]); plain_log(axes[2, j].xaxis)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylim(0.1, 900); plain_log(axes[0, 0].yaxis)
    axes[0, 0].set_yticks([0.1, 1, 10, 100])
    axes[1, 0].set_ylim(0, 300)
    axes[2, 0].set_ylim(0, None)
    fig.supxlabel("Arrival rate (req/s, Poisson)", y=0.035, fontsize=fz(7))
    axes[0, 0].set_ylabel("TTFT p95 (s)")
    axes[1, 0].set_ylabel("TPOT p95 (ms)")
    axes[2, 0].set_ylabel("Output tok/s")
    for ax_, yv, lab in ((axes[0, 2], 2, "2 s"), (axes[0, 2], 4.6, "4 s"), (axes[1, 2], 200, "200 ms")):
        ax_.text(1.03, yv, lab, transform=ax_.get_yaxis_transform(), fontsize=fz(5.8), color=INK2, va="center")
    fig.legend(handles=eng_handles(True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.6, columnspacing=1.0)
    fig.suptitle("C2 · Latency and throughput as load rises (mixed_real, conc. 256, max_num_seqs 256, L40S)",
                 x=0.06, ha="left", y=0.98, fontsize=fz(7.5))
    save(fig, out, "fig_c2_load_response")
    return pd.DataFrame(rows)


# ── C3: streaming quality ────────────────────────────────────────────────────
def ecdf(ax, x, marks=None, mark_offset=0.0, **kw):
    """Empirical CDF. With `marks` (a marker style), draws markers at fixed quantiles so that lines
    can be told apart without colour; `mark_offset` staggers the quantiles between series."""
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    if len(x):
        if marks:
            q = np.clip(np.array([0.15, 0.35, 0.55, 0.75, 0.92]) + mark_offset, 0.01, 0.99)
            kw.update(marker=marks, markevery=sorted(set((q * (len(x) - 1)).astype(int))), markersize=3.2,
                      markeredgecolor="white", markeredgewidth=0.4)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


def c3(d, out):
    s = d[(d.workload_name == "mixed_real") & (d.max_num_seqs == 256) & d.finished & (d.output_tokens > 20)]
    fig, axs = plt.subplots(1, 7, figsize=(7.0, 1.65), sharey=True,
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 0.2, 1, 1, 1], wspace=0.16))
    axs[3].set_visible(False)
    top, bot = axs[:3], axs[4:]
    rows = []
    for j, m in enumerate(MODEL):
        for e, st in ENG.items():
            g = s[(s.model == m) & (s.engine == e)]
            off = {"vllm": -0.04, "sglang": 0.0, "tensorrt-llm": 0.04}[e]
            ecdf(top[j], g.tpot_ms, marks=st["marker"], mark_offset=off, color=st["color"], linewidth=1.3)
            ecdf(bot[j], g.itl_max_ms / 1000, marks=st["marker"], mark_offset=off, color=st["color"], linewidth=1.3)
            rows.append(dict(model=m, engine=e, tpot_p50=g.tpot_ms.median(), tpot_p95=g.tpot_ms.quantile(.95),
                             tbt_p95_p50=g.itl_p95_ms.median(), tbt_max_p95_s=g.itl_max_ms.quantile(.95) / 1000,
                             share_stall_over_1s=(g.itl_max_ms > 1000).mean()))
        top[j].set_title(MODEL[m], loc="center", pad=3, fontsize=fz(5.7))
        bot[j].set_title(MODEL[m], loc="center", pad=3, fontsize=fz(5.7))
        top[j].set_xlim(20, 160); top[j].set_xticks([50, 100, 150]); top[j].set_xlabel("TPOT per request (ms)" if j == 1 else "")
        bot[j].set_xscale("log"); plain_log(bot[j].xaxis); bot[j].set_xticks([0.1, 1, 10])
        bot[j].set_xlabel("Longest gap between tokens (s)" if j == 1 else "")
        bot[j].axvline(1, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
    axs[0].set_ylabel("Share of requests"); axs[0].set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    fig.legend(handles=eng_handles(True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1),
               handlelength=1.6)
    fig.suptitle("C3 · Streaming quality once a response starts (mixed_real, conc. 256, max_num_seqs 256, finished requests)",
                 x=0.06, ha="left", y=0.99, fontsize=fz(7.5))
    save(fig, out, "fig_c3_streaming_quality")
    return pd.DataFrame(rows)


# ── C4: end-to-end ───────────────────────────────────────────────────────────
def c4(d, out):
    s = d[(d.max_num_seqs == 256) & d.latency_valid]
    cells = [(w, m) for w in WL for m in MODEL]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    rows = []
    for i, (w, m) in enumerate(cells):
        ax.axhspan(i - 0.45, i + 0.45, color="#f4f3f0" if i % 2 == 0 else "white", linewidth=0, zorder=0)
        for j, (e, st) in enumerate(ENG.items()):
            g = s[(s.workload_name == w) & (s.model == m) & (s.engine == e)]
            if g.empty:
                continue
            p50, p99 = g.e2e_user.median(), g.e2e_user.quantile(.99)
            rows.append(dict(workload=w, model=m, engine=e, e2e_p50=p50, e2e_p99=p99,
                             censored_share=(g.status == "timeout").mean()))
            y = i + (j - 1) * 0.24
            ax.plot([p50, p99], [y, y], color=st["color"], linewidth=1.1, zorder=2)
            ax.scatter(p50, y, s=18, color=st["color"], marker=st["marker"], zorder=3, edgecolor="white", linewidth=0.4)
            ax.scatter(p99, y, s=18, facecolor="white", edgecolor=st["color"], marker=st["marker"], zorder=3, linewidth=0.8)
    ax.set_xscale("log"); plain_log(ax.xaxis)
    ax.set_yticks(range(len(cells)), [f"{MODEL[m]}\n{w}" for w, m in cells], fontsize=fz(6.2))
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("End-to-end latency from arrival (s)")
    ax.legend(handles=eng_handles() + [
        plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=4, label="p50"),
        plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="white", linestyle="", markersize=4, label="p99")],
        loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=fz(6.2), columnspacing=0.8,
        handletextpad=0.2)
    ax.set_title("C4 · End-to-end latency (conc. 256, max_num_seqs 256)\ntimeouts counted at 300/600 s", loc="left")
    save(fig, out, "fig_c4_e2e")
    return pd.DataFrame(rows)


# ── C5 + C6: outcomes and SLO goodput ────────────────────────────────────────
def c56(d, out):
    s = d[d.max_num_seqs == 256]
    cells = [(w, m) for w in WL for m in MODEL]
    labels = [f"{MODEL[m]} · {w}" for w, m in cells]
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    recs = []
    # outcomes
    y = 0
    yt = []
    for i, (w, m) in enumerate(cells):
        for j, (e, st) in enumerate(ENG.items()):
            g = s[(s.workload_name == w) & (s.model == m) & (s.engine == e)]
            fin = 100 * g.finished.mean(); zt = 100 * (g.status == "error").mean(); to = 100 * (g.status == "timeout").mean()
            yy = i + (j - 1) * 0.27
            ax0.barh(yy, fin, height=0.25, color=st["color"], linewidth=0)
            ax0.barh(yy, zt, left=fin, height=0.25, color="white", hatch="//////", edgecolor=INK2, linewidth=0.3)
            ax0.barh(yy, to, left=fin + zt, height=0.25, color=INK, linewidth=0)
            recs.append(dict(workload=w, model=m, engine=e, finished=fin, zero_token=zt, timeout=to))
        yt.append(i)
    ax0.set_yticks(yt, labels, fontsize=fz(6)); ax0.invert_yaxis(); ax0.set_xlim(0, 100)
    ax0.grid(axis="y", visible=False); ax0.set_xlabel("Share of requests (%)")
    ax0.set_title("C5 · How requests end", loc="left")
    # goodput heatmaps
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    for k, (tier, (a, b)) in enumerate(TIERS.items()):
        ax = fig.add_subplot(gs[k + 1])
        M = np.array([[goodput(s[(s.workload_name == w) & (s.model == m) & (s.engine == e)], a, b)
                       for e in ENG] for w, m in cells])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=fz(6.5),
                    color="white" if v >= 55 else INK, fontweight="bold" if v == M[r].max() else "normal")
        ax.set_xticks(range(len(ENG)), [v["label"] for v in ENG.values()], fontsize=fz(6), rotation=20, ha="right")
        ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"C6 · {tier.capitalize()} goodput (%)\n≤ {a:g} s TTFT, ≤ {b:g} ms TPOT", loc="left",
                     fontsize=fz(6.6))
        for (w, m), row in zip(cells, M):
            for e, v in zip(ENG, row):
                recs.append(dict(workload=w, model=m, engine=e, tier=tier, goodput=v))
    handles = [mpatches.Patch(color=INK2, label="finished (engine colour)"),
               mpatches.Patch(facecolor="white", edgecolor=INK2, hatch="//////", label="zero-token reply (model)"),
               mpatches.Patch(color=INK, label="timeout (engine)")]
    fig.legend(handles=handles + eng_handles(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=fz(6))
    fig.suptitle("Reliability and SLO goodput (% of all requests; bold = best engine; conc. 256, max_num_seqs 256, L40S)",
                 x=0.06, ha="left", y=1.04, fontsize=fz(7.5))
    save(fig, out, "fig_c5_c6_outcomes_goodput")
    return pd.DataFrame(recs)


# ── C8: where the wait goes: a typical short and long request (timeline) ─────
C8_HATCH = {"vllm": "", "sglang": "/////", "tensorrt-llm": "....."}  # pattern per engine, besides colour
def c8(d, out, model="qwen15_14b"):
    """Median life of a short (code) and a long (reasoning) request on each engine, drawn as a
    timeline from arrival: grey = waiting for the first token, engine colour = generating.
    Under first-come, first-served admission both wait about as long (head-of-line blocking)."""
    s = d[(d.workload_name == "mixed_real") & (d.max_num_seqs == 256) & d.rate_label.isin(RATES) & d.finished
          & (d.output_tokens > 1)].copy()
    s["decode_s"] = s.e2e_s - s.ttft_s
    tasks = [("code", "short (code)"), ("reasoning", "long (reasoning)")]
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    rows, ylab, y = [], [], 0
    for e, st in ENG.items():
        g = s[(s.model == model) & (s.engine == e)]
        for t, tl in tasks:
            x = g[g.task_type == t]
            wait, dec, toks = x.ttft_user.median(), x.decode_s.median(), x.output_tokens.median()
            ax.barh(y, wait, height=0.62, color="#cfcdc7", linewidth=0)
            ax.barh(y, dec, left=wait, height=0.62, color=st["color"], hatch=C8_HATCH[e], edgecolor="white",
                    linewidth=0)
            ax.text(wait + dec + 4, y, f"{wait:.0f} s + {dec:.0f} s", va="center", fontsize=fz(5.8), color=INK)
            rows.append(dict(model=model, engine=e, task=t, wait_s=wait, decode_s=dec, output_tokens=toks))
            ylab.append(f"{st['label']}, {tl}")
            y += 1
        y += 0.5
    ypos = [i + 0.5 * (i // 2) for i in range(len(ylab))]
    ax.set_yticks(ypos, ylab, fontsize=fz(5.8)); ax.invert_yaxis()
    ax.set_xlim(0, 380); ax.set_xlabel("Seconds from arrival")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#cfcdc7", label="waiting for first token")] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=v["color"], hatch=C8_HATCH[k], edgecolor="white",
                      label=f"generating, {v['label']}") for k, v in ENG.items()]
    handles = [handles[0], handles[2], handles[1], handles[3]]  # read row-wise in a two-column legend
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.28), ncol=2, fontsize=fz(5.8),
              handlelength=1.6, handleheight=1.1, columnspacing=1.0)
    fig.suptitle(f"C8 · Where the wait goes: median short and long request, {MODEL[model]} "
                 "(mixed_real, conc. 256, max_num_seqs 256)", x=0.06, ha="left", y=1.02, fontsize=fz(7.5))
    save(fig, out, "fig_c8_wait_by_task")
    return pd.DataFrame(rows)


# ── C6: SLO goodput only (paper version, column width) ──────────────────────
def c6(d, out):
    s = d[d.max_num_seqs == 256]
    cells = [(w, m) for w in WL for m in MODEL]
    short = {"llama31_8b": "Llama", "mistral_nemo_12b": "Mistral", "qwen15_14b": "Qwen1.5"}
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True, gridspec_kw=dict(wspace=0.06))
    recs = []
    for ax, (tier, (a, b)) in zip(axes, TIERS.items()):
        M = np.array([[goodput(s[(s.workload_name == w) & (s.model == m) & (s.engine == e)], a, b)
                       for e in ENG] for w, m in cells])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=fz(6.4),
                    color="white" if v >= 55 else INK,
                    fontweight="bold" if round(v) == round(M[r].max()) else "normal")
        ax.set_xticks(range(len(ENG)), ["vLLM", "SGLang", "TRT-LLM"], fontsize=fz(6))
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(range(len(cells)), [f"{short[m]} · {'mixed' if w == 'mixed_real' else 'multi-turn'}"
                                          for w, m in cells], fontsize=fz(6))
        ax.tick_params(axis="y", length=0)
        ax.axhline(len(MODEL) - 0.5, color="white", linewidth=2.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(f"{tier.capitalize()} SLO\nTTFT ≤ {a:g} s, TPOT ≤ {b:g} ms", fontsize=fz(6.2))
        recs += [dict(workload=w, model=m, engine=e, tier=tier, goodput=v)
                 for (w, m), row in zip(cells, M) for e, v in zip(ENG, row)]
    fig.suptitle("C6 · SLO goodput (% of all requests; bold = best engine; conc. 256, max_num_seqs 256, L40S)",
                 x=0.06, ha="left", y=1.04, fontsize=fz(7.5))
    save(fig, out, "fig_c6_goodput")
    return pd.DataFrame(recs)


# ── C7: max_num_seqs ─────────────────────────────────────────────────────────
def c7(d, out):
    s = d[d.workload_name == "mixed_real"]
    metrics = [("goodput", "Goodput, strict (%)", False), ("ttft_p95", "TTFT p95 from arrival (s)", True),
               ("tpot_p95", "TPOT p95 (ms)", False)]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 4.2), gridspec_kw=dict(hspace=0.35, wspace=0.35))
    rows = []
    for i, m in enumerate(MODEL):
        for e, st in ENG.items():
            vals = {}
            for q in (64, 256):
                g = s[(s.model == m) & (s.engine == e) & (s.max_num_seqs == q)]
                vals[q] = dict(goodput=goodput(g, *TIERS["strict"]), ttft_p95=g[g.latency_valid].ttft_user.quantile(.95),
                               tpot_p95=g[g.finished].tpot_ms.quantile(.95))
                rows.append(dict(model=m, engine=e, max_num_seqs=q, **vals[q]))
            for j, (col, lab, logy) in enumerate(metrics):
                ax = axes[i, j]
                ax.plot(["64", "256"], [vals[64][col], vals[256][col]], color=st["color"], marker=st["marker"],
                        markersize=4, markeredgecolor="white", markeredgewidth=0.5)
        for j, (col, lab, logy) in enumerate(metrics):
            ax = axes[i, j]
            if logy:
                ax.set_yscale("log"); ax.set_ylim(0.1, 900); ax.set_yticks([0.1, 1, 10, 100]); plain_log(ax.yaxis)
            if col == "goodput":
                ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(lab, loc="left")
            if j == 0:
                ax.set_ylabel(MODEL[m])
            if i == 2:
                ax.set_xlabel("max_num_seqs")
    fig.legend(handles=eng_handles(True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("C7 · What raising max_num_seqs from 64 to 256 buys (mixed_real, conc. 256, L40S)",
                 x=0.06, ha="left", y=0.98, fontsize=fz(7.5))
    save(fig, out, "fig_c7_max_num_seqs")
    return pd.DataFrame(rows)


# ── S1: tight-TPOT tier (supplementary) ──────────────────────────────────────
def s1(d, out):
    s = d[d.max_num_seqs == 256]
    cells = [(w, m) for w in WL for m in MODEL]
    tiers = {"strict 2 s / 200 ms": TIERS["strict"], "tight TPOT 4 s / 50 ms": TIGHT, "TPOT only ≤ 50 ms": (1e9, 50.0)}
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.08))
    recs = []
    for k, (ax, (name, (a, b))) in enumerate(zip(axes, tiers.items())):
        M = np.array([[goodput(s[(s.workload_name == w) & (s.model == m) & (s.engine == e)], a, b)
                       for e in ENG] for w, m in cells])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=fz(6.5),
                    color="white" if v >= 55 else INK, fontweight="bold" if v == M[r].max() else "normal")
            recs.append(dict(tier=name, workload=cells[r][0], model=cells[r][1], engine=list(ENG)[c], goodput=v))
        ax.set_xticks(range(len(ENG)), [v["label"] for v in ENG.values()], fontsize=fz(6), rotation=20, ha="right")
        ax.set_yticks(range(len(cells)) if k == 0 else [], [f"{MODEL[m]} · {w}" for w, m in cells] if k == 0 else [],
                      fontsize=fz(6))
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, loc="left")
    fig.suptitle("S1 · A tight TPOT target changes the winner (% of requests meeting the SLO; conc. 256, max_num_seqs 256)",
                 x=0.06, ha="left", y=1.06, fontsize=fz(7.5))
    save(fig, out, "fig_s1_tight_tpot_goodput")
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/flowc"))
    ap.add_argument("--cache", type=Path, help="default: <rq1-dir>/rq1_flowc.parquet")
    ap.add_argument("--paper", action="store_true", help="PDF only, without in-figure titles (captions go in LaTeX)")
    a = ap.parse_args()
    global PAPER
    PAPER = a.paper
    a.out.mkdir(parents=True, exist_ok=True)
    fs = PAPER_FS if a.paper else {}
    d = load(a.rq1_dir, a.cache or a.rq1_dir / "rq1_flowc.parquet")
    for name, fn, csv in ((("c2", c2, "c2_load_response"), ("c3", c3, "c3_streaming_quality"), ("c4", c4, "c4_e2e"),
                           ("c56", c56, "c5_c6_outcomes_goodput"), ("c7", c7, "c7_max_num_seqs"),
                           ("c6", c6, "c6_goodput"), ("c8", c8, "c8_wait_by_task"), ("s1", s1, "s1_tight_tpot_goodput"))):
        style(fs.get(name, 1.0))
        fn(d, a.out).to_csv(a.out / f"{csv}.csv", index=False)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()


# ── shared: goodput column with its own diverging scale (used by the RQ2/RQ3 effect matrices) ──
GP_CMAP = mcolors.LinearSegmentedColormap.from_list("gp", ["#8c510a", "#d8b365", "#f5f3ee", "#5ab4ac", "#01665e"])
GP_NORM = mcolors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)


def overlay_goodput_column(ax, values, col, fontsize):
    """Redraw column `col` of a heatmap in the goodput scale (teal = higher goodput, brown = lower),
    with a white gap before it. `values` are changes in percentage points."""
    v = np.asarray(values, dtype=float)[:, None]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(v, cmap=GP_CMAP, norm=GP_NORM, aspect="auto",
              extent=(col - 0.5, col + 0.5, len(v) - 0.5, -0.5), zorder=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)  # keep the full matrix in view
    ax.axvline(col - 0.5, color="white", linewidth=3.5, zorder=3)
    for r, x in enumerate(v[:, 0]):
        ax.text(col, r, f"{round(x, 1) + 0.0:+.1f}", ha="center", va="center", fontsize=fontsize, zorder=4,
                color="white" if abs(x) >= 10 else INK)


def two_colorbars(fig, axes, cmap, norm, label_main, label_gp, fontsize, pad=0.06, height=0.025):
    """Latency colour bar (left, wide) and goodput colour bar (right, narrow) under the heatmaps."""
    fig.canvas.draw()
    boxes = [a.get_position() for a in np.atleast_1d(axes)]
    x0, x1, y0 = min(b.x0 for b in boxes), max(b.x1 for b in boxes), min(b.y0 for b in boxes)
    w = x1 - x0
    c1 = fig.add_axes([x0, y0 - pad - height, w * 0.6, height])
    c2 = fig.add_axes([x0 + w * 0.7, y0 - pad - height, w * 0.3, height])
    for cax, cm, nm, lab in ((c1, cmap, norm, label_main), (c2, GP_CMAP, GP_NORM, label_gp)):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fontsize); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize * 0.92)
