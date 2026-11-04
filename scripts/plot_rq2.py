#!/usr/bin/env python3
"""RQ2: prefill handling (prefix caching, chunked prefill) per engine. PDF figures.

    python scripts/plot_rq2.py --rq2 results/Aggregated_results/rq2_prefill_handling.csv \
        --out docs/figures/rq2 [--preview-dir /tmp/previews]

Design (Mistral-NeMo-12B, concurrency 256, max_num_seqs 256, 5 arrival processes):
  prefix caching on/off x chunked prefill off/512 tokens, on chat_multiturn and chat_real_long.
Definitions match scripts/plot_flowc.py:
  TTFT    first token - arrival for chat_real_long (open loop; client wait is 0 in RQ2),
          first token - send for chat_multiturn (later turns held by design).
  TPOT    (E2E - TTFT) / (tokens - 1);  longest gap = per-request max inter-chunk gap.
  Goodput % of all sent requests finishing with TTFT <= 2 s and TPOT <= 200 ms (strict).
Prompts longer than the 8,192-token context (1% of chat_real_long, identical on every
engine) are rejected by the client before sending and are excluded.
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
BASE = "baseline_full_fp16_tp1_nopfx"          # both features off
STRAT = {
    "baseline_full_fp16_tp1": "Prefix caching",
    "chunked_512_fp16_tp1_nopfx": "Chunked prefill",
    "chunked_512_fp16_tp1": "Both",
}
WL = {"chat_multiturn": "chat_multiturn (multi-turn)", "chat_real_long": "chat_real_long (long prompts)"}
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
RPS = {"low_poisson": 2, "med_poisson": 4, "high_poisson": 8, "vhigh_poisson": 16}
STRICT = (2.0, 200.0)
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
        "xtick.labelsize": 6.5 * fs, "ytick.labelsize": 6.5 * fs, "legend.fontsize": 6.5 * fs, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# Paper font scale per figure (F4 is set at column width).
PAPER_FS = {"f1": 1.25, "f2": 1.2, "f4": 1.12}


class Saver:
    def __init__(self, out: Path, preview: Path | None, paper: bool = False):
        self.out, self.preview, self.paper = out, preview, paper

    def __call__(self, fig, name):
        if self.paper:  # captions live in LaTeX: drop the figure title
            if fig._suptitle is not None:
                fig._suptitle.set_visible(False)
        fig.savefig(self.out / f"{name}.pdf")
        if self.preview:
            fig.savefig(self.preview / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {name}.pdf")


def plain_log(axis):
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def eng_handles(line=True):
    return [plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4,
                       linestyle="-" if line else "", label=v["label"]) for v in ENG.values()]


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, usecols=lambda c: c != "response_text", low_memory=False)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    d = d[d.send_time > 0].copy()  # over-length prompts rejected before sending
    k = ["engine", "workload_name", "strategy_name", "rate_label"]
    so = d.send_time - d.groupby(k).send_time.transform("min")
    ao = d.arrival_time - d.groupby(k).arrival_time.transform("min")
    cw = np.where(d.workload_name == "chat_real_long", (so - ao).clip(lower=0), 0.0)
    d["finished"] = (d.status == "ok") & (d.output_tokens > 1)
    ttft = np.where((d.status == "timeout") & (d.ttft_s <= 0), 300.0, d.ttft_s)
    d["ttft_user"] = ttft + cw
    d["tpot_ms"] = np.where(d.finished, (d.e2e_s - d.ttft_s) / (d.output_tokens - 1).clip(lower=1) * 1000, np.nan)
    return d


def summary(d: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        f = g[g.finished]
        meet = g.finished & (g.ttft_user <= STRICT[0]) & (g.tpot_ms <= STRICT[1])
        return pd.Series(dict(ttft_p50=g.ttft_user.median(), ttft_p95=g.ttft_user.quantile(.95),
                              tpot_p50=f.tpot_ms.median(), tpot_p95=f.tpot_ms.quantile(.95),
                              gap_p95_ms=f.itl_max_ms.quantile(.95), e2e_p50=f.e2e_s.median(),
                              goodput_strict=100 * meet.mean()))
    return d.groupby(["workload_name", "strategy_name", "engine"]).apply(agg).reset_index()


# ── F1: payoff matrix ────────────────────────────────────────────────────────
METRICS = [("ttft_p95", "TTFT p95"), ("tpot_p95", "TPOT p95"), ("gap_p95_ms", "Longest gap p95"),
           ("e2e_p50", "E2E p50")]


def f1_payoff(s, save):
    cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#6250d6", "#b9b1ee", "#f2f1ee", "#f2a7a6", "#e34948"])
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(wspace=0.06))
    recs = []
    for ax, (w, wlab) in zip(axes, WL.items()):
        rows, labels = [], []
        for strat, slab in STRAT.items():
            for e, st in ENG.items():
                b = s[(s.workload_name == w) & (s.strategy_name == BASE) & (s.engine == e)].iloc[0]
                x = s[(s.workload_name == w) & (s.strategy_name == strat) & (s.engine == e)].iloc[0]
                pct = [100 * (x[m] - b[m]) / b[m] for m, _ in METRICS]
                rows.append(pct); labels.append((slab, st["label"]))
                recs += [dict(workload=w, feature=slab, engine=e, metric=m, base=b[m], value=x[m], pct=p)
                         for (m, _), p in zip(METRICS, pct)]
        M = np.array(rows)
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:+.0f}%", ha="center", va="center", fontsize=fz(6),
                    color="white" if abs(v) >= 55 else INK)
        ax.set_xticks(range(len(METRICS)), [m[1].replace(" ", "\n", 1) for m in METRICS], fontsize=fz(6))
        ax.xaxis.tick_top()
        if ax is axes[0]:
            ax.set_yticks(range(len(labels)), [f"{s_} · {e_}" for s_, e_ in labels], fontsize=fz(6))
        else:
            ax.set_yticks([])
        for k in (3, 6):
            ax.axhline(k - 0.5, color="white", linewidth=2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(wlab, loc="left", pad=28 if FS > 1 else 20, fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("Change vs. both features off (%)   ←  better   |   worse  →", fontsize=fz(6.5))
    cb.outline.set_visible(False)
    fig.suptitle("RQ2 · What each prefill feature buys, per engine (Mistral-NeMo-12B, conc. 256, L40S)",
                 x=0.06, ha="left", y=1.12, fontsize=fz(7.5))
    save(fig, "fig_rq2_1_payoff_matrix")
    return pd.DataFrame(recs)


# ── F2: prefix caching by turn ───────────────────────────────────────────────
def f2_prefix_turns(d, save):
    c = d[(d.workload_name == "chat_multiturn") & d.strategy_name.isin(["baseline_full_fp16_tp1", BASE])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.25))
    recs = []
    for e, st in ENG.items():
        for strat, ls, mk in (("baseline_full_fp16_tp1", "-", st["marker"]), (BASE, (0, (3, 1.5)), None)):
            g = c[(c.engine == e) & (c.strategy_name == strat) & c.finished]
            t = g.groupby("turn_index").agg(ttft=("ttft_user", "median"), tpot=("tpot_ms", "median"),
                                            prompt=("prompt_token_len", "median"), n=("ttft_user", "size"))
            t = t[t.n >= 30]
            axes[0].plot(t.index + 1, t.ttft * 1000, color=st["color"], linestyle=ls, marker=mk, markersize=3.5,
                         markeredgecolor="white", markeredgewidth=0.5)
            axes[1].plot(t.index + 1, t.tpot, color=st["color"], linestyle=ls, marker=mk, markersize=3.5,
                         markeredgecolor="white", markeredgewidth=0.5)
            recs += [dict(engine=e, strategy=strat, turn=int(i) + 1, **r) for i, r in t.iterrows()]
    prompt = pd.DataFrame(recs).groupby("turn").prompt.median()
    for ax, lab in zip(axes, ["TTFT p50 (ms)", "TPOT p50 (ms)"]):
        ax.set_xlabel("Conversation turn"); ax.set_ylabel(lab)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(prompt.index, [f"{p / 1000:.1f}k" if p >= 1000 else f"{p:.0f}" for p in prompt.values])
        sec.tick_params(labelsize=fz(5.8), colors=INK2)
        sec.set_xlabel("median prompt tokens", fontsize=fz(6), color=INK2)
    handles = eng_handles() + [plt.Line2D([], [], color=INK2, label="prefix caching on"),
                               plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="off")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("RQ2 · Prefix caching keeps TTFT flat as the conversation grows (chat_multiturn, chunking off)",
                 x=0.06, ha="left", y=1.08, fontsize=fz(7.5))
    save(fig, "fig_rq2_2_prefix_by_turn")
    return pd.DataFrame(recs)


# ── F3: chunked prefill under load (long prompts) ────────────────────────────
def f3_chunk_load(d, save):
    c = d[(d.workload_name == "chat_real_long") & d.rate_label.isin(RATES)
          & d.strategy_name.isin(["baseline_full_fp16_tp1", "chunked_512_fp16_tp1"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True, gridspec_kw=dict(wspace=0.08))
    recs = []
    for ax, (e, st) in zip(axes, ENG.items()):
        for strat, ls, lab in (("baseline_full_fp16_tp1", (0, (3, 1.5)), "off"), ("chunked_512_fp16_tp1", "-", "512")):
            g = c[(c.engine == e) & (c.strategy_name == strat)]
            p = g.groupby("rate_label").ttft_user.quantile(.95).reindex(RATES)
            ax.plot([RPS[r] for r in RATES], p.values, color=st["color"], linestyle=ls,
                    marker=st["marker"] if lab == "512" else None, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
            recs += [dict(engine=e, chunking=lab, rps=RPS[r], ttft_p95=v) for r, v in p.items()]
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); plain_log(ax.xaxis)
        ax.set_yscale("log"); plain_log(ax.yaxis)
        ax.set_title(st["label"], loc="left"); ax.set_xlabel("Arrival rate (req/s)")
    axes[0].set_ylabel("TTFT p95 (s)")
    handles = [plt.Line2D([], [], color=INK2, label="chunked prefill (512 tokens)"),
               plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="no chunking")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("RQ2 · Chunked prefill under load: only TensorRT-LLM gets faster first tokens "
                 "(chat_real_long, prefix caching on)", x=0.06, ha="left", y=1.06, fontsize=fz(7.5))
    save(fig, "fig_rq2_3_chunked_under_load")
    return pd.DataFrame(recs)


# ── F4: chunked prefill and stalls ───────────────────────────────────────────
def f4_chunk_stalls(d, save):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1))
    recs = []
    for ax, w in zip(axes, WL):
        for e, st in ENG.items():
            for strat, ls in (("baseline_full_fp16_tp1", (0, (3, 1.5))), ("chunked_512_fp16_tp1", "-")):
                g = d[(d.workload_name == w) & (d.engine == e) & (d.strategy_name == strat) & d.finished]
                x = np.sort(g.itl_max_ms.dropna().values / 1000)
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=st["color"], linestyle=ls,
                        linewidth=1.2)
                recs.append(dict(workload=w, engine=e, strategy=strat, gap_p50_s=np.median(x),
                                 gap_p95_s=np.quantile(x, .95), share_over_1s=(x > 1).mean()))
        ax.set_xscale("log"); plain_log(ax.xaxis); ax.set_xticks([0.1, 1, 10])
        ax.axvline(1, color=INK2, linestyle=(0, (1, 1.5)), linewidth=0.7)
        ax.set_title(w, loc="left", fontsize=fz(6.8), pad=3)
        ax.set_xlabel("Longest gap (s)")
    axes[0].set_ylabel("Share of requests"); axes[0].set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    handles = eng_handles() + [plt.Line2D([], [], color=INK2, label="chunked (512)"),
                               plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="no chunking")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08), handlelength=1.6,
               columnspacing=0.9)
    fig.suptitle("RQ2 · Chunked prefill removes prefill-induced stalls on vLLM and TensorRT-LLM, not on SGLang "
                 "(prefix caching on)", x=0.06, ha="left", y=1.06, fontsize=fz(7.5))
    save(fig, "fig_rq2_4_chunked_stalls")
    return pd.DataFrame(recs)


# ── S1: TTFT vs TPOT trade-off arrows (supplementary) ────────────────────────
def s1_tradeoff(s, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.25))
    for ax, (w, wlab) in zip(axes, WL.items()):
        for e, st in ENG.items():
            a = s[(s.workload_name == w) & (s.strategy_name == "baseline_full_fp16_tp1") & (s.engine == e)].iloc[0]
            b = s[(s.workload_name == w) & (s.strategy_name == "chunked_512_fp16_tp1") & (s.engine == e)].iloc[0]
            ax.annotate("", xy=(b.ttft_p95, b.tpot_p95), xytext=(a.ttft_p95, a.tpot_p95),
                        arrowprops=dict(arrowstyle="-|>", color=st["color"], linewidth=1.3, mutation_scale=8))
            ax.scatter(a.ttft_p95, a.tpot_p95, s=18, facecolor="white", edgecolor=st["color"], marker=st["marker"],
                       linewidth=0.9, zorder=3)
            ax.scatter(b.ttft_p95, b.tpot_p95, s=20, color=st["color"], marker=st["marker"], zorder=3,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(wlab, loc="left"); ax.set_xlabel("TTFT p95 (s)"); ax.set_ylabel("TPOT p95 (ms)")
        ax.margins(0.15)
    handles = eng_handles(False) + [plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="white",
                                               linestyle="", label="no chunking"),
                                    plt.Line2D([], [], color=INK2, marker="o", linestyle="", label="chunked (512)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("RQ2 (supplementary) · Where chunked prefill moves each engine (arrow: off → on; down-left is better)",
                 x=0.06, ha="left", y=1.04, fontsize=fz(7.5))
    save(fig, "fig_rq2_s1_chunk_tradeoff")


# ── S2: goodput per strategy (supplementary) ─────────────────────────────────
def s2_goodput(s, save):
    order = [BASE, "baseline_full_fp16_tp1", "chunked_512_fp16_tp1_nopfx", "chunked_512_fp16_tp1"]
    names = ["Neither", "Prefix caching", "Chunked prefill", "Both"]
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.7), gridspec_kw=dict(wspace=0.05))
    for k, (ax, (w, wlab)) in enumerate(zip(axes, WL.items())):
        M = np.array([[s[(s.workload_name == w) & (s.strategy_name == o) & (s.engine == e)].goodput_strict.iloc[0]
                       for e in ENG] for o in order])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=fz(6.5), color="white" if v >= 55 else INK)
        ax.set_xticks(range(len(ENG)), [v["label"] for v in ENG.values()], fontsize=fz(6))
        ax.set_yticks(range(len(order)) if k == 0 else [], names if k == 0 else [], fontsize=fz(6))
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(wlab, loc="left")
    fig.suptitle("RQ2 (supplementary) · Strict goodput (% of requests with TTFT ≤ 2 s and TPOT ≤ 200 ms)",
                 x=0.06, ha="left", y=1.08, fontsize=fz(7.5))
    save(fig, "fig_rq2_s2_goodput")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq2"))
    ap.add_argument("--preview-dir", type=Path, help="also write PNG previews here")
    ap.add_argument("--paper", action="store_true", help="drop in-figure titles (captions go in LaTeX)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.preview_dir:
        a.preview_dir.mkdir(parents=True, exist_ok=True)
    fs = PAPER_FS if a.paper else {}
    save = Saver(a.out, a.preview_dir, a.paper)
    d = load(a.rq2)
    s = summary(d)
    s.to_csv(a.out / "rq2_summary.csv", index=False)
    style(fs.get("f1", 1.0)); f1_payoff(s, save).to_csv(a.out / "rq2_1_payoff_matrix.csv", index=False)
    style(fs.get("f2", 1.0)); f2_prefix_turns(d, save).to_csv(a.out / "rq2_2_prefix_by_turn.csv", index=False)
    style(fs.get("f3", 1.0)); f3_chunk_load(d, save).to_csv(a.out / "rq2_3_chunked_under_load.csv", index=False)
    style(fs.get("f4", 1.0)); f4_chunk_stalls(d, save).to_csv(a.out / "rq2_4_chunked_stalls.csv", index=False)
    style(); s1_tradeoff(s, save)
    s2_goodput(s, save)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
