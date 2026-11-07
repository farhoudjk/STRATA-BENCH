#!/usr/bin/env python3
"""Supplementary (RQ3): why n-gram speculation pays on vLLM but not on SGLang or TensorRT-LLM.

    python scripts/plot_supp_rq3_step_cost.py \
        --rq3 results/Aggregated_results/rq3_ngram_speculative_decoding.csv --out docs/figures/paper_supp

Accepted draft tokens are streamed together with the verified token, so the median number of tokens
per streamed chunk estimates the tokens produced per decoding step (1.00 without speculation). The
time of one decoding step is then TPOT x tokens per chunk, and its change with speculation is the
cost of verifying the drafts. Finished requests with > 20 output tokens, medians per configuration.

(a) tokens per step with n-gram; (b) change in step time; (c) change in median TPOT, the net effect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402
import plot_rq3 as R3  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

GROUPS = [("chat_multiturn", 32), ("chat_multiturn", 64), ("mixed_real", 32), ("mixed_real", 64)]
GLAB = ["multi-turn\nS=32", "multi-turn\nS=64", "mixed_real\nS=32", "mixed_real\nS=64"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq3", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    F.style(1.15)
    d = R3.load(a.rq3)
    f = d[d.finished & (d.output_tokens > 20)]
    g = f.groupby(["workload_name", "max_num_seqs", "engine", "strategy_name"]).agg(
        tpot=("tpot_ms", "median"), tpc=("tok_per_chunk", "median")).unstack("strategy_name")
    rows = []
    for w, q in GROUPS:
        for e in F.ENG:
            r = g.loc[(w, q, e)]
            tb, ts = r[("tpot", R3.BASE)], r[("tpot", R3.SPEC)]
            cb, cs = r[("tpc", R3.BASE)], r[("tpc", R3.SPEC)]
            rows.append(dict(workload=w, max_num_seqs=q, engine=e, tokens_per_step=cs,
                             step_time_change=100 * (ts * cs / (tb * cb) - 1), tpot_change=100 * (ts / tb - 1)))
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "supp_rq3_step_cost.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), gridspec_kw=dict(wspace=0.3))
    x = np.arange(len(GROUPS))
    panels = [("tokens_per_step", "Tokens per step\nwith n-gram", "(a) Drafts accepted"),
              ("step_time_change", "Step time change (%)", "(b) Cost of a step"),
              ("tpot_change", "TPOT change (%)", "(c) Net effect")]
    for ax, (col, ylab, title) in zip(axes, panels):
        for k, e in enumerate(F.ENG):
            v = [t[(t.workload == w) & (t.max_num_seqs == q) & (t.engine == e)][col].iloc[0] for w, q in GROUPS]
            ax.bar(x + (k - 1) * 0.26, v, width=0.24, color=F.ENG[e]["color"], linewidth=0)
        if col == "tokens_per_step":
            ax.axhline(1, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.7); ax.set_ylim(0.9, 1.65)
        else:
            ax.axhline(0, color=F.INK, linewidth=0.8)
        ax.axvline(1.5, color=F.GRID, linewidth=0.8)
        ax.set_xticks(x, ["S=32", "S=64", "S=32", "S=64"], fontsize=F.fz(5.8)); ax.grid(axis="x", visible=False)
        for xc, lab in ((0.5, "multi-turn"), (2.5, "mixed_real")):
            ax.text(xc, -0.22, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=F.fz(6.0))
        ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontsize=F.fz(6.6), pad=3)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v["color"], label=v["label"]) for v in F.ENG.values()],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(a.out / "fig_supp_rq3_step_cost.pdf")
    plt.close(fig)
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
