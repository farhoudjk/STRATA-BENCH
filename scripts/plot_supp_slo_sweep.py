#!/usr/bin/env python3
"""Supplementary: goodput vs. SLO target, with 95% bootstrap bands (L40S, RQ1, C = S = 256).

    python scripts/plot_supp_slo_sweep.py --rq1-dir results/Aggregated_results --out docs/figures/paper_supp

Goodput (share of ALL requests that finish within both targets) is recomputed from the per-request
measurements for paired TTFT/TPOT targets from 0.5 s / 50 ms to 8 s / 1000 ms. Bands: 2.5-97.5
percentiles of 1,000 bootstrap resamples of requests, drawn within each arrival process. They capture
sampling uncertainty within a run, not run-to-run variability.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

TARGETS = [(0.5, 50), (1, 100), (2, 200), (4, 500), (8, 1000)]
B = 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = F.load(a.rq1_dir, a.cache or a.rq1_dir / "rq1_flowc.parquet")
    d = d[d.max_num_seqs == 256]
    rng = np.random.default_rng(7)
    F.style(1.15)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.25, wspace=0.08))
    rows = []
    for i, (w, wl) in enumerate((("mixed_real", "mixed_real"), ("chat_multiturn", "multi-turn"))):
        for j, (m, ml) in enumerate(F.MODEL.items()):
            ax = axes[i, j]
            for e, st in F.ENG.items():
                g = d[(d.workload_name == w) & (d.model == m) & (d.engine == e)]
                fin = g.finished.to_numpy(bool); t = g.ttft_user.to_numpy(float)
                p = np.nan_to_num(g.tpot_ms.to_numpy(float), nan=1e12)
                strata = g.rate_label.to_numpy()
                idx = np.concatenate([rng.choice(np.flatnonzero(strata == s), (B, (strata == s).sum()))
                                      for s in np.unique(strata)], axis=1)
                pt, lo, hi = [], [], []
                for T, P in TARGETS:
                    ok = fin & (t <= T) & (p <= P)
                    pt.append(100 * ok.mean())
                    bs = 100 * ok[idx].mean(axis=1)
                    lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
                    rows.append(dict(workload=w, model=m, engine=e, ttft_s=T, tpot_ms=P, goodput=pt[-1],
                                     lo=lo[-1], hi=hi[-1]))
                x = np.arange(len(TARGETS))
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, linewidth=0)
                ax.plot(x, pt, color=st["color"], marker=st["marker"], markersize=3.2, linewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.4)
            ax.axvline(2, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.6)
            ax.set_ylim(-3, 103); ax.grid(axis="x", visible=False)
            ax.set_title(f"{ml} · {wl}", loc="center", fontsize=F.fz(6.0), pad=3)
            if j == 0:
                ax.set_ylabel("Goodput (%)")
    for ax in axes[-1]:
        ax.set_xticks(range(len(TARGETS)), [f"{T:g} s\n{P:g} ms" for T, P in TARGETS], fontsize=F.fz(5.2))
    fig.supxlabel("SLO target (TTFT / TPOT); dashed: strict SLO of the paper", y=-0.04, fontsize=F.fz(6.4))
    fig.legend(handles=F.eng_handles(True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(a.out / "fig_supp_slo_sweep.pdf")
    plt.close(fig)
    r = pd.DataFrame(rows); r.to_csv(a.out / "supp_slo_sweep.csv", index=False)
    print(r.pivot_table(index=["workload", "model", "engine"], columns="ttft_s", values="goodput").round(1).to_string())


if __name__ == "__main__":
    main()
