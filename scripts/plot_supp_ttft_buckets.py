#!/usr/bin/env python3
"""Supplementary: median TTFT vs. arrival-order bucket within each rate configuration (log scale).

    python scripts/plot_supp_ttft_buckets.py --rq1-dir results/Aggregated_results \
        --rq3 results/Aggregated_results/rq3_ngram_speculative_decoding.csv --out docs/figures/paper_supp

Each rate configuration replays a fixed trace. Requests are ordered by arrival (mixed_real, open loop)
or by dispatch (chat_multiturn, whose later turns are held until the previous turn completes) and split
into 15 equal-count buckets; each point is the bucket's median TTFT. TTFT is measured as in the paper:
from arrival for mixed_real (client-side waiting included), from send for chat_multiturn; requests
with no first token within 300 s count at 300 s; zero-token replies are excluded.

RQ1  fig_supp_rq1_ttft_buckets_<model>_<workload>.pdf  rows: max_num_seqs 64 / 256; columns: rates;
     colour: engine.  Concurrency 256, L40S.
RQ3  fig_supp_rq3_ttft_buckets_<workload>.pdf         rows: max_num_seqs 32 / 64; columns: rates;
     colour: engine; solid: no speculation, dashed: n-gram (k = 3).  Mistral-NeMo-12B.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402  (style, engines, models, RQ1 loader)
import plot_rq3 as R3  # noqa: E402  (RQ3 loader)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

RATES = [("low_poisson", "2 req/s"), ("med_poisson", "4 req/s"), ("high_poisson", "8 req/s"),
         ("vhigh_poisson", "16 req/s"), ("bursty", "bursty")]
NB = 15
FS = 1.15
CEIL = 300.0


def bucket_medians(g: pd.DataFrame, order_col: str) -> np.ndarray:
    y = g.sort_values(order_col).ttft_user.to_numpy()
    return np.array([np.median(b) for b in np.array_split(y, NB)]) if len(y) >= NB else np.full(NB, np.nan)


def axis_setup(ax, top):
    ax.set_yscale("log"); ax.set_ylim(0.03, top)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([0, 7, 14], ["1", "8", "15"]); ax.set_xlim(-0.6, NB - 0.4)


def grid(series, seqs, top, ylabel_fmt, xlabel, handles, path):
    """series: dict[(seqs, rate)] -> list of (y, colour, linestyle, marker)."""
    fig, axes = plt.subplots(len(seqs), len(RATES), figsize=(7.0, 1.3 * len(seqs) + 0.45), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.14, wspace=0.07))
    for i, s in enumerate(seqs):
        for j, (rate, rlab) in enumerate(RATES):
            ax = axes[i, j]
            for y, c, ls, mk in series.get((s, rate), []):
                ax.plot(np.arange(NB), y, color=c, linestyle=ls, marker=mk, markersize=2.6, linewidth=1.1,
                        markeredgecolor="white", markeredgewidth=0.3)
            if top > CEIL:
                ax.axhline(CEIL, color=F.INK2, linestyle=(0, (1, 1.5)), linewidth=0.7)
            axis_setup(ax, top)
            if i == 0:
                ax.set_title(rlab, loc="left", pad=3)
            if j == 0:
                ax.set_ylabel(ylabel_fmt.format(s))
    fig.supxlabel(xlabel, y=-0.02, fontsize=F.fz(6.8))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, -0.07), handlelength=1.8)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def ceiling_for(vals):
    m = np.nanmax(vals)
    return 600.0 if m > 150 else max(1.0, m * 1.8)


def rq1(rq1_dir: Path, cache: Path, out: Path) -> pd.DataFrame:
    d = F.load(rq1_dir, cache)
    d = d[d.latency_valid & (d.max_concurrent == 256)]
    recs = []
    for m, mlab in F.MODEL.items():
        for w, order_col, xl in (("mixed_real", "arrival_time", "Arrival-order bucket within the rate configuration"),
                                 ("chat_multiturn", "send_time", "Dispatch-order bucket within the rate configuration")):
            series, vals = {}, []
            for s in (64, 256):
                for rate, _ in RATES:
                    for e, st in F.ENG.items():
                        g = d[(d.model == m) & (d.workload_name == w) & (d.max_num_seqs == s) & (d.rate_label == rate)
                              & (d.engine == e)]
                        y = bucket_medians(g, order_col)
                        series.setdefault((s, rate), []).append((y, st["color"], "-", st["marker"]))
                        vals.append(y)
                        recs += [dict(model=m, workload=w, max_num_seqs=s, rate=rate, engine=e, bucket=b + 1, ttft_p50=v)
                                 for b, v in enumerate(y)]
            grid(series, (64, 256), ceiling_for(np.concatenate(vals)), "$S$ = {}\nTTFT p50 (s)", xl,
                 F.eng_handles(True), out / f"fig_supp_rq1_ttft_buckets_{m}_{w}.pdf")
    return pd.DataFrame(recs)


def rq3(rq3_csv: Path, out: Path) -> pd.DataFrame:
    d = R3.load(rq3_csv)
    d = d[d.status != "error"]
    recs = []
    for w, order_col, xl in (("mixed_real", "arrival_time", "Arrival-order bucket within the rate configuration"),
                             ("chat_multiturn", "send_time", "Dispatch-order bucket within the rate configuration")):
        series, vals = {}, []
        for s in (32, 64):
            for rate, _ in RATES:
                for e, st in F.ENG.items():
                    for strat, ls, lab in ((R3.BASE, "-", "off"), (R3.SPEC, (0, (3, 1.5)), "ngram")):
                        g = d[(d.workload_name == w) & (d.max_num_seqs == s) & (d.rate_label == rate) & (d.engine == e)
                              & (d.strategy_name == strat)]
                        y = bucket_medians(g, order_col)
                        series.setdefault((s, rate), []).append((y, st["color"], ls, st["marker"] if lab == "off" else None))
                        vals.append(y)
                        recs += [dict(workload=w, max_num_seqs=s, rate=rate, engine=e, speculation=lab, bucket=b + 1,
                                      ttft_p50=v) for b, v in enumerate(y)]
        handles = F.eng_handles(True) + [plt.Line2D([], [], color=F.INK2, label="no speculation"),
                                         plt.Line2D([], [], color=F.INK2, linestyle=(0, (3, 1.5)), label="n-gram ($k$ = 3)")]
        grid(series, (32, 64), ceiling_for(np.concatenate(vals)), "$S$ = {}\nTTFT p50 (s)", xl, handles,
             out / f"fig_supp_rq3_ttft_buckets_{w}.pdf")
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="default: <rq1-dir>/rq1_flowc.parquet")
    ap.add_argument("--rq3", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_supp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    F.style(FS)
    rq1(a.rq1_dir, a.cache or a.rq1_dir / "rq1_flowc.parquet", a.out).to_csv(a.out / "supp_rq1_ttft_buckets.csv",
                                                                          index=False)
    rq3(a.rq3, a.out).to_csv(a.out / "supp_rq3_ttft_buckets.csv", index=False)


if __name__ == "__main__":
    main()
