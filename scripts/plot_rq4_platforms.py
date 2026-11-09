#!/usr/bin/env python3
"""RQ4: the RQ1 load-response figure on two further platforms, engines compared within each platform.

    python scripts/plot_rq4_platforms.py --gb10-report rq5_gb10_report.html \
        --mig-dirs ../enginebench/results/rq5_h100_mig ../enginebench/results/rq5_h100_mig_lowrate \
        --out docs/figures/paper_rq4

Platforms are never put on a shared axis: each block is read like Fig. 2 (L40S).
  GB10      per-configuration summaries only (the report): TTFT p95 per configuration, TPOT = 1000 / median
            per-request decode rate, throughput in requests/s.
  H100 MIG  per-request data (EngineBench rq5_h100_mig and the low-rate sweep): definitions as in RQ1.
            TTFT from arrival (client wait included) on mixed_real; requests without a first token within
            the 120 s first-token timeout count at 120 s; TPOT = (E2E - TTFT) / (tokens - 1), p95 over finished
            requests; output throughput = output tokens of finished requests / run span.
mixed_real, C = S = 256, Poisson arrivals.

Also writes per-configuration tables (CSV) for both platforms, used for the RQ4 findings table, and, with
--rq1-dir, the appendix figure: vLLM preemptions vs. peak KV-cache use per run on all three platforms.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_flowc as F  # noqa: E402  (style, engine colours, models)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

RPS = {"r025_poisson": 0.25, "r05_poisson": 0.5, "r1_poisson": 1, "low_poisson": 2, "med_poisson": 4,
       "high_poisson": 8, "vhigh_poisson": 16}
MIG_TTFT_CAP, MIG_E2E_CAP = 120.0, 300.0


# ── data ─────────────────────────────────────────────────────────────────────
def load_gb10(path: Path) -> pd.DataFrame:
    s = path.read_text(encoding="utf-8")
    d = pd.DataFrame(json.loads(re.search(r"const D = (\{.*?\});\n", s, re.S).group(1))["rows"])
    d = d[(d.seqs == 256) & (d.workload == "mixed_real") & d.rate.isin(RPS)]
    return pd.DataFrame(dict(model=d.model, engine=d.engine, rps=d.rate.map(RPS), ttft_p95=d.ttft_p95,
                             tpot=1000 / d.tps, thr=d.rps, finished=d.ok_pct, kv_peak=d.kv_peak, preemptions=d.pre))


def load_mig(dirs: list[Path]) -> pd.DataFrame:
    fr = []
    for root in dirs:
        for d in sorted(glob.glob(str(root / "*__*"))):
            if "partial" in d:
                continue
            m, e, w, _ = os.path.basename(d).split("__")
            r = pd.read_csv(f"{d}/requests.csv", low_memory=False).drop(columns=["response_text"], errors="ignore")
            r["model"], r["engine"] = m, e
            fr.append(r)
    d = pd.concat(fr, ignore_index=True)
    d["engine"] = d.engine.replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    k = ["model", "engine", "workload_name", "max_num_seqs", "rate_label"]
    so = d.send_time - d.groupby(k).send_time.transform("min")
    ao = d.arrival_time - d.groupby(k).arrival_time.transform("min")
    d["client_wait"] = np.where(d.workload_name == "mixed_real", (so - ao).clip(lower=0), 0.0)
    err = d.error.fillna("")
    d["ttft_timeout"] = (d.status == "timeout") & err.str.startswith("ttft_timeout")
    d["e2e_timeout"] = (d.status == "timeout") & ~d.ttft_timeout
    d["finished"] = (d.status == "ok") & (d.output_tokens > 0)
    d["latency_valid"] = d.finished | (d.status == "timeout")
    d["ttft_user"] = np.where(d.ttft_timeout, MIG_TTFT_CAP, d.ttft_s) + d.client_wait
    d["e2e_user"] = np.select([d.ttft_timeout, d.e2e_timeout], [MIG_TTFT_CAP, MIG_E2E_CAP], d.e2e_s) + d.client_wait
    d["tpot_ms"] = np.where(d.finished & (d.output_tokens > 1),
                            (d.e2e_s - d.ttft_s) / (d.output_tokens - 1).clip(lower=1) * 1000, np.nan)
    d["rps"] = d.rate_label.map(RPS)
    return d


def mig_per_config(d: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        f = g[g.finished & (g.output_tokens > 1)]
        v = g[g.latency_valid]
        t0 = (g.send_time - g.client_wait).min()
        t1 = (f.send_time + f.e2e_s).max()
        st = np.concatenate([np.c_[f.send_time + f.ttft_s, np.ones(len(f))], np.c_[f.send_time + f.e2e_s, -np.ones(len(f))]])
        st = st[st[:, 0].argsort()]
        c = np.cumsum(st[:, 1]); dt = np.diff(st[:, 0], append=st[-1, 0]); busy = dt[c > 0].sum()
        meet = g.finished & (g.ttft_user <= 2) & (g.tpot_ms <= 200)
        return pd.Series(dict(ttft_p50=v.ttft_user.median(), ttft_p95=v.ttft_user.quantile(.95),
                              tpot_p50=f.tpot_ms.median(), tpot_p95=f.tpot_ms.quantile(.95),
                              thr=f.output_tokens.sum() / (t1 - t0), streaming=(c * dt).sum() / busy if busy else np.nan,
                              finished=100 * g.finished.mean(), goodput_strict=100 * meet.mean(),
                              e2e_p50=v.e2e_user.median(), zero_token=100 * (g.status == "error").mean(),
                              no_first_token=100 * g.ttft_timeout.mean(), n=len(g)))
    p = d[d.rate_label.isin(RPS)]
    return p.groupby(["workload_name", "max_num_seqs", "model", "engine", "rps"]).apply(agg).reset_index()


# ── figure ───────────────────────────────────────────────────────────────────
def block(axes, df, rows, xticks, ttft_cap, label):
    for j, m in enumerate(F.MODEL):
        for e, st in F.ENG.items():
            g = df[(df.model == m) & (df.engine == e)].sort_values("rps")
            for i, (col, *_r) in enumerate(rows):
                axes[i, j].plot(g.rps, g[col], color=st["color"], marker=st["marker"], markersize=3.0, linewidth=1.05,
                                markeredgecolor="white", markeredgewidth=0.4)
        axes[0, j].set_title(F.MODEL[m], loc="center", fontsize=F.fz(6.0), pad=3)
        axes[-1, j].set_xscale("log", base=2); axes[-1, j].set_xticks(xticks)
        axes[-1, j].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axes[-1, j].xaxis.set_minor_formatter(mticker.NullFormatter())
        axes[0, j].axhline(2, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.7)
        axes[0, j].axhline(ttft_cap, color=F.INK2, linestyle=(0, (1, 1.5)), linewidth=0.6)
        axes[1, j].axhline(200, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.7)
    for i, (col, ylab, logy, lim) in enumerate(rows):
        if logy:
            axes[i, 0].set_yscale("log"); F.plain_log(axes[i, 0].yaxis)
        axes[i, 0].set_ylim(*lim)
        axes[i, 0].set_ylabel(ylab)
    axes[0, 0].text(0.0, 1.22, label, transform=axes[0, 0].transAxes, fontsize=F.fz(7.0), fontweight="bold",
                    ha="left", va="bottom")


def figure(gb: pd.DataFrame, mig: pd.DataFrame, out: Path):
    F.style(1.08)
    fig = plt.figure(figsize=(7.0, 3.75))
    outer = fig.add_gridspec(1, 2, wspace=0.3)
    gl = outer[0].subgridspec(3, 3, hspace=0.14, wspace=0.1)
    gr = outer[1].subgridspec(3, 3, hspace=0.14, wspace=0.1)
    al = np.array([[fig.add_subplot(gl[i, j]) for j in range(3)] for i in range(3)])
    ar = np.array([[fig.add_subplot(gr[i, j]) for j in range(3)] for i in range(3)])
    for A in (al, ar):
        for i in range(3):
            for j in range(3):
                if j:
                    A[i, j].sharey(A[i, 0]); A[i, j].tick_params(labelleft=False)
                if i < 2:
                    A[i, j].sharex(A[2, j]); A[i, j].tick_params(labelbottom=False)
    block(al, gb, [("ttft_p95", "TTFT p95 (s)", True, (0.1, 400)), ("tpot", "TPOT p50 (ms)", True, (20, 800)),
                   ("thr", "Throughput (req/s)", False, (0, None))], [2, 4, 8, 16], 120, "GB10")
    block(ar, mig, [("ttft_p95", "TTFT p95 (s)", True, (0.1, 400)), ("tpot_p95", "TPOT p95 (ms)", True, (10, 800)),
                    ("thr", "Output tok/s", False, (0, None))], [0.5, 2, 8], 120, "H100 MIG 1g.24gb")
    for A in (al, ar):
        for j in range(3):
            A[2, j].tick_params(axis="x", labelsize=F.fz(5.6))
    fig.text(0.29, 0.015, "Arrival rate (req/s, Poisson)", ha="center", fontsize=F.fz(6.4))
    fig.text(0.74, 0.015, "Arrival rate (req/s, Poisson)", ha="center", fontsize=F.fz(6.4))
    fig.legend(handles=F.eng_handles(True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6)
    fig.savefig(out / "fig_rq4_load_platforms.pdf")
    plt.close(fig)


def preemption_points(rq1_dir: Path, gb10_report: Path, mig_dirs: list[Path]) -> pd.DataFrame:
    """vLLM only: peak KV-cache use vs preemptions per run, on each platform."""
    rows = []
    for m in F.MODEL:  # L40S, from the RQ1 per-request data (polled during the run)
        d = pd.read_csv(rq1_dir / f"rq1_{m}.csv", usecols=["engine", "max_concurrent", "max_num_seqs", "workload_name",
                                                           "rate_label", "kv_cache_usage_pct", "num_preemptions"],
                        low_memory=False)
        d = d[(d.engine == "vllm") & (d.max_concurrent == 256)]
        for _, g in d.groupby(["workload_name", "max_num_seqs", "rate_label"]):
            rows.append(dict(platform="L40S", model=m, kv=100 * g.kv_cache_usage_pct.max(), pre=g.num_preemptions.max()))
    s = gb10_report.read_text(encoding="utf-8")
    gb = pd.DataFrame(json.loads(re.search(r"const D = (\{.*?\});\n", s, re.S).group(1))["rows"])
    for _, r in gb[gb.engine == "vllm"].iterrows():
        rows.append(dict(platform="GB10", model=r.model, kv=r.kv_peak, pre=r.pre))
    for root in mig_dirs:
        for d in glob.glob(str(root / "*__vllm__*")):
            m = os.path.basename(d).split("__")[0]
            sm = pd.read_csv(f"{d}/server_metrics.csv")
            for _, g in sm.groupby("rate_label"):
                rows.append(dict(platform="H100 MIG", model=m, kv=100 * g.kv_cache_usage.max(),
                                 pre=g.preemptions_total.max() - g.preemptions_total.min()))
    return pd.DataFrame(rows)


PCOL = {"L40S": "#3b3a37", "GB10": "#7b4fd6", "H100 MIG": "#b07d00"}


def figure_preemption(p: pd.DataFrame, out: Path):
    F.style(1.1)
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    off = {"L40S": -1.4, "GB10": 0.0, "H100 MIG": 1.4}
    for plat, c in PCOL.items():
        for m in F.MODEL:
            x = p[(p.platform == plat) & (p.model == m)]
            xs = np.where(x.kv >= 99, x.kv + off[plat], x.kv)
            ax.scatter(xs, x.pre, s=16, color=c, marker={"llama31_8b": "o", "mistral_nemo_12b": "s",
                                                          "qwen15_14b": "^"}[m],
                       edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlim(0, 104)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axvline(100, color=F.INK2, linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_xlabel("Peak KV-cache use (%)"); ax.set_ylabel("Preemptions per run (vLLM)")
    h1 = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=5, label=k) for k, c in PCOL.items()]
    h2 = [plt.Line2D([], [], color=F.INK2, marker=mk, linestyle="", markersize=4.5, markerfacecolor="white", label=lab)
          for mk, lab in (("o", "Llama"), ("s", "Mistral"), ("^", "Qwen1.5"))]
    leg = ax.legend(handles=h1, loc="upper left", title="Platform", fontsize=F.fz(5.8), title_fontsize=F.fz(5.8))
    ax.add_artist(leg)
    ax.legend(handles=h2, loc="center left", bbox_to_anchor=(0.0, 0.45), title="Model", fontsize=F.fz(5.8),
              title_fontsize=F.fz(5.8))
    fig.savefig(out / "fig_appendix_a6_preemption_platforms.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb10-report", type=Path, required=True)
    ap.add_argument("--mig-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper_rq4"))
    ap.add_argument("--rq1-dir", type=Path, help="L40S RQ1 CSVs, for the appendix preemption figure")
    ap.add_argument("--appendix-out", type=Path, default=Path("docs/figures/paper_appendix"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.appendix_out.mkdir(parents=True, exist_ok=True)
    gb = load_gb10(a.gb10_report)
    mig_all = mig_per_config(load_mig(a.mig_dirs))
    gb.to_csv(a.out / "rq4_gb10_percfg.csv", index=False)
    mig_all.to_csv(a.out / "rq4_mig_percfg.csv", index=False)
    mig = mig_all[(mig_all.workload_name == "mixed_real") & (mig_all.max_num_seqs == 256)]
    figure(gb, mig, a.out)
    print("wrote", a.out / "fig_rq4_load_platforms.pdf")
    if a.rq1_dir:
        p = preemption_points(a.rq1_dir, a.gb10_report, a.mig_dirs)
        p.to_csv(a.appendix_out / "appendix_a6_preemption_platforms.csv", index=False)
        figure_preemption(p, a.appendix_out)
        print(p.assign(has=p.pre > 0).groupby(["platform", "has"]).kv.agg(["count", "min", "max"]).round(1))


if __name__ == "__main__":
    main()
