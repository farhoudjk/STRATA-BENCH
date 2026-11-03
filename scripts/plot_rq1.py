#!/usr/bin/env python3
"""RQ1 paper figures: KV-cache pressure and preemption across models and engines.

Reads the per-request RQ1 CSVs (one per model), drops the bulky response_text
column, caches a trimmed Parquet copy, and writes six vector figures plus the
aggregated numbers behind them.

    python scripts/plot_rq1.py --data results/Aggregated_results \
        --gb10-report path/to/rq5_gb10_report.html --out docs/figures/rq1

Figures (ACM single-column width unless noted):
  fig_rq1_1_kv_vs_preemption   peak KV-cache use vs preemptions, one point per rate config
  fig_rq1_2_saturation_timeline KV use and cumulative preemptions over a run (double column)
  fig_rq1_3_output_length      ECDF of output tokens per model, reasoning vs other tasks (confound check)
  fig_rq1_4_itl_cost           max inter-token latency, requests overlapping preemption vs not
  fig_rq1_5_engine_policy      running/waiting sequences per engine + TTFT (double column)
  fig_rq1_6_max_num_seqs       max_num_seqs 64 vs 256: preemptions, TTFT p95, throughput

Metric semantics (from benchmark/runner.py):
  num_preemptions        server-wide cumulative vLLM counter, snapshot at request completion
  num_preemptions_delta  server-wide preemptions during this request's lifetime, not
                         necessarily preemptions *of* this request
  kv_cache_usage_pct     server KV-cache use snapshot at request completion
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

# ── Style ────────────────────────────────────────────────────────────────────
# Colour follows the model in every figure. First three slots of the validated
# reference categorical palette (all-pairs safe for scatter); marker shape is a
# second encoding so figures survive grayscale print.
MODELS = {
    "llama31_8b": dict(label="Llama-3.1-8B (GQA)", color="#2a78d6", marker="o"),
    "mistral_nemo_12b": dict(label="Mistral-NeMo-12B (GQA)", color="#eb6834", marker="s"),
    "qwen15_14b": dict(label="Qwen1.5-14B (MHA)", color="#1baf7a", marker="^"),
}
ENGINES = {"vllm": "vLLM", "sglang": "SGLang", "tensorrt-llm": "TensorRT-LLM"}
RATE_ORDER = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson", "bursty"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, DBL_W = 3.33, 7.0  # ACM sigconf column / text width, inches

USECOLS = [
    "request_id", "workload_name", "task_type", "engine", "model", "run_id",
    "rate_label", "max_concurrent", "max_num_seqs", "send_time", "ttft_s", "e2e_s",
    "output_tokens", "target_output_tokens", "completion_ratio", "status",
    "kv_cache_usage_pct", "num_waiting_seqs", "num_running_seqs",
    "num_preemptions", "num_preemptions_delta", "itl_p95_ms", "itl_max_ms",
]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType (ACM requirement)
    })


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def model_legend(ax, models, **kw):
    handles = [plt.Line2D([], [], color=MODELS[m]["color"], marker=MODELS[m]["marker"],
                          linestyle="-", markersize=4, label=MODELS[m]["label"]) for m in models]
    ax.legend(handles=handles, **kw)


# ── Loading ──────────────────────────────────────────────────────────────────
def load(data_dir: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        print(f"Using cache {cache}")
        return pd.read_parquet(cache)
    files = sorted(data_dir.glob("rq1_*.csv"))
    if not files:
        sys.exit(f"No rq1_*.csv files in {data_dir}")
    frames = []
    for f in files:
        print(f"Reading {f.name} ...")
        df = pd.read_csv(f, usecols=lambda c: c in USECOLS, low_memory=False)
        if "model" not in df or df["model"].isna().all():
            df["model"] = f.stem.removeprefix("rq1_")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["engine"] = df["engine"].replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    for eng, g in df.groupby("engine"):
        kv = g["kv_cache_usage_pct"]
        if kv.fillna(0).eq(0).all():
            # Engine never reported KV use live (TensorRT-LLM in this harness): unknown, not 0%.
            df.loc[g.index, "kv_cache_usage_pct"] = np.nan
        elif kv.max(skipna=True) <= 1.0:
            # vLLM/SGLang export the KV gauge as a 0–1 fraction; normalise to percent.
            df.loc[g.index, "kv_cache_usage_pct"] *= 100.0
        if g["num_waiting_seqs"].fillna(0).eq(0).all() and g["num_running_seqs"].fillna(0).le(1).all():
            # Placeholder values (running<=1, waiting=0 throughout): not exposed live.
            df.loc[g.index, ["num_running_seqs", "num_waiting_seqs"]] = np.nan
    # TensorRT-LLM's paused-request count is scraped once after the run, when the server
    # is idle, so it always reads 0 (draft §3.3). Treat it as unmeasured.
    tr = df["engine"] == "tensorrt-llm"
    df.loc[tr, ["num_preemptions", "num_preemptions_delta"]] = np.nan
    df.to_parquet(cache, index=False)
    print(f"Cached trimmed data to {cache} ({cache.stat().st_size / 1e6:.1f} MB)")
    return df


CFG = ["model", "engine", "workload_name", "max_concurrent", "max_num_seqs", "run_id", "rate_label"]


def per_config(df: pd.DataFrame) -> pd.DataFrame:
    """One row per rate config, with preemptions *within* that config.

    num_preemptions is cumulative per server. If one server served several rate
    configs, the counter carries over, so each config's count is its max minus
    the previous config's max. A counter that drops marks a server restart.
    """
    rows = []
    for key, run in df.groupby(CFG[:-1], dropna=False):
        prev_max, carried = 0.0, False
        groups = sorted(run.groupby("rate_label"), key=lambda kv: kv[1]["send_time"].min())
        for rate, g in groups:
            p = g["num_preemptions"].dropna()
            if p.empty:
                pre = np.nan
            else:
                restarted = p.min() < prev_max
                base = 0.0 if restarted else prev_max
                carried |= (not restarted and prev_max > 0)
                pre = p.max() - base
                prev_max = p.max()
            ok = g[g["status"] == "ok"]
            t0, t1 = g["send_time"].min(), (g["send_time"] + g["e2e_s"]).max()
            rows.append(dict(zip(CFG, (*key, rate)), preemptions=pre, carried_over=carried,
                             kv_peak=g["kv_cache_usage_pct"].max(), kv_mean=g["kv_cache_usage_pct"].mean(),
                             n=len(g), ok_pct=100 * len(ok) / len(g),
                             ttft_p50=ok["ttft_s"].median(), ttft_p95=ok["ttft_s"].quantile(.95),
                             e2e_p99=ok["e2e_s"].quantile(.99),
                             req_per_s=len(ok) / (t1 - t0) if t1 > t0 else np.nan))
    return pd.DataFrame(rows)


def load_gb10(path: Path) -> pd.DataFrame:
    s = path.read_text(encoding="utf-8")
    m = re.search(r"const D = (\{.*?\});\n", s, re.S)
    if not m:
        sys.exit(f"No data block found in {path}")
    g = pd.DataFrame(json.loads(m.group(1))["rows"])
    return g.rename(columns={"pre": "preemptions", "workload": "workload_name", "seqs": "max_num_seqs"})


# ── Figures ──────────────────────────────────────────────────────────────────
def fig1(cfg: pd.DataFrame, gb10: pd.DataFrame | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    v = cfg[(cfg.engine == "vllm") & cfg.preemptions.notna()]
    for m in MODELS:
        d = v[v.model == m]
        ax.scatter(d.kv_peak, d.preemptions, s=16, marker=MODELS[m]["marker"],
                   color=MODELS[m]["color"], edgecolor="white", linewidth=0.4, zorder=3)
        if gb10 is not None:
            g = gb10[(gb10.model == m) & (gb10.engine == "vllm")]
            ax.scatter(g.kv_peak, g.preemptions, s=16, marker=MODELS[m]["marker"],
                       facecolor="none", edgecolor=MODELS[m]["color"], linewidth=0.8, zorder=3)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(0, 104)
    ax.axvline(100, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_xlabel("Peak KV-cache use in rate config (%)")
    ax.set_ylabel("Preemptions per rate config")
    handles = [plt.Line2D([], [], color=MODELS[m]["color"], marker=MODELS[m]["marker"],
                          linestyle="", markersize=4, label=MODELS[m]["label"]) for m in MODELS]
    if gb10 is not None:
        handles += [plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=4, label="L40S"),
                    plt.Line2D([], [], markerfacecolor="none", markeredgecolor=INK2, marker="o",
                               linestyle="", markersize=4, label="GB10")]
    ax.legend(handles=handles, loc="upper left", ncol=1)
    ax.set_title("Preemption appears only when the KV pool is full (vLLM)", loc="left")
    save(fig, out, "fig_rq1_1_kv_vs_preemption")


def _pick_run(df, model, engine, workload, conc=256, seqs=256):
    d = df[(df.model == model) & (df.engine == engine) & (df.workload_name == workload)
           & (df.max_concurrent == conc) & (df.max_num_seqs == seqs)]
    for rate in reversed(RATE_ORDER[:4]):  # highest Poisson rate available
        g = d[d.rate_label == rate]
        if len(g):
            return g.sort_values("send_time"), rate
    return d.iloc[0:0], None


def _timeline(g: pd.DataFrame, cols, window: int = 9) -> pd.DataFrame:
    """Metrics are snapshotted at request completion: order by completion time.

    A centred rolling median removes single-sample jitter without shifting steps.
    """
    t = (g.send_time + g.e2e_s)
    d = g.assign(t=t - g.send_time.min()).sort_values("t")
    for c in cols:
        d[c] = d[c].rolling(window, center=True, min_periods=1).median()
    return d


def fig2(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 2.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.12, wspace=0.25))
    rates = set()
    for j, m in enumerate(MODELS):
        g, rate = _pick_run(df, m, "vllm", "mixed_real")
        a_kv, a_pre = axes[0, j], axes[1, j]
        a_kv.set_title(MODELS[m]["label"], loc="left")
        if g.empty:
            a_kv.text(.5, .5, "no run", ha="center", transform=a_kv.transAxes); continue
        rates.add(rate)
        d = _timeline(g, ["kv_cache_usage_pct"], window=1)  # raw: smoothing hides short saturation spikes
        c = MODELS[m]["color"]
        a_kv.plot(d.t, d.kv_cache_usage_pct, color=c, linewidth=1.0)
        a_kv.axhline(100, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
        a_kv.set_ylim(0, 105)
        pre = d.num_preemptions.cummax() - d.num_preemptions.min()  # counter is monotonic
        a_pre.step(d.t, pre, where="post", color=c)
        a_pre.set_ylim(0, max(1.0, pre.max()) * 1.1)
        a_pre.set_xlabel("Time since first request (s)")
        if j == 0:
            a_kv.set_ylabel("KV-cache use (%)")
            a_pre.set_ylabel("Cumulative\npreemptions")
    fig.suptitle("KV-cache use and cumulative preemptions over one run — vLLM, mixed_real, "
                 f"conc. 256, max_num_seqs 256, {'/'.join(sorted(r for r in rates if r))}",
                 x=0.07, ha="left", y=1.03, fontsize=7.5)
    save(fig, out, "fig_rq1_2_saturation_timeline")


def fig3(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Same prompts, different output lengths: reasoning vs all other task types."""
    ok = df[(df.status == "ok") & (df.workload_name == "mixed_real") & (df.output_tokens > 0)]
    groups = [("reasoning", ok.task_type == "reasoning"), ("chat, code, summarization", ok.task_type != "reasoning")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True, squeeze=False)
    stats = []
    for ax, (name, mask) in zip(axes[0], groups):
        for m in MODELS:
            x = np.sort(ok[mask & (ok.model == m)].output_tokens.values)
            if not len(x):
                continue
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MODELS[m]["color"])
            stats.append(dict(model=m, tasks=name, n=len(x), mean_tokens=x.mean(), median_tokens=np.median(x),
                              p90_tokens=np.quantile(x, .9)))
        ax.set_xscale("log")
        ax.set_title(name, loc="left"); ax.set_xlabel("Output tokens per request")
    axes[0, 0].set_ylabel("Share of requests (ECDF)")
    handles = [plt.Line2D([], [], color=MODELS[m]["color"], label=MODELS[m]["label"]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0,
               handlelength=1.5)
    save(fig, out, "fig_rq1_3_output_length")
    return pd.DataFrame(stats)


def fig4(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.engine == "vllm") & (df.workload_name == "mixed_real") & (df.max_concurrent == 256)
           & (df.max_num_seqs == 256) & (df.status == "ok") & df.itl_max_ms.notna()]
    models = [m for m in MODELS if (d[d.model == m].num_preemptions_delta > 0).any()]
    if not models:
        print("  skip fig4: no requests overlapping preemption"); return
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for m in models:
        for hit, ls in ((False, "-"), (True, (0, (3, 1.5)))):
            x = d[(d.model == m) & ((d.num_preemptions_delta > 0) == hit)].itl_max_ms.values
            if len(x) < 5:
                continue
            x = np.sort(x)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                    color=MODELS[m]["color"], linestyle=ls)
    ax.set_xscale("log")
    ax.set_xlabel("Max inter-token latency per request (ms)")
    ax.set_ylabel("Share of requests (ECDF)")
    handles = [plt.Line2D([], [], color=MODELS[m]["color"], label=MODELS[m]["label"]) for m in models]
    handles += [plt.Line2D([], [], color=INK2, label="no preemption during request"),
                plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="preemption during request")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=1.8)
    ax.set_title("Preemption shows up as inter-token stalls (vLLM, mixed_real)", loc="left")
    save(fig, out, "fig_rq1_4_itl_cost")


def fig5(df: pd.DataFrame, out: Path) -> None:
    m = "qwen15_14b"
    engines = [e for e in ENGINES if ((df.model == m) & (df.engine == e)).any()]
    fig, axes = plt.subplots(1, len(engines) + 1, figsize=(DBL_W, 1.9),
                             gridspec_kw=dict(width_ratios=[1] * len(engines) + [0.9], wspace=0.6))
    ttft, rates = [], set()
    seq_axes = list(axes[:len(engines)])
    for ax in seq_axes[1:]:
        ax.sharey(seq_axes[0])  # shared scale makes admission caps comparable
    for ax, e in zip(seq_axes, engines):
        g, rate = _pick_run(df, m, e, "mixed_real")
        ax.set_title(ENGINES[e], loc="left")
        ax.set_xlabel("Time since first request (s)")
        if g.empty:
            continue
        rates.add(rate)
        ok = g[g.status == "ok"]
        ttft.append((ENGINES[e], ok.ttft_s.median(), ok.ttft_s.quantile(.95)))
        if g.num_running_seqs.fillna(0).eq(0).all() and g.num_waiting_seqs.fillna(0).eq(0).all():
            ax.text(.5, .5, "running/waiting\ncounts not\nexposed live", ha="center", va="center",
                    transform=ax.transAxes, color=INK2)
            ax.tick_params(labelbottom=False); ax.grid(False)
            continue
        d = _timeline(g, ["num_running_seqs", "num_waiting_seqs"])
        ax.plot(d.t, d.num_running_seqs, color=INK, linewidth=1.0)
        ax.plot(d.t, d.num_waiting_seqs, color=INK2, linewidth=1.0, linestyle=(0, (3, 1.5)))
    seq_axes[0].set_ylabel("Sequences")
    seq_axes[0].set_ylim(bottom=0)
    a = axes[-1]
    y = np.arange(len(ttft))
    a.barh(y, [r[2] for r in ttft], color=GRID, height=0.6)
    a.barh(y, [r[1] for r in ttft], color=MODELS[m]["color"], height=0.6)
    a.set_yticks(y, [r[0] for r in ttft]); a.invert_yaxis()
    a.set_xscale("log")
    a.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("TTFT (s)"); a.set_title("TTFT p50 / p95", loc="left")
    a.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=INK, label="running"),
               plt.Line2D([], [], color=INK2, linestyle=(0, (3, 1.5)), label="waiting"),
               matplotlib.patches.Patch(color=MODELS[m]["color"], label="TTFT p50"),
               matplotlib.patches.Patch(color=GRID, label="TTFT p95")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Under full-MHA KV pressure every engine queues; vLLM drains fastest\nQwen1.5-14B, mixed_real, conc. 256, "
                 f"max_num_seqs 256, {'/'.join(sorted(r for r in rates if r))}",
                 x=0.07, ha="left", y=1.1, fontsize=7.5)
    save(fig, out, "fig_rq1_5_engine_policy")


def fig6(cfg: pd.DataFrame, out: Path) -> pd.DataFrame:
    d = cfg[(cfg.workload_name == "mixed_real") & (cfg.max_concurrent == 256)]
    agg = (d.groupby(["model", "engine", "max_num_seqs"])
             .agg(preemptions=("preemptions", lambda s: s.sum(min_count=1)),
                  ttft_p95=("ttft_p95", "median"), req_per_s=("req_per_s", "median"))
             .reset_index())
    panels = [("preemptions", "Preemptions, vLLM (sum over rates)", True),
              ("ttft_p95", "TTFT p95 (s, median over rates)", True),
              ("req_per_s", "Completed req/s (median over rates)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 1.9), gridspec_kw=dict(wspace=0.35))
    ls_by_engine = {"vllm": "-", "sglang": (0, (3, 1.5)), "tensorrt-llm": (0, (1, 1.2))}
    for ax, (col, label, logy) in zip(axes, panels):
        for (m, e), g in agg.groupby(["model", "engine"]):
            g = g.sort_values("max_num_seqs")
            if g[col].isna().all() or len(g) < 2 or m not in MODELS:
                continue
            ax.plot(g.max_num_seqs.astype(str), g[col], color=MODELS[m]["color"],
                    marker=MODELS[m]["marker"], markersize=3.5, linestyle=ls_by_engine.get(e, "-"))
        if logy:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_title(label, loc="left"); ax.set_xlabel("max_num_seqs")
        ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=MODELS[m]["color"], marker=MODELS[m]["marker"],
                          markersize=3.5, label=MODELS[m]["label"]) for m in MODELS]
    handles += [plt.Line2D([], [], color=INK2, linestyle=ls, label=ENGINES[e]) for e, ls in ls_by_engine.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.16))
    save(fig, out, "fig_rq1_6_max_num_seqs")
    return agg


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("results/Aggregated_results"))
    ap.add_argument("--gb10-report", type=Path, help="RQ4 GB10 HTML report, overlaid on figure 1")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq1"))
    ap.add_argument("--cache", type=Path, help="trimmed Parquet cache (default: <data>/rq1_trimmed.parquet)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()

    df = load(a.data, a.cache or a.data / "rq1_trimmed.parquet")
    print(f"{len(df):,} requests · models {sorted(df.model.unique())} · engines {sorted(df.engine.unique())}")
    cfg = per_config(df)
    cfg.to_csv(a.out / "rq1_per_config.csv", index=False)
    if cfg.carried_over.any():
        print("  NOTE: preemption counters carry over between rate configs within a run "
              "(one server per invocation); per-config counts are differenced.")
    gb10 = load_gb10(a.gb10_report) if a.gb10_report else None

    fig1(cfg, gb10, a.out)
    fig2(df, a.out)
    fig3(df, a.out).to_csv(a.out / "rq1_output_length.csv", index=False)
    fig4(df, a.out)
    fig5(df, a.out)
    fig6(cfg, a.out).to_csv(a.out / "rq1_max_num_seqs.csv", index=False)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
