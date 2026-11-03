#!/usr/bin/env python3
"""Engine-comparison figures: vLLM vs SGLang vs TensorRT-LLM (colour = engine).

    python scripts/plot_engines.py --rq1 results/Aggregated_results/rq1_trimmed.parquet \
        --rq2 results/Aggregated_results/rq2_prefill_handling.csv \
        --rq3 results/Aggregated_results/rq3_ngram_speculative_decoding.csv \
        --gb10-report path/to/rq5_gb10_report.html --out docs/figures/engines

The RQ1 parquet is the cache written by scripts/plot_rq1.py (engine names normalised,
KV in percent, TensorRT-LLM's never-live counters set to NaN).
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
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker as mticker  # noqa: E402

ENG = {  # fixed order, validated categorical slots 1-3; marker = second encoding
    "vllm": dict(label="vLLM", color="#2a78d6", marker="o"),
    "sglang": dict(label="SGLang", color="#eb6834", marker="s"),
    "tensorrt-llm": dict(label="TensorRT-LLM", color="#1baf7a", marker="^"),
}
MODEL = {"llama31_8b": "Llama-3.1-8B (GQA)", "mistral_nemo_12b": "Mistral-NeMo-12B (GQA)",
         "qwen15_14b": "Qwen1.5-14B (MHA)"}
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
RPS = {"low_poisson": 2, "med_poisson": 4, "high_poisson": 8, "vhigh_poisson": 16}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, DBL_W = 3.33, 7.0


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Linux Biolinum O", "Libertinus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}")


def eng_handles(engines=ENG, line=False):
    return [plt.Line2D([], [], color=v["color"], marker=v["marker"], markersize=4.5,
                       linestyle="-" if line else "", label=v["label"]) for v in engines.values()]


def plain_log(ax, axis="both"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, f"{axis}axis")]):
        a.set_major_formatter(fmt)
        a.set_minor_formatter(mticker.NullFormatter())


def load_csv(p):
    d = pd.read_csv(p, usecols=lambda c: c != "response_text", low_memory=False)
    d["engine"] = d["engine"].replace({"tensorrtllm": "tensorrt-llm", "trtllm": "tensorrt-llm"})
    return d


# ── E1: where each engine puts the wait ──────────────────────────────────────
def e1_where_wait(r1, out):
    d = r1[(r1.workload_name == "mixed_real") & (r1.max_concurrent == 256) & (r1.max_num_seqs == 256)]
    models = list(MODEL)
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.25), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    rows = []
    for ax, m in zip(axes, models):
        for e, s in ENG.items():
            g = d[(d.model == m) & (d.engine == e)]
            ok = g[g.status == "ok"]
            if ok.empty:
                continue
            x, y = ok.ttft_s.quantile(.95), ok.itl_max_ms.quantile(.95) / 1000
            fin = 100 * len(ok) / len(g)
            rows.append(dict(model=m, engine=e, ttft_p95_s=x, itl_max_p95_s=y, finished_pct=fin))
            ax.scatter(x, y, s=46, color=s["color"], marker=s["marker"], edgecolor="white", linewidth=0.8, zorder=3)
            off = {"vllm": ((0, 7), "center"), "sglang": ((0, 7), "center"), "tensorrt-llm": ((0, -12), "center")}[e]
            ax.annotate(f"{fin:.0f}% done", (x, y), textcoords="offset points", xytext=off[0], ha=off[1],
                        fontsize=6, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.1, 1000); ax.set_ylim(0.1, 400)
        plain_log(ax)
        ax.set_title(MODEL[m], loc="left")
        ax.set_xlabel("TTFT p95 (s)")
        # quadrant guides
        ax.axvline(10, color=GRID, linewidth=0.8); ax.axhline(10, color=GRID, linewidth=0.8)
    axes[0].set_ylabel("Longest token gap p95 (s)")
    fig.legend(handles=eng_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Under KV pressure each engine puts the wait somewhere else  (mixed_real, conc. 256, max_num_seqs 256, L40S)\n"
                 "Right = waits before the first token  ·  Up = stalls in the middle of the response",
                 x=0.06, ha="left", y=1.1, fontsize=7.5)
    save(fig, out, "fig_eng_1_where_the_wait_goes")
    return pd.DataFrame(rows)


# ── E2: load knee ────────────────────────────────────────────────────────────
def e2_knee(r1, r3, out):
    panels = [
        ("Mistral-NeMo-12B · seqs 64",
         r3[(r3.strategy_name == "baseline_full_fp16_tp1") & (r3.workload_name == "mixed_real") & (r3.max_num_seqs == 64)]),
        ("Llama-3.1-8B · seqs 256",
         r1[(r1.model == "llama31_8b") & (r1.workload_name == "mixed_real") & (r1.max_concurrent == 256) & (r1.max_num_seqs == 256)]),
        ("Qwen1.5-14B · seqs 256",
         r1[(r1.model == "qwen15_14b") & (r1.workload_name == "mixed_real") & (r1.max_concurrent == 256) & (r1.max_num_seqs == 256)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    rows = []
    for ax, (title, d) in zip(axes, panels):
        for e, s in ENG.items():
            ok = d[(d.engine == e) & (d.status == "ok") & d.rate_label.isin(RATES)]
            if ok.empty:
                continue
            p = ok.groupby("rate_label").ttft_s.median().reindex(RATES)
            ax.plot([RPS[r] for r in RATES], p.values, color=s["color"], marker=s["marker"], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6)
            rows += [dict(panel=title, engine=e, rps=RPS[r], ttft_p50_s=v) for r, v in p.items()]
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([2, 4, 8, 16]); plain_log(ax)
        ax.set_ylim(0.05, 400)
        ax.axhline(1, color=INK2, linestyle=(0, (3, 2)), linewidth=0.7)
        ax.set_title(title, loc="left"); ax.set_xlabel("Arrival rate (req/s, Poisson)")
    axes[0].set_ylabel("TTFT p50 (s)")
    axes[0].text(2.1, 1.2, "1 s", fontsize=6, color=INK2)
    fig.legend(handles=eng_handles(line=True), loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Where each engine's latency knee sits  (mixed_real, conc. 256, L40S)",
                 x=0.06, ha="left", y=1.04, fontsize=7.5)
    save(fig, out, "fig_eng_2_load_knee")
    return pd.DataFrame(rows)


# ── E3: fast kernels ≠ fast service ──────────────────────────────────────────
def e3_kernels_vs_service(r1, out):
    d = r1[(r1.max_concurrent == 256) & (r1.max_num_seqs == 256) & (r1.rate_label == "vhigh_poisson")
           & (r1.status == "ok") & (r1.output_tokens > 20)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    rows = []
    for (m, w), g in d.groupby(["model", "workload_name"]):
        pts = []
        for e, s in ENG.items():
            h = g[g.engine == e]
            if h.empty:
                continue
            x, y = h.itl_p50_ms.median(), h.ttft_s.quantile(.95)
            pts.append((x, y)); rows.append(dict(model=m, workload=w, engine=e, itl_p50_ms=x, ttft_p95_s=y))
            ax.scatter(x, y, s=26, color=s["color"], marker=s["marker"], edgecolor="white", linewidth=0.6, zorder=3)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=GRID, linewidth=0.7, zorder=1)  # same model+workload
    ax.set_yscale("log"); plain_log(ax, "y")
    ax.set_xlabel("Decode speed: ITL p50 (ms)  ←  faster kernels")
    ax.set_ylabel("TTFT p95 (s)  ↑  worse service")
    ax.legend(handles=eng_handles(), loc="lower right")
    ax.set_title("Fastest decoder ≠ fastest service\n(each grey line = one model × workload, 16 req/s)", loc="left")
    save(fig, out, "fig_eng_3_kernels_vs_service")
    return pd.DataFrame(rows)


# ── E4: how requests end ─────────────────────────────────────────────────────
def e4_outcomes(r1, out):
    d = r1.copy()
    full = d.groupby(["model", "engine"]).status.value_counts(normalize=True).unstack(fill_value=0) * 100
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 1.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    cats = [("ok", "finished", None), ("error", "zero-token reply (model)", "////"),
            ("timeout", "timeout (engine)", None)]
    shade = {"ok": 1.0, "error": 0.0, "timeout": 0.45}
    for ax, m in zip(axes, MODEL):
        y = np.arange(len(ENG))
        left = np.zeros(len(ENG))
        for key, _, hatch in cats:
            vals = np.array([full.loc[(m, e), key] if (m, e) in full.index and key in full else 0 for e in ENG])
            cols = [ENG[e]["color"] for e in ENG]
            if key == "ok":
                ax.barh(y, vals, left=left, color=cols, height=0.62, edgecolor="white", linewidth=1)
            elif key == "error":
                ax.barh(y, vals, left=left, color="white", hatch=hatch, height=0.62, edgecolor=INK2, linewidth=0.5)
            else:
                ax.barh(y, vals, left=left, color=INK, height=0.62, edgecolor="white", linewidth=1)
            for yi, (l, v) in enumerate(zip(left, vals)):
                if key != "ok" and v >= 6:
                    ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=6,
                            color="white" if key == "timeout" else INK)
            left += vals
        ax.set_yticks(y, [ENG[e]["label"] for e in ENG]); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_title(MODEL[m], loc="left"); ax.set_xlabel("Share of requests (%)")
        ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=INK2, label="finished (engine colour)"),
               matplotlib.patches.Patch(facecolor="white", edgecolor=INK2, hatch="////", label="zero-token reply → model/prompt issue"),
               matplotlib.patches.Patch(color=INK, label="timeout → engine issue")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("How requests end: zero-token failures track the model, timeouts track the engine  (all RQ1 runs, L40S)",
                 x=0.06, ha="left", y=1.05, fontsize=7.5)
    save(fig, out, "fig_eng_4_request_outcomes")
    return full.reset_index()


# ── E5: feature payoff per engine ────────────────────────────────────────────
def _pct(a, b):
    return 100 * (b - a) / a


def e5_features(r2, r3, out):
    ok2, ok3 = r2[(r2.status == "ok") & (r2.output_tokens > 1)].copy(), r3[r3.status == "ok"]
    ok2["tpot_ms"] = (ok2.e2e_s - ok2.ttft_s) / (ok2.output_tokens - 1) * 1000  # robust to tokens per chunk

    def q(d, strat, wl, col, fn, extra=None):
        g = d[(d.strategy_name == strat) & (d.workload_name == wl)]
        if extra is not None:
            g = g[extra(g)]
        return g.groupby("engine")[col].agg(fn)

    p95 = lambda s: s.quantile(.95)  # noqa: E731
    seqs64 = lambda g: g.max_num_seqs == 64  # noqa: E731
    rows = [
        ("Prefix caching → TTFT p95\nchat_multiturn",
         q(ok2, "baseline_full_fp16_tp1_nopfx", "chat_multiturn", "ttft_s", p95),
         q(ok2, "baseline_full_fp16_tp1", "chat_multiturn", "ttft_s", p95)),
        ("Chunked prefill → TTFT p95\nchat_real_long",
         q(ok2, "baseline_full_fp16_tp1", "chat_real_long", "ttft_s", p95),
         q(ok2, "chunked_512_fp16_tp1", "chat_real_long", "ttft_s", p95)),
        ("Chunked prefill → TPOT p95\nchat_real_long",
         q(ok2, "baseline_full_fp16_tp1", "chat_real_long", "tpot_ms", p95),
         q(ok2, "chunked_512_fp16_tp1", "chat_real_long", "tpot_ms", p95)),
        ("N-gram spec. → E2E p50\nchat_multiturn",
         q(ok3, "baseline_full_fp16_tp1", "chat_multiturn", "e2e_s", "median", seqs64),
         q(ok3, "spec_ngram3_fp16_tp1", "chat_multiturn", "e2e_s", "median", seqs64)),
        ("N-gram spec. → E2E p50\nmixed_real",
         q(ok3, "baseline_full_fp16_tp1", "mixed_real", "e2e_s", "median", seqs64),
         q(ok3, "spec_ngram3_fp16_tp1", "mixed_real", "e2e_s", "median", seqs64)),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    recs = []
    for i, (name, a, b) in enumerate(rows):
        ax.axhspan(i - 0.42, i + 0.42, color="#f4f3f0" if i % 2 == 0 else "white", zorder=0, linewidth=0)
        for j, (e, s) in enumerate(ENG.items()):
            if e not in a or e not in b:
                continue
            v = _pct(a[e], b[e]); recs.append(dict(feature=name.replace("\n", " · "), engine=e, before=a[e], after=b[e], pct=v))
            yy = i + (j - 1) * 0.22
            ax.plot([0, v], [yy, yy], color=s["color"], linewidth=1.2, zorder=2)
            ax.scatter(v, yy, s=22, color=s["color"], marker=s["marker"], edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(v + (2 if v >= 0 else -2), yy, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=5.8, color=INK2)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(-75, 45); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change vs feature off (%)   ←  better   |   worse  →")
    ax.legend(handles=eng_handles(), loc="upper right", fontsize=6)
    ax.set_title("Same feature, different payoff per engine (L40S)", loc="left")
    save(fig, out, "fig_eng_5_feature_payoff")
    return pd.DataFrame(recs)


# ── E6: the ranking survives a hardware change ───────────────────────────────
def e6_hardware(r1, gb10, out):
    q = r1[(r1.model == "qwen15_14b") & (r1.max_concurrent == 256) & (r1.max_num_seqs == 256)]
    l40 = {}
    for e in ENG:
        g = q[q.engine == e]
        per = g[g.status == "ok"].groupby(["workload_name", "rate_label"]).ttft_s.median()
        l40[e] = dict(ttft=per.median(), fin=100 * (g.status == "ok").mean())
    gq = gb10[(gb10.model == "qwen15_14b")]
    gb = {}
    for e in ENG:
        g = gq[gq.engine == e]
        gb[e] = dict(ttft=g[g.seqs == 256].ttft_p50.median(), fin=(g.ok_pct * g.n).sum() / g.n.sum())
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.1), gridspec_kw=dict(wspace=0.5))
    for ax, key, lab, logy in ((axes[0], "ttft", "TTFT p50 (s)", True),
                               (axes[1], "fin", "Finished (%)", False)):
        for e, s in ENG.items():
            ax.plot([0, 1], [l40[e][key], gb[e][key]], color=s["color"], marker=s["marker"], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6)
        # direct labels, pushed apart so they never collide
        ys = sorted(((gb[e][key], e) for e in ENG), key=lambda t: t[0])
        placed = []
        for v, e in ys:
            pos = np.log10(v) if logy else v
            gap = 0.12 if logy else 7
            if placed and pos - placed[-1] < gap:
                pos = placed[-1] + gap
            placed.append(pos)
            ax.text(1.08, 10 ** pos if logy else pos, ENG[e]["label"], va="center", fontsize=6, color=INK2)
        ax.set_xticks([0, 1], ["L40S\n44 GiB", "GB10\n121 GiB"]); ax.set_xlim(-0.2, 1.6)
        ax.grid(axis="x", visible=False); ax.set_title(lab, loc="left", fontsize=7)
        if logy:
            ax.set_yscale("log"); plain_log(ax, "y")
        else:
            ax.set_ylim(0, 105)
    fig.suptitle("Qwen1.5-14B (MHA), L40S vs GB10: TensorRT-LLM is last on both;\nvLLM pulls ahead with more memory (newer engine versions on GB10)",
                 x=0.02, ha="left", y=1.1, fontsize=7.5)
    save(fig, out, "fig_eng_6_hardware_ranking")
    return pd.DataFrame([dict(engine=e, platform=p, **v) for p, dd in (("L40S", l40), ("GB10", gb)) for e, v in dd.items()])


# ── E7: TTFT depends on time-in-run (methodology) ────────────────────────────
def e7_time_in_run(r1, out):
    d = r1[(r1.model == "llama31_8b") & (r1.workload_name == "mixed_real") & (r1.max_concurrent == 256)
           & (r1.max_num_seqs == 256) & (r1.status == "ok") & (r1.engine != "tensorrt-llm")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, rate in zip(axes, ["low_poisson", "vhigh_poisson"]):
        for e in ("vllm", "sglang"):
            g = d[(d.engine == e) & (d.rate_label == rate)]
            t = g.send_time - g.send_time.min()
            ax.scatter(t, g.ttft_s.clip(lower=0.05), s=3, color=ENG[e]["color"], marker=ENG[e]["marker"],
                       alpha=0.6, linewidth=0, zorder=3)
        ax.set_yscale("log"); plain_log(ax, "y"); ax.set_ylim(0.04, 400)
        ax.set_title(f"{RPS[rate]} req/s", loc="left"); ax.set_xlabel("Request send time (s)")
    axes[0].set_ylabel("TTFT (s)")
    fig.legend(handles=eng_handles(dict((k, ENG[k]) for k in ("vllm", "sglang"))), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Lower load, worse TTFT: late arrivals hit a full KV pool\n(Llama-3.1-8B, mixed_real, 500 requests)",
                 x=0.02, ha="left", y=1.1, fontsize=7.5)
    save(fig, out, "fig_eng_7_time_in_run")


def load_gb10(path):
    s = Path(path).read_text(encoding="utf-8")
    return pd.DataFrame(json.loads(re.search(r"const D = (\{.*?\});\n", s, re.S).group(1))["rows"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1", type=Path, required=True)
    ap.add_argument("--rq2", type=Path, required=True)
    ap.add_argument("--rq3", type=Path, required=True)
    ap.add_argument("--gb10-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/engines"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    r1, r2, r3, gb = pd.read_parquet(a.rq1), load_csv(a.rq2), load_csv(a.rq3), load_gb10(a.gb10_report)
    e1_where_wait(r1, a.out).to_csv(a.out / "eng_1_where_the_wait_goes.csv", index=False)
    e2_knee(r1, r3, a.out).to_csv(a.out / "eng_2_load_knee.csv", index=False)
    e3_kernels_vs_service(r1, a.out).to_csv(a.out / "eng_3_kernels_vs_service.csv", index=False)
    e4_outcomes(r1, a.out).to_csv(a.out / "eng_4_request_outcomes.csv", index=False)
    e5_features(r2, r3, a.out).to_csv(a.out / "eng_5_feature_payoff.csv", index=False)
    e6_hardware(r1, gb, a.out).to_csv(a.out / "eng_6_hardware_ranking.csv", index=False)
    e7_time_in_run(r1, a.out)
    print(f"Done → {a.out}/")


if __name__ == "__main__":
    main()
