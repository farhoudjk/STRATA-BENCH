"""Usage: python scripts/bootstrap_cis.py <data_root> <out_dir>
Bootstrap 95% CIs (numpy, vectorised): resample requests within each rate config, B=1000."""
import sys; sys.path.insert(0, "/home/user/STRATA-BENCH/scripts")
import numpy as np, pandas as pd
from pathlib import Path
import plot_rq2 as R2, plot_rq3 as R3
S = sys.argv[1] if len(sys.argv) > 1 else "data"  # dir with data/rq1_flowc.parquet and data/rq23/*.csv
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/stats"
rng = np.random.default_rng(1); B = 1000

def arrays(df):
    fin = df.finished.to_numpy(bool)
    return dict(ttft=df.ttft_user.to_numpy(float), fin=fin, tpot=df.tpot_ms.to_numpy(float),
                gap=df.itl_max_ms.to_numpy(float) if "itl_max_ms" in df else np.full(len(df), np.nan),
                e2e=df.e2e_s.to_numpy(float), strata=df.rate_label.to_numpy())

def idx_matrix(strata):
    out = np.empty((B, len(strata)), dtype=np.int64); pos = 0
    for s in np.unique(strata):
        g = np.flatnonzero(strata == s); out[:, pos:pos + len(g)] = rng.choice(g, (B, len(g))); pos += len(g)
    return out

def metric_boot(A, name, I):
    t, f, tp, gp, e = A["ttft"][I], A["fin"][I], A["tpot"][I], A["gap"][I], A["e2e"][I]
    if name == "ttft_p95": return np.quantile(t, .95, axis=1)
    if name == "ttft_p50": return np.quantile(t, .5, axis=1)
    if name == "goodput": return 100 * np.mean(f & (t <= 2) & (np.nan_to_num(tp, nan=1e9) <= 200), axis=1)
    x = {"tpot_p95": tp, "tpot_p50": tp, "gap_p95_ms": gp, "e2e_p50": e}[name]
    x = np.where(f, x, np.nan); q = .95 if name.endswith("p95") or name == "gap_p95_ms" else .5
    return np.nanquantile(x, q, axis=1)

def point(A, name):
    I = np.arange(len(A["ttft"]))[None, :]; return metric_boot(A, name, I)[0]

def diff_ci(a, b, names_kinds):
    A, Bb = arrays(a), arrays(b); Ia, Ib = idx_matrix(A["strata"]), idx_matrix(Bb["strata"]); res = []
    for name, kind in names_kinds:
        xa, xb = metric_boot(A, name, Ia), metric_boot(Bb, name, Ib)
        v = 100 * (xb - xa) / xa if kind == "pct" else xb - xa
        pa, pb = point(A, name), point(Bb, name)
        pt = 100 * (pb - pa) / pa if kind == "pct" else pb - pa
        lo, hi = np.nanpercentile(v, [2.5, 97.5]); res.append((name, kind, pt, lo, hi))
    return res

rows = []
def add(rq, cell, a, b, nk):
    for name, kind, pt, lo, hi in diff_ci(a, b, nk):
        rows.append(dict(rq=rq, cell=cell, metric=name, unit=kind, est=pt, lo=lo, hi=hi))

d2 = R2.load(Path(f"{S}/data/rq23/rq2_prefill_handling.csv"))
for w in R2.WL:
    for strat, slab in R2.STRAT.items():
        for e in R2.ENG:
            sel = lambda s: d2[(d2.workload_name == w) & (d2.strategy_name == s) & (d2.engine == e)]
            add("RQ2", f"{w}|{slab}|{e}", sel(R2.BASE), sel(strat),
                [("ttft_p95", "pct"), ("tpot_p95", "pct"), ("gap_p95_ms", "pct"), ("e2e_p50", "pct"), ("goodput", "pp")])
d3 = R3.load(Path(f"{S}/data/rq23/rq3_ngram_speculative_decoding.csv"))
for w in ["mixed_real", "chat_multiturn"]:
    for q in (32, 64):
        for e in R3.ENG:
            sel = lambda s: d3[(d3.workload_name == w) & (d3.max_num_seqs == q) & (d3.engine == e) & (d3.strategy_name == s)]
            add("RQ3", f"{w}|S={q}|{e}", sel(R3.BASE), sel(R3.SPEC),
                [("ttft_p50", "pct"), ("tpot_p50", "pct"), ("tpot_p95", "pct"), ("gap_p95_ms", "pct"), ("e2e_p50", "pct"), ("goodput", "pp")])
for t in ["reasoning", "code", "chat", "summarization"]:
    for e in R3.ENG:
        sel = lambda s: d3[(d3.workload_name == "mixed_real") & (d3.max_num_seqs == 64) & (d3.engine == e) & (d3.task_type == t)
                           & (d3.strategy_name == s) & d3.finished & (d3.output_tokens > 20)]
        add("RQ3-task", f"{t}|{e}", sel(R3.BASE), sel(R3.SPEC), [("tpot_p50", "pct")])
RATES = ["low_poisson", "med_poisson", "high_poisson", "vhigh_poisson"]
for e in R3.ENG:
    for q in (64, 32):
        for r in ("low_poisson", "vhigh_poisson"):
            sel = lambda s: d3[(d3.workload_name == "mixed_real") & (d3.max_num_seqs == q) & (d3.engine == e) & (d3.rate_label == r) & (d3.strategy_name == s)]
            add("RQ3-load", f"mixed|S={q}|{r}|{e}", sel(R3.BASE), sel(R3.SPEC), [("tpot_p50", "pct")])
d1 = pd.read_parquet(f"{S}/data/rq1_flowc.parquet"); d1 = d1[(d1.max_num_seqs == 256)]
d1 = d1.assign(itl_max_ms=d1.itl_max_ms)
for w in ["mixed_real", "chat_multiturn"]:
    for m_ in ["llama31_8b", "mistral_nemo_12b", "qwen15_14b"]:
        g = d1[(d1.workload_name == w) & (d1.model == m_)]
        for e1, e2 in (("vllm", "sglang"), ("vllm", "tensorrt-llm"), ("sglang", "tensorrt-llm")):
            nk = [("goodput", "pp")] + ([("e2e_p50", "pct")] if w == "chat_multiturn" else [])
            add("RQ1", f"{w}|{m_}|{e2} vs {e1}", g[g.engine == e1], g[g.engine == e2], nk)
r = pd.DataFrame(rows); r["noise"] = (r.lo <= 0) & (r.hi >= 0)
r.to_csv(f"{OUT}/bootstrap_cis.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 600)
print(r.round(1).to_string())
