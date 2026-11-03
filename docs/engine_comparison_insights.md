# Engine comparison: insights across RQ1–RQ4

vLLM vs SGLang vs TensorRT-LLM. All numbers are computed from the per-request CSVs
(L40S: `rq1_*.csv`, `rq2_prefill_handling.csv`, `rq3_ngram_speculative_decoding.csv`)
and the GB10 report data. TTFT = time to first token; ITL = inter-token latency;
E2E = end-to-end latency. Percentiles cover finished requests only.

## Summary: no engine dominates

| Aspect | Best | Worst | Evidence |
|---|---|---|---|
| Light-load latency | vLLM (TTFT) — decode is a tie | — | §1 |
| Per-token decode speed under load | TensorRT-LLM | vLLM | §2 |
| Load before latency explodes | vLLM ≈ SGLang | TensorRT-LLM | §3 |
| Time to first token under KV pressure | vLLM | TensorRT-LLM | §4, §5 |
| Smooth streaming under KV pressure | SGLang / TensorRT-LLM | vLLM | §4 |
| Graceful degradation (MHA model) | vLLM | TensorRT-LLM | §5 |
| E2E on uniform traffic | TensorRT-LLM | SGLang | §6 |
| Gain from prefix caching | tie | — | §7 |
| Gain from chunked prefill (long prompts) | TensorRT-LLM (vLLM: smoothness only) | SGLang | §8 |
| Gain from n-gram speculation | vLLM | TensorRT-LLM | §9 |
| Observability | vLLM | TensorRT-LLM | §11 |

**Thesis.** At light load the engines are interchangeable; kernels are within a few percent.
They diverge only once the KV pool or admission capacity saturates, and they diverge in
*where they put the wait*. Engine choice is a choice of failure mode, not of peak speed.

---

## 1. Light load: kernels are a tie, vLLM starts fastest
RQ1, concurrency 16, 2 req/s.

| Model | ITL p50 (SGLang / TRT-LLM / vLLM) | TTFT p50 (SGLang / TRT-LLM / vLLM) |
|---|---|---|
| Llama-3.1-8B | 27.4 / 27.3 / 28.1 ms | 85 / 94 / **67** ms |
| Mistral-NeMo-12B | 35.7 / 37.1 / 37.9 ms | 112 / 123 / **85** ms |

Decode speed differs by ≤ 6%; vLLM's first token is 20–30% faster. Below saturation,
engine choice barely matters.

## 2. Under load, TensorRT-LLM has the fastest decode and vLLM the slowest
RQ1, concurrency 256, max_num_seqs 256, 16 req/s, ITL p50:

| Workload · model | SGLang | TRT-LLM | vLLM |
|---|---|---|---|
| mixed_real · Qwen1.5-14B | 52.4 | **41.6** | 60.8 ms |
| mixed_real · Mistral-NeMo | 49.8 | **40.1** | 58.0 ms |
| chat_multiturn · Qwen1.5-14B | 57.3 | **54.7** | 59.6 ms |

TensorRT-LLM's batched decode is up to 46% faster per token than vLLM's. The fastest
kernels do not make the fastest service (§3–§5).

## 3. TensorRT-LLM saturates at about half the arrival rate
RQ3 baseline, Mistral-NeMo, mixed_real, max_num_seqs 64. TTFT p50:

| Arrival rate | vLLM | TRT-LLM |
|---|---|---|
| 2 req/s | 0.08 s | 0.1 s |
| 4 req/s | **0.09 s** | **28.8 s** |
| 8 req/s | 9.4 s | 49.9 s |
| 16 req/s | 19.0 s | 64.4 s |

The penalty is the same for every task type (TTFT p95 ≈ 89–91 s for chat, code, reasoning
and summarization alike), so it is not long-prompt prefill. In RQ1, 2,995 of
TensorRT-LLM's 3,037 timeouts are requests that **never received a first token**.
The bottleneck is admission throughput, not decode.

**Before publishing:** audit the TensorRT-LLM adapter's `max_num_tokens`, KV-cache memory
fraction and batch-size settings. If a default is limiting admission, this is a
configuration finding, not an engine property.

## 4. Under KV pressure each engine moves the wait somewhere different
RQ1, Llama-3.1-8B, mixed_real, concurrency 256, max_num_seqs 256:

| Engine | TTFT p95 | Max ITL p95 | Finished | Where requests wait |
|---|---|---|---|---|
| vLLM | **0.27 s** | **122 s** | 92.0% | mid-stream (evict, recompute) |
| SGLang | 138 s | 0.67 s | 93.1% | before the first token (admission) |
| TRT-LLM | 299 s | 0.34 s | 82.1% | before the first token, then timeout |

vLLM admits immediately and later evicts: users see a fast start, then a stall of up to
minutes. SGLang holds requests back: a long wait, then smooth streaming. The right engine
depends on the SLO: TTFT-bound (chat UX) favours vLLM, streaming-smoothness-bound (voice,
agents) favours SGLang.

## 5. When the model exceeds KV capacity, every engine queues; vLLM drains fastest
RQ1, Qwen1.5-14B (full MHA), mixed_real, concurrency 256, max_num_seqs 256:

| Engine | TTFT p50 | TTFT p95 | Finished | Peak running |
|---|---|---|---|---|
| vLLM | **37 s** | 148 s | 100% | 64 |
| SGLang | 90 s | 230 s | 100% | 41 |
| TRT-LLM | 286 s | 300 s | **58.7%** | n/a |

max_num_seqs (256) is irrelevant: the KV pool caps concurrency at 41–83. vLLM keeps the
most sequences running because it is willing to preempt, and that converts directly into
lower queueing delay. **Preemption is how vLLM buys concurrency, not a defect.**

**The ranking holds on GB10** (RQ4) with newer engine versions: Qwen median TTFT
2.1 s (vLLM) / 56.9 s (SGLang) / 105 s (TRT-LLM); TRT-LLM finishes 35% of requests.
Same order, different hardware and software: a robust result.

## 6. On uniform traffic, TensorRT-LLM is the fastest end to end
E2E median, baseline strategy, max_num_seqs 256 (RQ2):

| Workload | SGLang | TRT-LLM | vLLM |
|---|---|---|---|
| chat_multiturn | 17.1 s | **15.4 s** | 16.5 s |
| chat_real_long | 41.6 s | **38.5 s** | 42.0 s |

TensorRT-LLM wins when request shapes are homogeneous and load stays below its admission
knee, and loses on heterogeneous `mixed_real` (§3). It is a throughput engine for
predictable traffic.

## 7. Prefix caching: equal benefit, incomparable hit-rate metrics
RQ2, chat_multiturn, TTFT p95 off → on:

| Engine | TTFT p95 off → on | Reported hit rate |
|---|---|---|
| vLLM | 0.46 → 0.20 s (−57%) | 0.60 |
| SGLang | 0.51 → 0.22 s (−57%) | 0.93 |
| TRT-LLM | 0.50 → 0.25 s (−50%) | 0.30 |

Reported hit rates differ 3×, but the latency benefit is nearly identical. The engines
define "hit rate" differently (token vs block granularity, cumulative vs windowed).
**Compare caching by its latency effect, never by engine-reported hit rate.**

## 8. Chunked prefill is engine-specific: it smooths streaming on vLLM and TensorRT-LLM, hurts SGLang
RQ2, chat_real_long (long prompts), prefix caching on, no chunking → 512-token chunks:

| Engine | TTFT p95 | TPOT p95 | Longest gap p95 | E2E p50 |
|---|---|---|---|---|
| TRT-LLM | 80.0 → **69.3 s** (−13%) | 99.6 → **83.9 ms** (−16%) | 1.02 → **0.12 s** (−88%) | 38.5 → **34.1 s** (−11%) |
| vLLM | 83.0 → 80.2 s (−3%) | 118.5 → **99.7 ms** (−16%) | 1.11 → **0.31 s** (−73%) | 42.0 → 39.3 s (−7%) |
| SGLang | 79.3 → 88.9 s (**+12%**) | 113.5 → 127.7 ms (+12%) | 4.11 → 6.12 s (+49%) | 41.6 → 46.7 s (+12%) |

TensorRT-LLM gains on every metric; vLLM gains mainly in streaming smoothness; SGLang gets worse
on every metric.

**Correction.** An earlier version of this note said chunking slows vLLM's decode by 29%. That
used the per-request median gap between streamed chunks (ITL p50), which rises because chunked
iterations are more frequent and smaller, not because tokens arrive more slowly. TPOT, which is
robust to chunk size, shows vLLM's decode improves (p95 −16%).

## 9. N-gram speculation helps vLLM, barely helps SGLang, hurts TensorRT-LLM
RQ3, max_num_seqs 64, E2E median, baseline → n-gram (k=3):

| Engine | chat_multiturn | mixed_real |
|---|---|---|
| vLLM | 17.9 → 14.5 s (**−19%**) | 12.0 → 10.5 s (**−12%**) |
| SGLang | 18.8 → 17.6 s (−6%) | 13.1 → 12.8 s (−3%) |
| TRT-LLM | 16.8 → 19.5 s (**+16%**) | 23.1 → 27.7 s (**+20%**) |

vLLM's draft acceptance is 34% (the only engine where it is measured). TensorRT-LLM's
loss is confounded: n-gram required disabling its overlap scheduler. A baseline with the
overlap scheduler off is needed to attribute it.

**Measurement trap.** With speculation, ITL p50 *rises* (SGLang 48.5 → 78.9 ms, TRT-LLM
47.6 → 89.1 ms) even though throughput improves, because accepted tokens arrive in
multi-token chunks. Per-chunk ITL is not comparable between speculative and
non-speculative runs. Report time per output token, (E2E − TTFT) / (tokens − 1), instead.

## 10. Failures: zero-token errors come from the model, timeouts from the engine
- "No tokens received" errors are near-identical across engines (vLLM 1,637, SGLang 1,639,
  TRT-LLM 1,796): a model/prompt-template issue (Mistral-NeMo), not an engine one.
- Timeouts differ 5×: TRT-LLM 3,037, vLLM 698, SGLang 534. Almost all of TRT-LLM's are
  first-token timeouts (§3).

## 11. Methodology findings worth reporting
1. **TTFT is not monotonic in arrival rate on finite traces.** Llama mixed_real: TTFT p95 is
   185 s at 2 req/s but 0.3 s at 16 req/s. At low rates the last arrivals land after long
   generations fill the KV pool; at high rates all 500 arrive (in 32 s) before it fills,
   and the cost moves to mid-stream stalls. Fixed-request-count benchmarks should report
   latency against time-in-run, or use duration-based traces.
2. **Observability differs as much as performance.** vLLM exposes preemptions, KV, queue
   and speculative acceptance live; SGLang exposes KV and queue (retractions only from
   0.5.x); TensorRT-LLM exposes none of these live in this harness. Operators cannot
   diagnose what they cannot see.
3. **Engine metrics with the same name are not the same metric** (prefix-cache hit rate, §7;
   ITL under speculation, §9).

## Threats specific to these comparisons
- TensorRT-LLM configuration (§3) and the overlap-scheduler confound (§9) must be resolved
  before engine-level claims about it are final.
- One run per configuration; no confidence intervals.
- Latency percentiles exclude timed-out requests, which flatters TensorRT-LLM most.
