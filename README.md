# Muse Glimmer 30B on 2× Intel Arc Pro B70 — the llm-scaler Cookbook

A complete, reproducible port of **Muse Glimmer 30B** (BF16 + official draft model) onto a
dual Intel Arc Pro B70 rig using Intel's **llm-scaler** vLLM image — with online FP8
quantization, dual-GPU tensor parallelism, and the official **dflash** speculative-decoding
stack. Every number in this document is measured on real hardware at **stock 180 W power
limits** with BetterBench and stream-timing harnesses, all included.

**The short answer to "is it near 200 tok/s?":** on summarize-heavy, highly-predictable
text the drafter accepts ~10 tokens per verify step and decode hits **222 tok/s median**.
Across all eight real-workload categories the median is **100 tok/s**. Free-form prose
floors at ~55. All three numbers are true; the workload decides which one you live in.

| | |
|---|---|
| Hardware | 2× Intel Arc Pro B70 (32 GB, dual-slot), Fedora, kernel 7.2.9-201 (see §Kernel) |
| Engine | `intel/llm-scaler-vllm:0.31.0-b1-experimental` (vLLM 0.31 + XPU, CUDA-over-ZeGraph shim) |
| Model | Muse Glimmer 30B BF16 (~58 GB) + official assistant drafter, online FP8 |
| Spec decode | dflash, depth 15, draft TP2 |
| Context | 131,072 native — **needle-verified at 128.9K** · 196K over-trust experimental (see §5) |
| Measured | 100 tok/s all-category median · 222 summarization · ~1,330 tok/s prefill (flat to 64K) |

---

## 1. Five deltas between "works on paper" and "boots on B70"

1. **`VLLM_XPU_ENABLE_VLLM_EXTENSION=1`** — vLLM 0.31's XPU platform probe raises
   `NotImplementedError` without it. This is the shim gate for the CUDA-flavored stack.
2. **`VLLM_USE_V1=0`** — the v1 engine path has no XPU support in this image; v0 required.
3. **Online FP8 + dual TP.** BF16 weights are ~58 GB — exceeds one card. 32+32 GB with
   online quantization is the operating point (`--quantization fp8 --tensor-parallel-size 2`).
4. **`--gpu-memory-utilization 0.90` per card** — verified load-stable (30+ min full
   benchmark suites, zero TTM evictions). Do not raise on the validated kernel; if you are
   forced onto a TTM-regressed kernel, ladder from 0.85 (+0.02 per step; 0.92 is a reported
   known-good) — see §Kernel.
5. **Spec decode is config, not luck:**
   ```json
   {"model":"/models/<drafter>","method":"dflash",
    "num_speculative_tokens":15,"draft_tensor_parallel_size":2,
    "max_model_len":131072}
   ```

## 2. Launch recipe

```bash
docker run -d --name glimmer --device /dev/dri \
  -v /dev/shm:/dev/shm \
  -v $HOME/models/muse-glimmer-30b-bf16:/models/muse-glimmer-30b-bf16 \
  -v $HOME/models/muse-glimmer-30b-assistant-official:/models/muse-glimmer-30b-assistant-official \
  -p 8210:8200 -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_XPU_ENABLE_VLLM_EXTENSION=1 -e VLLM_USE_V1=0 \
  -e MODEL_PATH=/models/muse-glimmer-30b-bf16 -e SERVED_NAME=glimmer -e PORT=8200 \
  -e QUANTIZATION=fp8 -e TP=2 -e GPU_UTIL=0.90 -e MAX_MODEL_LEN=131072 \
  -e KV_CACHE_DTYPE=fp8 \
  -e SPECULATIVE_CONFIG='{"model":"/models/muse-glimmer-30b-assistant-official","method":"dflash","num_speculative_tokens":15,"draft_tensor_parallel_size":2,"max_model_len":131072}' \
  intel/llm-scaler-vllm:0.31.0-b1-experimental
```

First boot is compile+capture (~2.5 min: torch.compile, dflash target-verify capture,
drafter graph capture). `health=200` is your ready signal.

## 3. Bench results (all at stock 180 W)

### 3.1 BetterBench quick suite — baseline (d15, BF16 KV)

**Canonical numbers are the native-on-DUT pair** (run on the B70 machine itself, FP8-KV first/cold, bf16-KV second/warm): runs `20261028-123123-glimmer-keeper-d15-fp8kv-180w` and `20261028-131750-glimmer-baseline-bf16kv-180w`. Keeper medians: summarization 151.7, JSON 122.0, math 115.6, file-edit 113.7, code 83.4, prose 80.9, reasoning 75.6, chat 68.9 — **ALL-median 98.6 t/s**. Paired compare: every category noise (KV dtype decode-neutral; kept for capacity). Earlier client-generated quick run `20261028-093646` (ALL 100.2, summarization 222.7) is a sample-variance illustration — native 5-pass medians show summarization is a ~150-190 high-variance lane, not a stable 222.

> **Reading the HTML reports:** their environment metadata block captures the *benchmark
> client* machine (a Ryzen 9 9950X host with its own 2x Radeon AI PRO R9700 pair, used as
> the load generator). **The device under test in every number here is the 2x Intel Arc
> Pro B70 machine** (Core Ultra 7 270K Plus, Fedora, kernel 7.2.9) — all decode/prefill
> figures come from the server-side engine logs on the B70 rig, not the client.
, [charted report (live, native)](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-keeper-native-fp8kv.html).

| category | median decode tok/s |
|---|--:|
| summarization | **222.7** |
| json | 119.6 |
| math | 107.7 |
| reasoning | 100.1 |
| file_edit | 97.7 |
| code | 95.0 |
| prose | 75.3 |
| chat | 74.9 |
| **all categories** | **100.2** |

- **Prefill: ~1,330 tok/s, flat from 2K to 64K tokens** (TTFT p50 1.25 s → 35.7 s linear).
- **Concurrency:** aggregate ~85 tok/s at 1→8 streams; decode median per-request holds
  96–110 tok/s; TTFT queues linearly.
- 36/40 decode runs hit the quick preset's `max_tokens` cap. **Caveat: Glimmer is a reasoning model** — most quick-preset runs spend the entire token budget on reasoning tokens and never reach an answer token (see the report's reasoning/answer split: 87-100% reasoning share). These decode numbers are *generation* speed; time-to-first-answer at short budgets is lower, and a raised-`max_tokens` full run is the right follow-up.
- [LoRA serving on B70](recipes/lora-serving.md) — base+adapter per-request routing on vLLM 0.31; torch sidecars belong on RDNA3

  steady-state; a rigorous full run raises the cap.

### 3.2 The 200 tok/s question, mechanically

Decode ≈ `target_per_step × acceptance_length`. On repetitive/high-entropy-anticipation
text (summarization copy lanes, counting, JSON echoes) dflash-d15 accepts **~10 tokens per
verify step** → 200+. On novel prose acceptance collapses to 2–3 → ~55. The drafter is
free (30 GB drafter at depth-15 costs one verify step); the target verification cadence is
the constant.

### 3.3 Draft depth A/B: depth-4 vs depth-15 (keeper: 15)

| metric | d15 | d4 |
|---|--:|--:|
| prose tok/s | 54 | 56.1 (+4%) |
| counting tok/s | 112.6 | 86.6 (−23%) |
| mean acceptance length | ~10 | 3.2–3.4 |
| draft acceptance rate | 34–40% | 56–60% |

Raising acceptance by shortening the draft does **not** raise prose throughput: prose is
target-verify-bound, not drafter-bound. Depth 15 ships.

### 3.4 Optimization A/B (paired BetterBench comparisons)

| change | verdict | keep? |
|---|---|---|
| `KV_CACHE_DTYPE=fp8` | decode-neutral (all categories noise); halves KV footprint | **yes** — enables the 196K lane |
| async scheduling (`--async-scheduling`) | all-noise, mixed sign — no benefit at batch-1 | no |

Reports: [`reports/betterbench-fp8kv.html`](reports/betterbench-fp8kv.html),
[`reports/betterbench-async-sched.html`](reports/betterbench-async-sched.html).

## 4. Kernel: the single biggest performance variable

A Fedora kernel bump (7.2.11 series) regressed the in-kernel Intel `xe` driver:
**decode 5–15 tok/s with clocks, power, and utilization telemetry all looking perfect.**
Same image, same weights; booting the validated kernel (7.2.9-201) restored 100+ tok/s.
Mechanism, independently confirmed upstream: the 7.2.11rc1 TTM change evicts VRAM pages
to system RAM far earlier — per-token page thrash over PCIe. Fedora subsequently pulled
the build.

Rules that came out of it:

- **Symptom triage:** "slow decode + full clocks + 100% compute" ⇒ `uname -r` before
  touching anything else.
- Pin the validated kernel, remove the bad one, install a boot guard that banners loudly
  on drift (`/etc/validated-kernel` + a systemd oneshot).
- If a TTM-aggressive kernel ever becomes unavoidable:
  `--gpu-memory-utilization` ladder from **0.85**, +0.02 per step with a decode bench at
  each step (0.92 reported stable at full speed). On the validated kernel, 0.90 is
  load-verified — do not lower.
- Kernel A/B protocol for new releases: `grub-reboot` the candidate for exactly one boot,
  run the decode ladder, reboot back to the pinned kernel. Never test-drive kernels on
  your production default entry.

## 5. Long context: what 131K really delivers, and the YaRN trap

Native `max_position_embeddings` is 131,072. We measured the real end of that window with a
needle test (`scripts/needle196k.py`): ~129K tokens of numbered filler, unique facts at
50% and 92% depth, temperature 0, exact-match scoring.

**Result — the full native window is real:**

| depth | context (measured tokens) | needle@50% | needle@92% | prefill |
|---|--:|---|---|--:|
| 100K | 101,648 | **pass** | **pass** | 91 s (1,116 t/s cold) |
| native max | 128,852 | **pass** | **pass** | 25 s |
| over limit | >131,072 | clean HTTP 400, engine survives | — | — |

Bonus: **prefix caching works on this stack**. The second needle pass reuses the identical
128K prefix and pays `TTFT = 0.56 s` instead of ~2 minutes — append-only prompts (chat
history, document Q&A loops) get effectively free re-prefill.

**Negative result — do not YaRN this model.** Stretching past 131K with
`--hf-overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 1.5}}'` boots cleanly,
serves `max_model_len=196,608`, and **collapses instruction-following from token zero** —
even a 40-token prompt echoes its own input into the reasoning field until
`finish_reason: length`. Glimmer is a hybrid sliding-window attention model
(window 2,048 on most layers, 2 KV heads); rope-scaling the full stack corrupts
attention geometry everywhere. Plain over-trust (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
no rope surgery) answers correctly at short depth — a 196K-configured lane then
**native-crashed (SYCL) during a 137K prefill**. Practical ceiling today: the native
131K window, fully usable. The >131K story on XPU is an open investigation, not a recipe.

Prefill math for planning: ~1,100–1,330 t/s ⇒ a full 131K prompt pays ~100–120 s TTFT
cold. Long-context here is an offline document-analysis cadence, not chat speed.

## 6. Power is a benchmark parameter

Both cards are held at the stock 180 W limit (shared-circuit environment; sustained
multi-hundred-watt GPU draw is a breaker trip, not a tuning knob). A historical
measurement at a 275 W limit showed ~171 tok/s on the counting lane (vs 112 at 180 W) —
**power headroom moves decode ~1.5×, second only to the kernel.** Report your power
limit in every B70 benchmark you publish, or your numbers are fiction.

## 7. Operational pitfalls (each one cost an hour)

1. **Quote your systemd drop-ins.** A `SPECULATIVE_CONFIG` line with literal `\n` and
   unescaped quotes crash-loops the unit at "activating" with a dead container.
   `systemd-analyze verify` after every unit edit.
2. **Co-tenant eviction.** Anything GPU-resident (forecast models, engines, other
   inference servers) contends for VRAM and power. The engine launcher must stop
   co-tenants before start and restore them on stop.
3. **Tunnel hygiene.** Long-lived SSH tunnels need `ServerAliveInterval=30`,
   `ServerAliveCountMax=3`, and a pre-start `fuser -k <port>/tcp` — a zombie ssh holding
   the local bind makes systemd restart-loop. (Two benchmark suites died of this before
   we found it; always check the endpoint by hand before trusting a zero-filled report.)
4. **Bench the workload, not a prompt.** "Prose" spans 55→100 tok/s depending on
   predictability (see §3.2). Publish the category table, not a headline.

## 8. Reproduce this

- [`scripts/bench_glim.py`](scripts/bench_glim.py) — stream-timed 512-token decode probe
- [`scripts/bench_glim2.py`](scripts/bench_glim2.py) — workload-sensitive pair (counting vs prose)
- [`scripts/needle196k.py`](scripts/needle196k.py) — depth-50/92 needle harness (sized by true tokens; reasoning-delta aware)
- **Charted reports (live, native on DUT):** [FP8-KV keeper](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-keeper-native-fp8kv.html) . [bf16-KV baseline](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-baseline-native-bf16kv.html) — fingerprint captured on the B70 machine. Client-generated A/Bs (banner-labeled): [fp8kv](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-fp8kv.html) . [async](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-async-sched.html) . [batched A/B seqs8 — ALL 105.9, conc 8-wide 223 agg t/s](https://munikor.github.io/glimmer-b70-cookbook/reports/betterbench-batched-seqs8-180W.html)
- Bench tool: [BetterBench](https://github.com/GGZ14/BetterBench) (Apache-2.0; ran 192 requests/config across 8 categories)

## License & disclaimers

MIT — see [LICENSE](LICENSE). The benchmark harness, model weights, and engine image have
their own licenses (BetterBench: Apache-2.0; the model and Intel images: see their
upstream terms — check them before commercial use of *their* artifacts; this repo's
content — text, scripts, reports — is MIT).

This is a hobbyist rig writeup, not vendor guidance. Results are specific to this exact
stack (image tag, kernel, power limit, cooling, motherboard, tenancy); your mileage will
differ, intentionally. No warranty, no support obligations, no affiliation with Intel,
Meta, or the model authors. Verify everything on your own hardware before trusting it
with production, power budgets, or your weekend.
