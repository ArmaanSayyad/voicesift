# Feasibility decision — 29 September 2026

> Historical development document. Its proposed scope and implementation state may be superseded. See [VoiceSift’s current purpose and capabilities](../../docs/README.md#project-scope-and-current-capabilities).

**Ready to begin the main local cascade workbench implementation.** The deterministic simulator and component feasibility checks are complete. This is not a claim that the live workbench is built or that its behavioral evaluation has passed.

Use Parakeet → local Qwen → Kokoro as the primary cascade, with Silero VAD and explicit versioned repair state. Build the ordinary cascade (C0) and repair-aware cascade (C1) with identical components, cancellation plumbing and evaluation inputs. Keep Laya as an optional offline experimental critic. Keep Moshi Q4 as an offline native-speech comparator until a separate live performance gate passes. Neither optional component should block the core demonstration.

## What actually ran

Machine: Apple M6, 32 GiB unified memory, macOS 27.0.1, Python 3.12.14. The speech models use the Mac's integrated Metal acceleration; Laya used CPU with four Torch threads. No remote/rented GPU or paid API was used. Ollama `qwen3.5:9b` was already installed. Model downloads and inference were local/free; this does not establish unrestricted redistribution rights for upstream models or datasets.

Exact installed package snapshots are in `requirements/`. Key versions: mlx-audio 0.5.7 with MLX 0.32.3; moshi-mlx 0.3.0 with MLX 0.26.5 and rustymimi 0.4.1; Silero VAD 6.2.3; Laya source revision `6d942c92081fbc139e736bbd9ac0023223c29b7f`, model revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Other model revisions are in `evidence/models.json`. Environments are separated to preserve these incompatible runtime versions.

The first audio attempt failed because a cache snapshot directory name was inferred as a model architecture; explicit `model_type` fixed it. The next attempt exposed the missing `en_core_web_sm` language model; installing version 3.8.0 fixed that. Both failures are retained locally in ignored artifacts. The final audio probe completed. Moshi's older Q4 snapshot lacks `config.json`, so its probe uses the package's `config_v0_1()` configuration, following its legacy loader pattern.

These are small, purpose-built development checks, not held-out benchmarks. Tests used generated English speech from one voice. No human microphone capture, physical playback, echo cancellation, accent/noise robustness, external dataset evaluation or perceptual listening study occurred. Timing samples are observations on a normal desktop, not controlled performance guarantees. Early independent Laya/Ollama probes overlapped; the successful audio probe had Ollama resident but no intentionally concurrent model generation. Moshi was run after unloading that Ollama model. The resident cascade probe ran afterward.

## Measured component results

| Component | Observation | Decision |
|---|---|---|
| Kokoro | First call 7.30 s for 3.125 s of audio. Five warm calls: 0.159–0.536 s for 3.175–10.3 s of audio; real-time factors approximately 0.047–0.052. | Feasible after warm-up. This API yielded a whole segment for each tested phrase, not token-level incremental PCM. |
| Parakeet | First full-utterance call 1.242 s; five warm calls 0.039–0.138 s. Negation/correction phrases were transcribed intelligibly; the first phrase became “Planned dinner” rather than “Plan dinner.” | Feasible. No claim of perfect transcription or human-speech WER. |
| Prefix transcription | At 0.5 s: “Do not”; 1.0 s: “Do not book Friday.”; 2.0 s: “…I said Saturday.”; 3.0 s: “…at eight.” | Demonstrates why revision availability must be logged. These are repeated prefix decodes, not a native streaming ASR benchmark. |
| Silence | Two seconds of zero-valued audio produced an empty ASR transcript and no Silero speech spans. | Basic negative control passes; not an environmental-noise test. |
| Silero ONNX | Two padded synthetic clips processed in 10–18 ms, returning speech spans. | Runtime feasible. Boundaries lack human ground truth and are not precision/recall or onset-latency measurements. |
| Qwen via Ollama | Six calls across three structured corrections and two repetitions preserved the requested/unmentioned fields in all six. | Feasible for schema-constrained proposals; application validation is still required. Tiny constructed test, not a repair-success estimate. |
| Laya CPU | 1/6 judgments matched the constructed labels; inference 91–172 ms. Peak process RSS about 2.81 GiB. | Runtime passes; this schema/prompt configuration fails the semantic smoke check. Do not use it as an oracle or live decision authority. |

Laya's result is configuration-specific and small-sample evidence, not a general model capability estimate. For example, it labeled a Friday plan after a Saturday correction as “incorporated,” including one case with output probability 0.7684. Those values are model outputs, not validated confidence. Any future adoption needs schema/prompt experiments, label-order controls and held-out calibration. We should not optimize it against the final demonstration cases and then present those same cases as evaluation.

## Resident cascade probe

Parakeet and Kokoro were loaded in the same Python process while Ollama served Qwen locally. The probe transcribed a completed synthetic correction (“Actually Saturday. Keep the time and party size.”), requested a schema-constrained state update, verbalized the result and synthesized it. All three repetitions yielded Saturday / 19:00 / two people.

| Run | ASR | LLM | TTS | First generated audio after full input was available |
|---|---:|---:|---:|---:|
| First | 2.204 s | 1.490 s | 3.870 s | 7.549 s |
| Warm 1 | 0.108 s | 1.416 s | 0.270 s | 1.785 s |
| Warm 2 | 0.062 s | 1.431 s | 0.224 s | 1.710 s |

This establishes a functioning resident component chain. It does **not** include endpoint detection, microphone input time, network transport to a frontend, output queueing, AudioWorklet scheduling, DAC latency or sustained simultaneous ASR/TTS. It is therefore not a measured end-to-end conversational response time. Two warm samples are insufficient for percentile latency claims. Model output phonemization emitted a word-count warning for the time-formatted response; add a deterministic time/number verbalizer and test spoken output during implementation. Process RSS in the evidence excludes the separate Ollama server and is not total system memory consumption.

## Moshi Q4 result

The native probe fed one synthetic utterance plus silence through Mimi encoding, Moshi generation and Mimi decoding at 24 kHz / 1,920 samples per step. It ran offline without input pacing or speaker playback. The 30 s and 60 s runs used new generator/codec state and the same resident model.

| Input duration | Compute wall time | Real-time factor | Frame p95 | Frames above 80 ms |
|---|---:|---:|---:|---:|
| 30 s | 27.543 s | 0.918 | 91.8 ms | 64 / 375 |
| 60 s | 72.079 s | 1.201 | 273.7 ms | 164 / 750 |

Peak process RSS was approximately 5.48 GiB; MLX peak approximately 5.32 GiB. These counters are overlapping measures and must not be added. Output durations were 29.92 s and 59.92 s; account for the codec/generator output delay when aligning streams. The second run exceeded real-time throughput, so we cannot promise stable live operation. Offline processing remains feasible. Generated text was conversational but largely unrelated to the reservation request; no successful native task repair was demonstrated. Further conditioning/interaction testing and human listening are necessary before including it in a behavioral comparison. No Q8 or PersonaPlex inference was attempted.

## Deterministic harness delivered

`src/repair_bench/core.py` provides a stable virtual scheduler, hash-linked event log with causal parent validation, bounded PCM queue, partial consumption ledger, epoch cancellation and rejection of late obsolete audio. It also provides availability-aware transcript revisions, state version/epoch checks, top-level evaluator-field filtering, stateful sample-rate conversion, interval-union overlap and censored yield-delay measurements.

The fixture separates input, generated and simulated rendered audio. At exactly sample 7,200 it cancels the original stream; the late epoch-zero chunk is discarded; a new epoch-one repair segment begins at 12,000. It records 107 events. No obsolete samples are consumed after cancellation. A cancellation cannot retract a block the simulated renderer has already consumed; callers must split render intervals at intervention boundaries. That contract is tested at an arbitrary sample offset as well as the main fixture's block boundary.

**16 tests pass**, including Hypothesis-generated packet boundaries, stable scheduling and partial cancellation. Other tests cover byte-identical artifact replay, audio/event corruption and truncation detection, availability leakage, stale state proposals, preservation of unmentioned fields, backpressure, impulse alignment at three sample-rate pairs and ineligible/right-censored yield metrics. The manifest verifies all four artifact hashes and the final event hash. The checked fixture is synthetic tones, not a behavioral conversation. The local model outputs are deliberately outside the determinism guarantee.

This is a reusable simulator foundation, not a production event system: event persistence occurs at the end of the fixture rather than durably on each append, there are no cross-process clock mappings, and the state wrapper does not enforce a domain schema. The orchestrator-facing public scenario includes future intervention scheduling and must never be passed wholesale into a model prompt. Implement a separate time-filtered controller view before wiring adapters.

## Implementation sequence and remaining gates

1. **Build one runnable vertical slice:** local backend, resident ASR/LLM/TTS workers, explicit schemas and deterministic time/number verbalization. Warm components before a run, display readiness and record versions. Keep model workers isolated by environment; log cancellations with epoch IDs. Do not run Moshi alongside the primary cascade.
2. **Implement browser audio and clocks:** AudioWorklet input/output, sample-rate negotiation, clock mapping, bounded queues, rendered-sample acknowledgments, cancellation inside partially consumed buffers, disconnect/error states and replay. Validate physical playback/loopback latency separately; do not relabel generation times as heard speech.
3. **Implement C0/C1 fairly:** same audio pipeline, ASR revisions, prompts/budgets where applicable and interruption plumbing; C1 adds explicit state repair. Isolate future interventions and evaluator labels from controller-visible context. Treat premature silence, partial corrections, backchannels and stale inference results as explicit cases.
4. **Add the consented recording workflow:** user-triggered start/stop, local preview, discard/re-record, metadata and deletion controls. Request microphone permission only in that workflow. Capture varied human repair phrases; hold out some recordings/scenarios from tuning.
5. **Run the live integration gate:** sustained audio streaming, ASR/TTS contention, underruns/overruns, queue growth, disconnects, cancellation timing and repeated interventions. Measure at least several complete multi-turn sessions and retain failures. Set latency targets against measured hardware behavior rather than the synthetic component timings above.
6. **Add the workbench UI and evaluation:** synchronized waveforms/events/transcript revisions/state diffs, paired C0/C1 runs, semantic state assertions, overlap/yield metrics with eligibility/censoring, annotated failures and export. Run small human-recorded evaluations before any efficacy claims. Optional bounded YODAS samples can test external ASR/VAD robustness; they are not labeled conversational-repair ground truth.
7. **Optional research adapters:** Moshi offline runs with explicit alignment and generation assumptions; Laya advisory results with prompt/schema provenance. Promote either only after its specific remaining gate passes.

No further user decision is needed to start this scoped implementation. The default is a fully local primary cascade, optional experimental comparators, and no paid API use. Current evidence is sufficient to proceed with implementation, but not to claim a live interview demo is already reliable.

## Evidence and upstream references

Full measured JSON is in `evidence/`; scripts reproduce the probes and write ignored local artifacts. Upstream implementations/model pages: [Moshi](https://github.com/kyutai-labs/moshi), [Moshi Q4](https://huggingface.co/kyutai/moshika-mlx-q4), [mlx-audio](https://github.com/Blaizzy/mlx-audio), [Parakeet MLX](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3), [Kokoro MLX](https://huggingface.co/mlx-community/Kokoro-82M-bf16), [Silero](https://github.com/snakers4/silero-vad), [Laya](https://github.com/NandhaKishorM/laya). Numerical conclusions above come from the saved local probes, not upstream benchmark claims.
