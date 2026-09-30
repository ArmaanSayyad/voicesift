# Product direction checkpoint — 29 September 2026

> Historical development document. Its proposed scope and implementation state may be superseded. See [VoiceSift’s current purpose and capabilities](../../docs/README.md#project-scope-and-current-capabilities).

The interview goal is to demonstrate useful technical judgment for a team building speech-to-speech models. The user's earlier account says Velvet is moving toward its own voice models; we have not verified its internal stack, dataset needs, model endpoints or priorities. A custom agent scaffold is not automatically useful to a model research team, and improving a cascaded agent with an external state reducer does not establish improvement of an end-to-end speech model.

## Reuse before extending

- Pipecat Evals already supports scripted/simulated scenarios, audio mode, timed interruptions, transcripts/events and local speech/judging. Its documentation says it exercises 100+ example agents before releases. This is direct overlap, not merely an adjacent orchestration library: https://docs.pipecat.ai/pipecat/evals/overview
- Full-Duplex-Bench v2 already evaluates correction and entity tracking: https://aclanthology.org/2026.acl-short.4/
- Tau-Voice evaluates task outcomes and full-duplex interaction; its documented voice setup includes hosted provider dependencies, so zero-cost local operation is not assumed: https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/voice/README.md
- LiveKit Agents provides an established voice runtime with testing infrastructure; distinguish open-source components from hosted debugging/simulation products: https://github.com/livekit/agents and https://docs.livekit.io/testing/overview/

Keep the deterministic harness and initial evidence slice as reusable development assets. Before further custom live transport development, check whether a repair-focused extension on an existing harness meets the actual requirements. Do not claim the broad workbench concept or using a classifier for evaluation is novel.

## Most promising narrow Laya experiment

Start with an **offline repair-event miner**: classify a short transcript context and a candidate user turn as revising/retracting an earlier request versus other/uncertain, then preserve the surrounding turns and audio references for human review. Extend to response-quality judgments only if separately validated. Typed output is useful for machine-readable corpus metadata; it is not a correctness guarantee.

Potential value: reduce review effort to find correction-rich conversations; construct a balanced held-out evaluation slice; identify candidate post-training examples while retaining native audio and overlap. Utility depends on the team's corpus and current workflow. It is a hypothesis, not an established result.

Laya cannot hear prosody, overlap or speech boundaries. Existing transcripts or an ASR stage are required; transcription costs and errors count toward total utility. Do not infer that a transcript's 'yeah' is a backchannel without acoustic/context evidence. Do not cut independent speech segments and claim the resulting examples preserve full-duplex timing.

Use a small human-labeled pilot, then separate development/calibration and held-out conversations. Compare keyword search, a majority baseline, Laya and the available local Qwen model. Evaluate precision among top-ranked review candidates, recall on the audited population, coverage, wall time and review minutes per confirmed useful example. Include implicit corrections, negations, new requests that resemble corrections, quoted speech and ASR errors. Split by source conversation/template so near-duplicates do not leak.

Admission is empirical: Laya should beat the simpler baseline at the chosen review budget, or have a useful accuracy/compute tradeoff against the local LLM. Never auto-delete data, auto-label preference pairs as ground truth or use this initial classifier as an RL reward. A 1/6 result on our earlier response-repair judgment schema is evidence against that configuration, not a measurement of repair-event detection. The latter remains untested.

Its current README explicitly characterizes Laya as a base to specialize and warns about raw probability calibration: https://github.com/NandhaKishorM/laya#fine-tuning

PersonaPlex uses synthetic dialogue data and additionally trains the released checkpoint on real conversational data. This supports the relevance of conversational data workflows, but does not prove that our proposed miner improves a speech model: https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf

The useful deliverable would be a small auditable data/benchmark extension and measured results. Laya earns inclusion only if it helps; the speech team's adoption and downstream model gains remain unproven.
