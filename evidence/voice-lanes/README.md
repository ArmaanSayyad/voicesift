# Audio-only curation evaluation evidence

This experiment tests six clip-level requirements through the experimental `audio_goals` judge and offline export harness. It does not enable these goals in the interruption-only web app or measure complete conversations.

Pinned sources:

- [PolyAI/MINDS-14](https://huggingface.co/datasets/PolyAI/minds14), revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`, US-English train partition, all 563 clips; CC BY 4.0.
- [DynamicSuperb VocalSound subset](https://huggingface.co/datasets/DynamicSuperb/VocalSoundRecognition_VocalSound), revision `beb7fe456e01f1a9959daae8dd507fa7790f4b62`, test partition, 50 per source category (300 total). [Original VocalSound release and license](https://github.com/YuanGongND/vocalsound): CC BY-SA 4.0. This is a benchmark mirror subset, not the full official VocalSound test split.
- [xbgoose RAVDESS mirror](https://huggingface.co/datasets/xbgoose/ravdess), revision `a4a6c53ad083c4f16e92d1625e99113effe7569d`, train partition, 30 per emotion (240 total). [Original source](https://zenodo.org/records/1188976): Livingstone and Russo (2018), CC BY-NC-SA 4.0; commercial use has separate licensing. These are acted expressions, not measurements of actual psychological states.

Source cards are locally cached. The source parquet downloads total 582,466,480 bytes. Audio is decoded and resampled to 16 kHz PCM16 WAV without trimming, retaining mono/stereo channels. The frozen sample contains 1,103 unique source-audio hashes and about 116.31 minutes of audio. All 24 RAVDESS actors appear; mirror emotion/actor metadata agrees with the documented source filename codes for all 1,440 mirror records. This is a metadata check, not an independent audio relabeling.

The model sees audio bytes and two plain-language requirements per lane, nothing else. Source filenames, IDs, transcripts, labels, categories, actor attributes, and dataset instructions are not included. The requirements ask about main intent/dominant vocal event/dominant performed expression to match the single-label source tasks. Context-specific labels and real conversational mixtures may differ.

`protocol.json` freezes source revisions/file hashes, request templates, model, and seed before inference. `selection.json` contains source row indices, labels, audio hashes, and reference decisions. Predictions are scored against these existing labels; no model-generated gold labels or prompt tuning are used. The prompt is zero-shot. These public datasets may have appeared in model training; contamination is unknown. Dataset partitions named train are used for evaluation only: no training or fine-tuning is performed here.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/evaluate_voice_lanes.py --run
# Optional, once: recover only transport failures, preserving original attempts.
.venv/bin/python scripts/evaluate_voice_lanes.py --retry-transport-errors
```

Downloads and audio remain in ignored `artifacts/voice-lanes/`. Existing predictions are cached by exact request hash. An additional retry directory preserves the original transport failures and records the uniform recovery rule; completed answers are never selectively rerun. Selected clip ZIPs are written to `artifacts/voice-lanes/exports/`, with source references and explicit machine-selected/unverified status. Audio is not committed to Git.

Accuracy counts only correct yes/no answers divided by all evaluated examples. Unclear responses and remaining errors count as incorrect, including on negative examples. Precision is true positives divided by all yes decisions; recall includes undecided/failed positives in its denominator. Tables also report always-reject accuracy, because high negative prevalence can otherwise inflate apparent performance. Scores describe these sampled category mixtures; they do not establish deployment prevalence or guaranteed export purity.
