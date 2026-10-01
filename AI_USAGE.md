# AI_USAGE.md

> **Note to self before submitting:** the factual log below was written while Claude Code was doing
> the work. The sections marked **✏️ (my words)** must be rewritten in my own words — the assignment
> requires that *I* can defend every decision, and these parts are about my judgement, not the tool's.

## Tool used

* **Claude Code** (Anthropic, model Claude Opus 5.5) running in the Claude desktop app with access to
  this repository and a terminal. It wrote most of the boilerplate (data pipeline, training loop,
  configs, evaluation script, notebook cells) and launched the training runs on my Apple M4 laptop.

## Representative examples of AI assistance

1. **Pipeline scaffolding.** Turned the starter notebook into a small package (`cvproj/`) plus
   `train.py --config` / `evaluate.py --checkpoint` entry points, with every experiment captured as a
   YAML file in `configs/` and logged to `results/experiments.csv` + `results/runs/*.json`.
   *Verification:* re-ran the starter TNet through the new pipeline (e0) and checked that it lands in
   the same ~40–45 % range as the original notebook before trusting any later number.
2. **Stratified validation split.** Replaced the starter's `random_split` with a per-class split
   (30 val images/class). *Verification:* checked the class counts of the split (exactly 120/30 per
   class) and that the split is identical across runs (fixed seed, independent of the training seed).
3. **Throughput debugging.** The first from-scratch run was ~4× slower than expected. The assistant
   wrote `scripts/bench.py`, which separates data-loading time from GPU step time and compares memory
   formats; this showed the GPU step, not the data loader, was the bottleneck (see below).
4. **Experiment bookkeeping and plots.** Helper functions in the notebook (`summary`, `plot_runs`)
   that build the experiment tables and train/val curves directly from the logged JSON files, so the
   report numbers cannot drift from the actual runs.

## An incorrect / ineffective AI suggestion

* **`channels_last` memory format.** The assistant added `model.to(memory_format=torch.channels_last)`
  as a "free" speed-up — a common recommendation for CUDA GPUs with tensor cores. On the Apple MPS
  backend it is the opposite: `scripts/bench.py` measured **100 ms → 342 ms per step** for SceneCNN and
  **224 ms → 500 ms** for ResNet-18. It was removed (commit "Drop channels_last …").
  *Lesson:* performance advice is hardware-specific; measure before adopting it.
* **Unbuffered logging.** The first background run printed nothing for 10+ minutes because stdout was
  block-buffered when redirected to a file, which made a slow run indistinguishable from a hung run.
  Fixed by flushing every epoch log line.

## How AI-generated code was verified

* Sanity-reproduced the starter baseline before comparing anything to it (e0).
* Checked split sizes/class balance and that the test set is only touched by `evaluate.py`.
* Read the MixUp/CutMix implementation and checked that the CutMix mixing weight is recomputed from
  the *clipped* box area (otherwise the label weights would not match the pasted pixels).
* Compared best-epoch vs last-epoch validation accuracy for every run to make sure model selection
  on the 480-image val set was not itself producing optimistic numbers.

## ✏️ (my words) A decision I made rather than accepting the AI's recommendation

_TODO — describe one decision in my own words. Candidates from this project:_
* _whether to keep pushing augmentation after it hurt in e2, or to diagnose it (underfitting, still
  improving at the last epoch) with controlled follow-ups (e2b light aug, e2c longer schedule);_
* _choosing the final model on the accuracy/efficiency trade-off rather than simply the largest model;_
* _deciding to re-run the closest comparisons with extra seeds before believing a <1 % difference._
