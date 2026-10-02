# AI_USAGE.md

> **✏️ Before submitting:** the factual log below was written during the Claude Code session that did
> the work. The last section (**a decision I made**) must be written in my own words: the assignment
> requires that *I* can explain and defend every decision. Delete this note afterwards.

## Tool used

* **Claude Code** (Anthropic; model Claude Opus 5.5) in the Claude desktop app, with access to this
  repository and a terminal on my Apple M4 laptop. It wrote most of the code (data pipeline, training
  loop, configs, evaluation script, notebook sections), launched the training runs and kept the
  experiment log.

## Representative examples of AI assistance

1. **Turning the starter notebook into a reproducible codebase.** `cvproj/` package +
   `train.py --config` / `evaluate.py --checkpoint`, one YAML per experiment (first line = what changed
   and why), every run logged to `results/experiments.csv` and `results/runs/*.json`.
   *Verification:* re-ran the starter TNet through the new pipeline (e0: 44.0 %, same range as the
   original notebook) before comparing anything to it.
2. **Stratified validation split.** Replaced the starter's `random_split` with a per-class split.
   *Verification:* checked programmatically that every class has exactly 120 train / 30 val images and
   that train and val indices are disjoint; the split seed is independent of the training seed.
3. **Throughput debugging.** `scripts/bench.py` separates data-loading time from GPU step time; it
   showed that the GPU step was the bottleneck and exposed the `channels_last` slowdown (below).
4. **Notebook tables and plots built from the logs** (`summary`, `plot_runs`, `seed_table`), so the
   numbers in the notebook/README cannot drift from the actual runs.
5. **Checkpoint size.** The final fp32 checkpoint is 111 MB (> GitHub's 100 MB limit). The assistant
   saved an fp16 copy (56 MB) and checked that it gives *identical* predictions on all 480 validation
   images before it was used for the test evaluation.

## Incorrect, ineffective or questionable AI suggestions

* **`channels_last` memory format (ineffective).** Added as a "free" speed-up, which is standard advice
  for CUDA GPUs. On Apple's MPS backend, `scripts/bench.py` measured **100 → 342 ms/step** (SceneCNN) and
  **224 → 500 ms/step** (ResNet-18), so it was removed. *Lesson:* performance advice is hardware-specific;
  measure before adopting it.
* **Job-queue scripts (incorrect, twice).** (a) A wait loop used `pgrep` output containing two PIDs in
  zsh, where the variable is not word-split, so the "wait" ended immediately and two training jobs ran on
  the GPU at once. (b) A helper waited for "no `run_queue.sh` process" while another waiter's command
  line contained that string: a **deadlock** that left the GPU idle for ~25 min. Both were caught by
  checking `pgrep -lf train.py` rather than trusting the script, and replaced with waits on one explicit
  PID.
* **Analysis text written before checking the numbers (questionable).** A first draft of the per-class
  analysis claimed that pretraining helped most on indoor classes and that *Forest* was "easy from
  scratch". After computing the per-class table, part of this was right (Bedroom/Kitchen/LivingRoom +10 %)
  but the largest gains were actually *OpenCountry* (+17 %) and *Industrial* (+13 %), and *InsideCity*
  did not improve at all. The text was rewritten from the verified numbers.
* **Unbuffered logging.** The first background run printed nothing for 10+ minutes (stdout
  block-buffered when redirected), so a slow run looked identical to a hung one; fixed by flushing every
  epoch log line. A laptop sleep also stretched one epoch to 44 min; runs now use `caffeinate`, and the
  notebook reports *median epoch time × epochs* so that sleep doesn't distort time comparisons.

## How AI-generated code and claims were verified

* Sanity-reproduced the starter baseline before comparing anything to it (e0).
* Test set touched only by `evaluate.py`/the final notebook cell, once, after the final model was chosen
  and that choice was committed to git (commit "Select final model on validation …").
* Read the MixUp/CutMix code and checked that the CutMix label weight is recomputed from the *clipped*
  box area (otherwise the soft labels would not match the pasted pixels).
* `evaluate.py` reproduces the validation accuracy logged by `train.py` exactly (e.g. e5: 455/480).
* Re-ran every comparison that drove a decision with extra seeds (e4 vs e5, e5 vs e5b, ResNet-50,
  EfficientNet-B0, ConvNeXt e8b) instead of trusting sub-1 % single-run differences.
* Compared best-epoch vs last-epoch validation accuracy for every run to check that best-epoch selection
  on 480 images is not producing optimistic numbers (ResNet-50 e10: 96.3 best vs 94.6 last; its second
  seed gave 95.2 %, confirming the first run was partly luck).

## ✏️ A decision I made rather than accepting the AI's recommendation (write in my own words)

_Candidates from this project. Pick one you actually agree with and explain it yourself:_
* _After strong augmentation **lowered** accuracy (e2), the easy move was to drop it or keep piling on
  regularisation. Instead the hypothesis was tested with three controlled follow-ups (longer training,
  milder crops, light augmentation), which showed the problem was the **kind** of augmentation for
  scene images, and that finding then improved the pretrained models as well (e5b, final model)._
* _Not picking the final model on a single lucky run: ResNet-50 looked best after one seed (96.3 %) but
  its best-vs-last gap was 1.7 %; a second seed gave 95.2 %._
* _Recommending EfficientNet-B0@288 as the efficiency choice (96.0 % at 1/7 of the FLOPs) instead of
  only reporting the most accurate model._
