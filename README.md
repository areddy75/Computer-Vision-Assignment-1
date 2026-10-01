# ITCS 6169/8169 Assignment 1 — The CNN Challenge (16-class scene recognition)

Small-data scene classification (2,400 training images, 16 classes, 256×256 greyscale) with
convolutional neural networks. This repository contains the code, configs, logs and the final
checkpoint behind my report.

<!-- RESULTS:START -->
**Final result:** _filled in after the final evaluation_
<!-- RESULTS:END -->

## Repository layout

```text
train.py                 # train one experiment from a YAML config (never touches the test set)
evaluate.py              # val + test accuracy, per-class accuracy, confusion matrix, unlabelled-folder predictions
cvproj/
  data.py                # stratified train/val split, in-memory dataset, augmentation pipelines
  models.py              # TNet (starter), SceneCNN (own design), torchvision CNNs (ResNet, ConvNeXt, EfficientNet, MobileNet)
  engine.py              # training loop (AdamW, warm-up + cosine, label smoothing, MixUp/CutMix), evaluation, flip-TTA
  utils.py               # seeding, device selection
configs/                 # one YAML per experiment; first line says what changed and why; best.yaml = final model
results/
  experiments.csv        # one row per training run (written by train.py)
  runs/*.json            # full per-epoch history + config for every run
  logs/*.log             # raw training logs
  eval/                  # evaluate.py outputs (reports, confusion matrices, test2 predictions)
scripts/                 # run_queue.sh / run_seeds.sh (batch runs), bench.py (throughput check)
assignment1/             # assignment handout + the course notebook, completed with all experiments
AI_USAGE.md              # how AI coding tools were used (required)
```

## Setup

Tested with Python 3.13.9, PyTorch 2.14.1 / torchvision 0.29.1 on macOS (Apple M4, MPS backend).
The code also runs on CUDA or CPU (`cvproj/utils.py::get_device`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the dataset under `data/` (it is not committed):

```text
data/train/<class>/*.jpg   # 16 classes x 150 images
data/test/<class>/*.jpg    # 16 classes x 25 images (labelled, used once for the final number)
data/test2/*.jpg           # 400 unlabelled images (predictions written to results/eval/)
```

## Reproducing the results

```bash
# train the final model (writes checkpoints/<name>.pt, results/runs/<name>.json, a row in results/experiments.csv)
python train.py --config configs/best.yaml

# evaluate: validation + test accuracy, per-class accuracy, confusion matrix, predictions for data/test2
python evaluate.py --checkpoint checkpoints/<name>.pt --tta --predict-dir data/test2

# any other experiment, or a different seed
python train.py --config configs/e5_r18_labelsmooth.yaml --set train.seed=1 --tag seed1
```

Seeds are fixed (`train.seed`, default 0) and the train/val split is fixed (`data.split_seed: 0`)
for every experiment. GPU kernels are not bit-exact across hardware/software versions, so a re-run
may differ by a few validation images.

The notebook `assignment1/ITCS_6169_8169_Assignment1_2026_Starter.ipynb` walks through every
experiment (tables + learning curves are built from `results/runs/*.json`) and ends with the single
final test-set evaluation. Set `RUN_TRAINING = True` in it to re-train everything from the notebook.

## Validation protocol

* Stratified 80/20 split of `data/train`: 120 train / 30 val images per class (1,920 / 480).
* Model selection = epoch with the best validation accuracy (ties → lower val loss).
* The labelled test set is used **once**, for the selected final model.
* One val image = 0.21 %; differences under ~1 % are treated as noise unless confirmed with extra seeds.

<!-- JOURNEY:START -->
<!-- JOURNEY:END -->

## Pretrained weights

The pretrained models start from **ImageNet-1k** torchvision weights; the original 1000-way
classifier is replaced by a new 16-way linear layer and **all** parameters are fine-tuned (except
experiment e7, a linear probe in which only the new layer is trained). The greyscale images are
replicated to 3 channels and normalised with ImageNet statistics.

| model | torchvision weights |
|---|---|
| ResNet-18 | `ResNet18_Weights.IMAGENET1K_V1` |
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2` |
| ConvNeXt-Tiny | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| MobileNetV3-Large | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |

No vision transformers, CLIP, DINO or other foundation models are used.
