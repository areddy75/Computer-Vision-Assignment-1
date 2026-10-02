# ITCS 6169/8169 Assignment 1 — The CNN Challenge (16-class scene recognition)

Small-data scene classification (2,400 training images, 16 classes, 256×256 greyscale) with
convolutional neural networks. This repository contains the code, configs, logs and the final
checkpoint behind my report.

<!-- RESULTS:START -->
**Final result (single evaluation of the model selected on validation):**

| | accuracy |
|---|---:|
| **Test set** (400 images, 25/class) | **96.75 %** (387/400) |
| Validation (480 images, stratified 20 % of `data/train`) | 96.67 % (464/480) |

Final model: **ConvNeXt-Tiny** initialised from ImageNet-1k, all layers fine-tuned, 224×224 greyscale→3-channel input,
light augmentation (reflect-pad random crop + horizontal flip), AdamW (lr 1e-4, weight decay 0.05), 1 warm-up
epoch + cosine decay, label smoothing 0.1, batch 32, 30 epochs, best-validation-epoch selection (epoch 16),
no test-time augmentation. Config: [`configs/best.yaml`](configs/best.yaml). Checkpoint:
[`checkpoints/final_convnext_tiny_fp16.pt`](checkpoints/final_convnext_tiny_fp16.pt) (56 MB fp16 copy of the
fp32 training checkpoint; verified to give identical predictions on all validation images;
sha256 of the fp32 original `b0024f41…c1b589`).

```bash
python evaluate.py --checkpoint checkpoints/final_convnext_tiny_fp16.pt --predict-dir data/test2
```
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
  eval/                  # final-model evaluation (report json, confusion matrices, test2 predictions)
  eval_val/              # validation-only analyses of non-final models (test set untouched)
scripts/                 # run_queue.sh / run_seeds.sh (batch runs), bench.py (throughput check)
assignment1/             # assignment handout + the course notebook, completed with all experiments
AI_USAGE.md              # how AI coding tools were used (required)
report/                  # 2-page report (report.tex -> report.pdf) and its figure
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

## Experimental journey (validation accuracy, fixed split)

| # | experiment | val acc | observation |
|---|---|---:|---|
| e0 | starter TNet, grey 64 px | 44.0 % | 99 % train acc, strongly overfits |
| e1 | own 8-layer CNN (BN + GAP), 128 px, AdamW + cosine | 87.7 % | depth + GAP head matter far more than parameters |
| e2 / e2b | + strong / **light** augmentation | 87.3 / 90.2 % | strong aug hurt (see failure analysis); light aug helps |
| e3 / e4 | ResNet-18 scratch / **ImageNet init** | 86.7 / 92.5 % (mean 92.9) | pretraining +5.8 %; capacity alone does not help |
| e7 | ResNet-18 linear probe | 88.5 % | frozen features good, fine-tuning adds +6 % |
| e5 | + label smoothing 0.1 | 94.8 % (mean 94.4) | +1.5 % mean over 3 seeds |
| e6 | + MixUp/CutMix | 92.9 % | hurt: over-regularises a 30-epoch / 1.9k-image run |
| e5b | light instead of strong aug (pretrained) | 95.4 % (mean 95.4) | +1.0 % over 2 seeds |
| e8–e11 | ConvNeXt-T / EffNet-B0 / ResNet-50 / MobileNetV3 | 95.2 / 95.2 / 96.3 / 94.6 % | families within ~1 % |
| e8b | ConvNeXt-T lr 3e-4 → 1e-4 | 96.0 % (mean 96.5) | LR matters more than the family |
| e12/e13 | ResNet-18 at 160 / 288 px | 92.3 / 95.8 % | resolution helps (+1.4 % at 288); also EffNet-B0 96.0 %, ResNet-50 96.5 % at 288 |
| **e16** | **ConvNeXt-T, lr 1e-4, light aug (final)** | **96.7 %** | **test 96.75 %** |

Every run's config is in `configs/`, its log in `results/logs/` and its full history in `results/runs/`;
`results/experiments.csv` has one row per run (including the extra-seed runs). The notebook contains the
learning curves, the failure analysis, per-class analysis and the final evaluation.

**Best accuracy–efficiency trade-off:** EfficientNet-B0 at 288 px reaches 96.0 % validation accuracy with
1.3 GFLOPs/image (1/7 of ConvNeXt-T) and 4.0 M parameters (`configs/e15_effb0_res288.yaml`).

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
