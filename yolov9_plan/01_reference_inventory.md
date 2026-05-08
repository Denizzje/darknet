# Reference Inventory

The local reference repo is `reference_yolov9_repo/yolov9`.

## Detection Variants

Detection YAMLs are in `reference_yolov9_repo/yolov9/models/detect`.

| Family | YAMLs | Final Head | First Port Priority |
|---|---|---:|---:|
| GELAN | `gelan.yaml`, `gelan-{t,s,m,c,e}.yaml` | `DDetect` | Yes |
| YOLOv9 dual PGI | `yolov9.yaml`, `yolov9-{t,s,m,c,e}.yaml` | `DualDDetect` | After GELAN |
| YOLOv9 coarse/fine | `yolov9-cf.yaml` | `TripleDDetect` | Later |
| Baseline anchor-free | `yolov7-af.yaml` | `Detect` | Useful fixture |

All YOLOv9 detection YAMLs still declare `anchors: 3`, but the `DDetect`/`DualDDetect` path is anchor-free. `anchors` is legacy parser compatibility in `models/yolo.py:713`.

## Blocks Used By Detection YAMLs

Core blocks and definitions:

- `Conv`: Conv2d + BatchNorm + SiLU, `models/common.py:43`.
- `AConv`: `avg_pool2d(k=2, s=1, p=0)` followed by stride-2 3x3 Conv, `models/common.py:60`.
- `ADown`: avg-pool prefilter, split channels, conv branch plus maxpool branch, concat, `models/common.py:70`.
- `RepConvN`: train-time 3x3 branch + 1x1 branch, fusable for inference, `models/common.py:86`.
- `DFL`: fixed projection over distribution bins, `models/common.py:232`.
- `RepNCSP`: CSP block using `RepNBottleneck`, `models/common.py:373`.
- `SPPELAN`: 1x1 Conv, three serial maxpools, concat, 1x1 Conv, `models/common.py:563`.
- `ELAN1`: tiny/small ELAN block, `models/common.py:580`.
- `RepNCSPELAN4`: main GELAN block, `models/common.py:601`.
- `CBLinear`: one conv whose output is split into unequal channel groups for PGI, `models/common.py:652`.
- `CBFuse`: nearest-resizes selected `CBLinear` splits and sums with current feature, `models/common.py:662`.

Most of these can be represented as expanded Darknet cfg sections. `CBLinear` is the notable exception if represented literally, because it emits a tuple of unequal splits. The preferred Darknet translation is to emit separate 1x1 convolutions for each required split.

## Detection Heads

Head classes are in `reference_yolov9_repo/yolov9/models/yolo.py`.

- `DDetect`, `models/yolo.py:78`:
  - `reg_max = 16`.
  - `no = nc + 4 * reg_max`.
  - For each scale, box branch predicts 64 distribution logits and class branch predicts `nc` logits.
  - In training, returns raw per-scale tensors.
  - In inference, applies DFL, decodes distances from grid center anchor points, multiplies by stride, applies sigmoid to class logits, and returns `[x, y, w, h, class_scores...]`.

- `DualDDetect`, `models/yolo.py:190`:
  - Splits input features into two equal scale groups.
  - Training returns `[aux_branch, main_branch]`.
  - Inference decodes both; reference dual detection scripts select the second/main output.

- `TripleDDetect`, `models/yolo.py:335`:
  - Has three scale groups.
  - Reference inference returns the third/fine branch.

Anchor points are generated in `utils/tal/anchor_generator.py:8` as grid centers `(x + 0.5, y + 0.5)`. `dist2bbox()` at `utils/tal/anchor_generator.py:23` converts `[left, top, right, bottom]` distances into `xywh` or `xyxy`.

## Loss And Assignment

Training scripts select different losses:

- `train.py:39` uses `utils.loss_tal.ComputeLoss` for GELAN / `DDetect`.
- `train_dual.py:40` uses `utils.loss_tal_dual.ComputeLoss` for raw YOLOv9 dual PGI.
- `train_triple.py:40` uses `utils.loss_tal_triple.ComputeLoss`.

The active YOLOv9 detection loss is:

- Task-Aligned Assignment (`TaskAlignedAssigner`) with defaults `topk=10`, `alpha=0.5`, `beta=6.0`.
- BCE classification over class logits, no objectness loss.
- CIoU box loss.
- Distribution Focal Loss over the 16-bin distance distributions.
- Gains: box `7.5`, class `0.5`, DFL `1.5`.
- Dual branch weighting: auxiliary loss is weighted `0.25`; main loss is weighted `1.0`.

Source starts:

- `utils/loss_tal.py:62`: `BboxLoss`.
- `utils/loss_tal.py:106`: single-head `ComputeLoss`.
- `utils/loss_tal_dual.py:106`: dual-head `ComputeLoss`.
- `utils/tal/assigner.py:51`: `TaskAlignedAssigner`.

## Training And Export Assumptions

- Model construction is dynamic YAML parsing with `eval()` module lookup and channel bookkeeping in `models/yolo.py:713`.
- Strides are inferred by a dummy `256x256` forward in `models/yolo.py:604`, not stored explicitly in YAML.
- The reference README prefers converted weights for normal inference with `detect.py`/`val.py`; raw dual weights need `detect_dual.py`/`val_dual.py`.
- `detect_dual.py` selects the second/main decoded branch.
- `attempt_load(..., fuse=True)` fuses Conv+BN and RepConvN branches for inference. The Darknet port should support converted/fused weights first, then raw train-time branch weights.
- NMS consumes `[x, y, w, h, class_scores...]` with no objectness term. Darknet's current post-processing assumes `objectness * class_probability`, so this is a required runtime change.
