# Port Design

## Design Goals

- Preserve existing Darknet cfg and training workflows.
- Avoid a broad YAML parser rewrite for the first implementation.
- Express GELAN topology with existing layers wherever possible.
- Add one focused native runtime surface for YOLOv9 anchor-free detection.
- Make inference parity measurable before implementing full training.

## Proposed Components

### 1. YOLOv9 CFG Generator

Add a small tooling path that converts selected PyTorch YOLOv9 YAMLs into Darknet cfgs. This can start as a repo script and later become a maintained conversion tool.

Responsibilities:

- Read `reference_yolov9_repo/yolov9/models/detect/*.yaml`.
- Expand supported modules into Darknet cfg sections.
- Emit `cfg/gelan-t.cfg`, `cfg/gelan-s.cfg`, and later `cfg/yolov9-t.cfg`, `cfg/yolov9-s.cfg`, etc.
- Track layer index mapping from PyTorch YAML indices to Darknet cfg section indices.
- Emit comments for block boundaries and source YAML layer indices.
- Emit explicit strides for the YOLOv9 output layer rather than relying on dummy-forward stride inference.

Initial target configs:

- `gelan-t.yaml`: smallest single-head `DDetect` model.
- `gelan-s.yaml`: second single-head model for scale coverage.
- LegoGears smoke cfg derived from `gelan-t` with `classes=5`, smaller input, and reduced max batches.

### 2. YOLOv9 Output Layer

Add a new layer type, working name `[yolov9]`.

Parser fields:

- `classes`: number of classes.
- `reg_max`: default `16`.
- `layers`: comma-separated raw prediction layers, one per scale for single-head.
- `strides`: explicit scale strides, for example `8,16,32`.
- `branch_count`: `1` for `DDetect`, `2` for `DualDDetect`, `3` for `TripleDDetect` later.
- `inference_branch`: default `0` for single branch, `1` for dual main branch.
- `aux_loss_weight`: default `0.25` for dual auxiliary branches.
- `box_normalizer`, `cls_normalizer`, `dfl_normalizer`: defaults `7.5`, `0.5`, `1.5`.
- `tal_topk`, `tal_alpha`, `tal_beta`: defaults `10`, `0.5`, `6.0`.

Runtime behavior:

- Validate each input tensor has `classes + 4 * reg_max` channels.
- Generate anchor points `(x + 0.5, y + 0.5)` per scale.
- DFL project distribution logits into four distances.
- Decode `ltrb` distances into `xywh` for inference.
- Use sigmoid class scores directly. There is no objectness term.
- Fill Darknet detections with `objectness=1.0` internally and `prob[class_id]=class_score`, or add an explicit no-objectness post-processing path. The explicit path is cleaner, but the compatibility shim may reduce API churn.
- Backward path computes TAL assignment across all scales in the branch and writes deltas into each input prediction layer.

Files likely touched:

- `src-lib/darknet_enums.hpp`
- `src-lib/darknet_enums.cpp`
- `src-lib/darknet_layers.hpp`
- `src-lib/darknet_cfg.hpp`
- `src-lib/darknet_cfg.cpp`
- `src-lib/darknet_network.cpp`
- `src-lib/yolov9_layer.hpp`
- `src-lib/yolov9_layer.cpp`
- `src-lib/yolov9_layer_kernels.cu` or additions to existing CUDA helpers
- `src-lib/box.cpp` / `box.hpp` if post-processing needs no-objectness helpers
- `src-lib/weights.cpp` only if the new layer owns parameters; the recommended design has no owned weights in `[yolov9]`

### 3. Expanded Head Towers

Represent the `DDetect` head conv towers directly in cfg before the `[yolov9]` layer.

For each scale:

- Box branch:
  - Conv 3x3, swish.
  - Grouped Conv 3x3, `groups=4`, swish.
  - Final grouped 1x1 Conv, `groups=4`, `filters=4 * reg_max`, `activation=linear`, no BN.
- Class branch:
  - Conv 3x3, swish.
  - Conv 3x3, swish.
  - Final 1x1 Conv, `filters=classes`, `activation=linear`, no BN.
- Route concat box + class logits.

Then one `[yolov9]` layer consumes the three concatenated logits via `layers=...`.

### 4. Weight Conversion Strategy

Implement conversion in stages:

1. Darknet-trained weights only.
   Proves cfg, forward, backward, save/load.

2. Fused PyTorch inference weights.
   Convert Conv+BN and fused RepConvN into Darknet convolutional sections. Use PyTorch `attempt_load(..., fuse=True)` or the reference converted weights where available.

3. Raw PyTorch train-time weights.
   Map RepConvN branch tensors and PGI split tensors into expanded Darknet cfg sections.

Converter validation:

- Every Darknet weighted layer consumes exactly one source tensor set.
- Tensor shapes match before writing.
- Emit a mapping report: cfg layer index, Darknet shape, PyTorch key, source shape.
- Run fixed-image parity after conversion.

### 5. Dual PGI Strategy

After `DDetect` works:

- Expand `CBLinear` as independent 1x1 convs from the same source layer, one conv per required split.
- Expand `CBFuse` as upsample plus shortcut sum.
- Emit two groups of prediction logits: auxiliary branch and main branch.
- Configure `[yolov9] branch_count=2 inference_branch=1`.
- During training, run TAL and DFL loss for each branch and apply the reference weighting: auxiliary `0.25`, main `1.0`.
- During inference, default to main branch only. Add a debug option to emit auxiliary detections if useful.

## Non-Goals For First Pass

- Segmentation and panoptic YOLOv9 support.
- Full PyTorch YAML parser inside Darknet C++.
- `TripleDDetect` parity.
- ONNX export for the new YOLOv9 layer.
- Quantization or TensorRT-specific behavior.
