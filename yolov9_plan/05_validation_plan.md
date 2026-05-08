# Validation Plan

## Local Dataset

Use `test_training_set/LegoGears_v2` for smoke tests.

Facts:

- 5 classes in `LegoGears.names`: `red light`, `pin`, `center`, `small gear`, `medium gear`.
- 90 images and YOLO `.txt` labels across `set_01`, `set_02_empty`, and `set_03`.
- Existing `LegoGears.data` uses `/home/stephane/...` paths and should not be used as-is.
- Existing `LegoGears.cfg` is a DarkMark-generated YOLOv4-tiny-style baseline at `224x160`.

Required generated artifacts:

- `test_training_set/LegoGears_v2/LegoGears_train.local.txt`
- `test_training_set/LegoGears_v2/LegoGears_valid.local.txt`
- `test_training_set/LegoGears_v2/LegoGears.local.data`
- `tools/legogears_yolov9_make_local_data.sh`

Suggested split:

- Train: `set_01` plus `set_02_empty`.
- Valid: `set_03`.

Do not check in machine-specific absolute paths. Generate local paths during validation.

## Unit Tests

Add focused tests before full training:

- DFL projection:
  - Known logits -> softmax -> expected bin-weighted distance.
  - Compare to `models/common.py:232`.

- Anchor points:
  - Feature shapes `[1, c, h, w]` and strides `[8,16,32]`.
  - Expected points `(0.5,0.5)`, `(1.5,0.5)`, etc.
  - Compare to `utils/tal/anchor_generator.py:8`.

- Decode:
  - Known `ltrb` distances and anchor points -> expected `xywh` / `xyxy`.
  - Compare to `utils/tal/anchor_generator.py:23`.

- No-objectness detection extraction:
  - Class scores should be used directly.
  - `objectness * class_prob` behavior must not suppress YOLOv9 detections.

- CFG shape tests:
  - Parse `gelan-t` cfg.
  - Assert selected block output dimensions match the PyTorch model summary.

## Inference Parity

Fixtures:

- Use one reference image from `artwork/` and at least one LegoGears image.
- Save PyTorch preprocessed tensor, raw final logits per scale, decoded boxes before NMS, and NMS output.

Comparison levels:

1. Preprocessing parity:
   Same resize/letterbox, RGB order, normalization, and input dimensions.

2. Raw output parity:
   For converted weights, compare per-scale logits before DFL.

3. Decode parity:
   Compare DFL distances and decoded boxes before NMS.

4. NMS parity:
   Compare final boxes/classes/scores with the same confidence and IoU thresholds.

Tolerances:

- CPU float path should be tight enough for direct tensor comparisons.
- CUDA/half paths need wider tolerances; document them after measurement.

## Training Gate

Mandatory GELAN + `DDetect` gate:

- Model: `gelan-t` adapted to `classes=5`.
- Dataset: local LegoGears split.
- Input size: start with a small multiple of 32, preferably close to `224x160` if the model shape logic supports it; otherwise use `320x320`.
- Run exactly the cfg's `max_batches=3000` training schedule with `-map`.

Command:

```bash
./darknet detector train test_training_set/LegoGears_v2/LegoGears.local.data cfg/gelan-t-legogears.cfg -dont_show -map
```

Pass criteria:

- No parser errors.
- No NaN/Inf loss.
- `box_loss`, `cls_loss`, and `dfl_loss` are reported.
- Loss trends downward over the short run, allowing noisy small-dataset behavior.
- `.weights` file is produced and reloads.
- `detector test` produces plausible detections on at least one validation image.
- `-map` runs during training and reports a nonzero validation signal by the end of the `3000` batches.

Collapse gate:

- Fail if final mAP is effectively zero after the `3000`-batch run.
- Fail if saved weights reload but emit no detections on the non-empty `set_03` validation images at normal confidence thresholds.
- Fail if predictions collapse to only background / a single class across the non-empty validation images.
- Fail if decoded boxes collapse to near-zero area or full-image boxes for most validation detections.
- Record the final mAP, per-class AP when available, final loss line, final weights path, and two representative validation image detections before declaring the GELAN + `DDetect` phase complete.

## Regression Coverage

Run before merging YOLOv9 work:

- Existing `darknet_tests` / `ctest` when GTest is available.
- Parse smoke for existing `cfg/yolov4.cfg`, `cfg/yolov7.cfg`, and `cfg/yolov7-tiny.cfg`.
- A small inference smoke on an existing YOLOv4/YOLOv7 cfg if weights are available locally.
- CPU-only build.
- CUDA build when CUDA is available.
- ROCm build if the project continues to support it in this environment.

## CI Practicality

Avoid large binary weights in CI. CI can still validate:

- New C++ unit tests.
- New cfg files parse.
- Converter mapping dry-run against synthetic tensor metadata.
- Fixture tests with tiny serialized tensors if needed.
