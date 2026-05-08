# Work Breakdown

## Phase 0: Fixtures And Baseline

Tasks:

- Generate local LegoGears train/valid lists and a local `.data` file without `/home/stephane/...` paths.
- Add the local generation command/script to recreate those files from the checked-in LegoGears image folders.
- Add a small fixture directory for fixed images and expected PyTorch outputs.
- Run PyTorch reference `gelan-t` / `gelan-s` forward on fixed images and save raw per-scale logits plus decoded detections.
- Record exact preprocessing: image size, letterbox settings, RGB/BGR conversion, normalization, thresholds, NMS IoU.

Acceptance:

- Reproducible fixture script runs from this repo.
- Fixture output includes raw logits, decoded boxes, class scores, and NMS output.
- Local LegoGears paths are generated, not hand-edited into source data.
- `test_training_set/LegoGears_v2/LegoGears.local.data` points only at repo-local paths.

## Phase 1: CFG Expansion For GELAN

Tasks:

- Implement or hand-author a first `gelan-t` Darknet cfg using expanded primitives.
- Add parser smoke tests or a small CLI/script check that loads cfg and dumps layer shapes.
- Confirm local avgpool output matches PyTorch `avg_pool2d(k=2, s=1, p=0)`.
- Confirm `ADown`, `SPPELAN`, `ELAN1`, and `RepNCSPELAN4` shape transitions.
- Add generated cfg comments that identify source YAML layer numbers.

Acceptance:

- `parse_network_cfg_custom()` loads the new cfg.
- Layer output shapes match PyTorch for at least `gelan-t`.
- No new C++ layer types are needed before the output layer.

## Phase 2: YOLOv9 Inference Layer

Tasks:

- Add `ELayerType::YOLOV9` and `[yolov9]` parser support.
- Add `yolov9_layer.hpp/.cpp` with CPU forward, resize, detection extraction, and memory cleanup integration.
- Implement DFL projection and `dist2bbox` decode.
- Add no-objectness detection handling.
- Wire `darknet_network.cpp` detection counting and filling paths.
- Add CPU unit tests for DFL projection, anchor point generation, and decode on tiny tensors.
- Add CUDA forward kernels after CPU behavior is verified.

Acceptance:

- Darknet can run inference through a `gelan-t` cfg with random weights without shape or memory errors.
- CPU DFL/decode tests match PyTorch fixtures within tolerance.
- Detection output format works with existing `detector test` and NMS code.

## Phase 3: Weight Conversion For Inference Parity

Tasks:

- Build a converter from fused PyTorch model tensors to Darknet `.weights`.
- Handle Conv+BN fusion and fused RepConvN.
- Map expanded cfg layers back to PyTorch modules.
- Produce a conversion report and fail on unmapped or mismatched tensors.
- Compare Darknet decoded detections with PyTorch decoded detections before NMS.

Acceptance:

- Converted `gelan-t` weights load without size mismatch.
- Fixed-image decoded boxes/classes match PyTorch within documented tolerance.
- NMS output differences are explainable and bounded.

## Phase 4: YOLOv9 Training Loss

Tasks:

- Implement Task-Aligned Assignment over all grid points in one branch.
- Implement BCE class loss without objectness.
- Implement CIoU box loss using existing `box.cpp` math where possible.
- Implement DFL loss and delta propagation into distribution logits.
- Add backward path that distributes deltas to each prediction input layer.
- Implement CPU path first; add CUDA path for practical training.
- Add logging for `box_loss`, `cls_loss`, `dfl_loss`, and total loss.
- Run the required LegoGears training gate for the GELAN-t cfg:

```bash
./darknet detector train test_training_set/LegoGears_v2/LegoGears.local.data cfg/gelan-t-legogears.cfg -dont_show -map
```

Acceptance:

- The LegoGears training gate completes all `3000` batches with `-map`.
- Loss is finite and decreases over a small number of iterations.
- Weights save and reload.
- Saved weights produce detections on a LegoGears image.
- Collapse gate passes: no NaN/Inf loss, final weights exist, validation mAP is not effectively zero, and detections on non-empty validation images do not collapse to only background / a single class.
- Phase 5 does not start until this gate is recorded with the command, cfg, data file, final mAP, and representative validation detections.

## Phase 5: Dual PGI / YOLOv9

Tasks:

- Expand `CBLinear` into split convs and `CBFuse` into resize/sum in generated cfgs.
- Add dual branch support to `[yolov9]`.
- Implement branch weighting: auxiliary `0.25`, main `1.0`.
- Default inference to main branch.
- Add parity fixtures for `yolov9-t` and `yolov9-s`.

Acceptance:

- `yolov9-t` cfg parses and runs.
- Dual branch training smoke completes.
- Main-branch inference matches PyTorch dual detection script behavior.

## Phase 6: Repo Integration And Documentation

Tasks:

- Add cfgs to `cfg/` and ensure install rules cover needed assets.
- Add README docs for YOLOv9 training, inference, and conversion.
- Add CI smoke where practical: cfg parse, unit tests, converter dry-run without large weights.
- Add CUDA/ROCm notes if GPU kernels diverge.

Acceptance:

- Existing YOLOv4/YOLOv7 cfgs still parse and tests still pass.
- New YOLOv9 docs provide a minimal end-to-end LegoGears example.
- CI validates the new code paths without requiring large model downloads.
