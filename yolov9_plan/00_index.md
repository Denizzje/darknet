# YOLOv9 Port Plan

This folder captures the research and work plan for adding YOLOv9 support to this Darknet tree. The reference implementation inspected for this plan is the local PyTorch repo at `reference_yolov9_repo/yolov9`; the target implementation is the C++/CUDA Darknet code under `src-lib` plus configs under `cfg`.

## Recommended Target Order

1. Support GELAN + `DDetect` first.
   This gives a real YOLOv9-family model with the smallest new surface: GELAN blocks, anchor-free DFL decoding, TAL loss, and no PGI auxiliary branch.

2. Add YOLOv9 raw dual PGI + `DualDDetect` second.
   This adds the auxiliary branch and PGI routing after the anchor-free head and loss are already proven.

3. Treat `TripleDDetect` / `yolov9-cf.yaml` as a later stretch target.
   The reference has different inference behavior for the third/fine branch, so it should not block basic YOLOv9 support.

## Documents

- `01_reference_inventory.md`: what the PyTorch YOLOv9 reference contains and what must be ported.
- `02_darknet_gap_analysis.md`: what this Darknet tree already supports and what is missing.
- `03_port_design.md`: proposed architecture, cfg strategy, new layer design, and conversion strategy.
- `04_work_breakdown.md`: phased implementation tasks with acceptance criteria.
- `05_validation_plan.md`: tests, fixtures, dataset handling, parity checks, and training smoke tests.
- `06_risks_and_decisions.md`: decisions to lock early and risks that need explicit mitigation.

## Key Findings

- YOLOv9 detection heads are anchor-free and distribution-based. They output `nc + 4 * reg_max` channels per grid point, with `reg_max = 16`, and no objectness channel.
- Darknet's current `[yolo]` layer is anchor/objectness based. It cannot be adapted by cfg changes alone.
- Most GELAN topology can be expanded into existing Darknet primitives: `convolutional`, `route`, `shortcut`, `upsample`, `maxpool`, and `local_avgpool`.
- The minimum native runtime addition should be a YOLOv9 anchor-free multi-scale output/loss layer, not a new layer for every PyTorch block.
- The included local smoke dataset is `test_training_set/LegoGears_v2`, but its `.data` file uses author-local `/home/stephane/...` paths and must be regenerated for local validation.

## Stage Gates

- Gate 1: Darknet parses a GELAN-tiny/small cfg and produces expected tensor shapes.
- Gate 2: Inference-only YOLOv9/GELAN forward path produces decoded boxes comparable to PyTorch on fixed images.
- Gate 3: The mandatory LegoGears run completes `3000` batches with `-map`, loss is finite, weights save/load, inference works on saved weights, and the collapse gate passes.
- Gate 4: Raw YOLOv9 dual PGI branch trains with auxiliary weighting and uses the main branch for inference.

Gate 3 command:

```bash
./darknet detector train test_training_set/LegoGears_v2/LegoGears.local.data cfg/gelan-t-legogears.cfg -dont_show -map
```

Collapse gate: fail the GELAN + `DDetect` phase if the `3000`-batch `-map` run produces NaN/Inf loss, saves no final weights, reloads but emits no validation detections, reports effectively zero mAP after training, or collapses predictions to only background / a single class across the non-empty validation images.
