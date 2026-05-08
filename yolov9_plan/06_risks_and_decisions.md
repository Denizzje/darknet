# Risks And Decisions

## Decisions To Lock Early

### Start With GELAN + DDetect

Decision: implement single-branch GELAN support before raw dual YOLOv9 PGI.

Reason: it exercises the new anchor-free DFL head and TAL loss without the extra PGI branch complexity.

### Expand Blocks Into CFG Primitives

Decision: do not add native C++ layers for `ELAN1`, `RepNCSPELAN4`, `SPPELAN`, `AConv`, or `ADown` in the first pass.

Reason: these blocks are compositions of conv, split/concat, pooling, upsample, and sum operations that Darknet mostly already has. Native block layers would increase parser, memory, resize, CPU, CUDA, and weight loader surface without solving the hardest YOLOv9 problem.

### Add A New Anchor-Free Output Layer

Decision: add a new `[yolov9]` output/loss layer rather than mutating `[yolo]`.

Reason: YOLOv9 has no anchors and no objectness term. Overloading `[yolo]` would add many conditionals to mature YOLOv3/v4/v7 code and increase regression risk.

### Prefer Fused/Converted Inference Weights First

Decision: support fused PyTorch inference weights before raw train-time PyTorch weights.

Reason: fused weights avoid RepConvN branch mapping during the first parity milestone. Training from scratch in Darknet can still use expanded train-time branch cfgs.

## Main Risks

### TAL Loss Complexity

Risk: Task-Aligned Assignment is global across all scales and differs substantially from the current Darknet YOLO anchor assignment.

Mitigation:

- Implement CPU reference-style assignment first with tiny fixtures.
- Compare assignment masks and target scores with PyTorch on small tensors.
- Only optimize or port to CUDA after correctness is fixed.

### DFL Backward Correctness

Risk: DFL loss and DFL decode both involve softmax over distribution bins. Incorrect gradients can still produce finite but bad training.

Mitigation:

- Add finite-difference or PyTorch-comparison tests for tiny logits.
- Log separate `dfl_loss`.
- Keep decode and loss projection implementations visibly aligned with the reference.

### No-Objectness Post-Processing

Risk: Existing detection code assumes objectness exists and multiplies objectness by class probability.

Mitigation:

- Add a YOLOv9-specific detection extraction path.
- Keep the public `Detection` structure compatible by storing class scores in `prob[]`.
- Do not let YOLOv9 scores pass through objectness multiplication.

### CFG Expansion Index Bugs

Risk: Expanded block cfgs will be large, and off-by-one route indices are easy to introduce.

Mitigation:

- Generate cfgs from YAML rather than hand-maintaining large files long term.
- Include source YAML index comments.
- Add shape assertions and parser smoke tests.

### Pooling Semantics

Risk: PyTorch `avg_pool2d(k=2, s=1, p=0)` in `AConv`/`ADown` may not exactly match Darknet `local_avgpool` defaults.

Mitigation:

- Emit `padding=0` explicitly.
- Add unit tests comparing small tensors against PyTorch.

### CBLinear / CBFuse Translation

Risk: `CBLinear` returns unequal split tensors, which Darknet `route groups` cannot represent directly.

Mitigation:

- Translate `CBLinear` into separate 1x1 conv layers per split.
- Convert PyTorch conv weights by slicing output channels into those convs.
- Test each split shape and CBFuse sum against PyTorch.

### Weight Converter Maintenance

Risk: Converter mapping can drift from generated cfgs.

Mitigation:

- Generate cfg and mapping from the same source graph.
- Emit a mapping report.
- Fail closed on any unmapped or extra weighted layer.

### Performance

Risk: A correct CPU TAL implementation may be too slow for real training.

Mitigation:

- Use CPU TAL only for correctness and small smoke tests.
- Add CUDA kernels or efficient GPU helper functions before declaring training support complete.

### LegoGears Training Collapse

Risk: The new anchor-free loss can complete forward/backward while learning a degenerate solution, especially on the tiny LegoGears split with many empty frames.

Mitigation:

- Treat the `3000`-batch LegoGears `-map` run as a mandatory gate, not an optional smoke.
- Fail the phase if mAP remains effectively zero, saved weights emit no validation detections, predictions collapse to only background / one class, or decoded boxes collapse to near-zero/full-image boxes.
- Record final mAP, per-class AP when available, loss components, final weights path, and representative validation detections before starting dual PGI work.
