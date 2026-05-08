# Darknet Gap Analysis

The target repo is a C++/CUDA Darknet implementation with cfg parsing in `src-lib/darknet_cfg.cpp`, layer metadata in `src-lib/darknet_layers.hpp`, and layer enum/name mapping in `src-lib/darknet_enums.*`.

## Existing Support That Helps

- Config entry points:
  - `src-lib/weights.cpp:39`: `parse_network_cfg()`.
  - `src-lib/darknet_cfg.cpp:807`: parser dispatch over `ELayerType`.
  - `src-lib/darknet_cfg.hpp:239`: parse method declarations.

- Existing layer primitives:
  - `convolutional`: parser `darknet_cfg.cpp:1409`, runtime `convolutional_layer.cpp`.
  - `route`: parser `darknet_cfg.cpp:1519`, runtime `route_layer.cpp`.
  - `shortcut`: parser `darknet_cfg.cpp:1742`, runtime `shortcut_layer.cpp`.
  - `upsample`: parser `darknet_cfg.cpp:1726`, runtime `upsample_layer.cpp`.
  - `maxpool`: parser `darknet_cfg.cpp:1591`, runtime `maxpool_layer.cpp`.
  - `local_avgpool`: parser `darknet_cfg.cpp:1888`, runtime also `maxpool_layer.cpp` with average pooling.
  - `yolo`: parser `darknet_cfg.cpp:1624`, runtime `yolo_layer.cpp`.

- Useful cfg features already present:
  - `route` supports `groups` and `group_id`, which can split channels.
  - `convolutional` supports grouped convs through `groups`.
  - `swish` activation is supported in the activation name map at `darknet_enums.cpp:81`.
  - IoU losses include `ciou`; NMS includes `diounms`.
  - Weight save/load already handles normal convolutional layers, so expanded YOLOv9 blocks can avoid a new weight format for block internals.

## Missing For YOLOv9

### 1. Anchor-Free Detection Layer

Darknet `[yolo]` assumes anchors, objectness, and `n * (classes + 4 + 1)` outputs. YOLOv9 `DDetect` outputs `classes + 4 * reg_max` per grid point, uses no objectness channel, and decodes boxes through DFL distances from grid centers.

Required additions:

- New layer type, likely `[yolov9]` or `[dfl_yolo]`.
- Multi-scale input support through `layers=...`, because TAL assignment and DFL decode operate across all feature levels.
- Detection extraction path that uses class scores directly rather than objectness multiplied by class probability.
- CPU and GPU forward paths for DFL softmax/projection and decode.

### 2. TAL + DFL Training Loss

Current `yolo_layer.cpp` loss assigns anchors with Darknet-era matching and writes deltas for objectness, class, and bbox. YOLOv9 needs:

- Task-Aligned Assignment over all grid points from all scales.
- BCE class loss without objectness.
- CIoU box loss on decoded boxes.
- DFL cross-entropy over left/right adjacent bins for each side.
- Delta distribution back to the per-scale raw logits.
- Dual/triple branch weighting after single-head is proven.

### 3. PGI / Dual Branch Plumbing

GELAN can be supported without PGI. Raw YOLOv9 dual models need:

- CFG representation for `CBLinear` and `CBFuse`.
- Auxiliary and main branch inputs to the same output/loss layer.
- Training loss that weights auxiliary branch at `0.25` and main branch at `1.0`.
- Inference that exports or uses only the main branch by default.

### 4. Weight Conversion

Darknet `.weights` files store layer weights in cfg order. PyTorch `.pt` files store module names and tensors. A converter is required for practical parity:

- Fused/converted inference weights first.
- Raw branch weights second.
- Shape validation against the generated Darknet cfg.
- Mapping for expanded blocks, especially RepConvN and split PGI convs.

### 5. Validation Coverage

Current tests are utility-level (`src-test/test_box.cpp`, `test_image.cpp`, `test_random.cpp`, `test_misc.cpp`, `test_time.cpp`). There are no model parse, layer shape, training smoke, or inference parity tests.

## Feasibility Of CFG Expansion

Recommended approach: expand YOLOv9 blocks into existing Darknet primitives instead of adding native composite layers for every PyTorch block.

| PyTorch Block | Darknet Mapping |
|---|---|
| `Conv` | `[convolutional] batch_normalize=1 activation=swish` |
| `AConv` | `[local_avgpool] size=2 stride=1 padding=0` then stride-2 3x3 conv |
| `ADown` | local avgpool, `route groups=2`, branch conv/maxpool+conv, concat route |
| `SPPELAN` | conv, maxpool x3, route concat, conv |
| `ELAN1` | conv, channel split with route groups, conv chains, concat route, conv |
| `RepConvN` | training cfg as 3x3 conv + 1x1 conv + shortcut sum + swish; fused inference as one 3x3 conv |
| `RepNCSPELAN4` | expanded CSP/ELAN sequence using `RepNCSP`, routes, shortcuts, convs |
| `CBLinear` | separate 1x1 convs for each split instead of tuple output |
| `CBFuse` | nearest upsample required split tensors, shortcut sum |
| `DDetect` / `DualDDetect` | new YOLOv9 output/loss layer |

This keeps the new C++ runtime focused on the part that existing Darknet cannot express: anchor-free DFL decode and TAL training.
