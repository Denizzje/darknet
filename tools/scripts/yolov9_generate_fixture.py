#!/usr/bin/env python3
"""Generate YOLOv9 reference fixtures from the PyTorch implementation.

The fixture is an ``.npz`` file with raw per-scale logits, DFL-projected
distances, decoded boxes, class scores, and simple NMS outputs.  It is intended
as the PyTorch side of Darknet parity checks; the comparison script accepts the
same key layout for Darknet-produced dumps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = ROOT / "reference_yolov9_repo" / "yolov9"


def import_reference(reference_root: Path) -> Any:
	if not (reference_root / "models" / "yolo.py").exists():
		raise ValueError(f"{reference_root} does not look like the YOLOv9 reference root")
	sys.path.insert(0, str(reference_root.resolve()))
	try:
		from models.experimental import attempt_load  # type: ignore
		from models.yolo import DetectionModel  # type: ignore
		from utils.tal.anchor_generator import dist2bbox, make_anchors  # type: ignore
	finally:
		try:
			sys.path.remove(str(reference_root.resolve()))
		except ValueError:
			pass
	return attempt_load, DetectionModel, make_anchors, dist2bbox


def load_image_tensor(path: Path, width: int, height: int) -> Any:
	try:
		from PIL import Image
	except ImportError as exc:
		raise RuntimeError("loading images requires Pillow; use --random-input if Pillow is unavailable") from exc
	import torch

	image = Image.open(path).convert("RGB").resize((width, height))
	array = np.asarray(image, dtype=np.float32) / 255.0
	array = np.transpose(array, (2, 0, 1))[None]
	return torch.from_numpy(array)


def make_input(args: argparse.Namespace) -> Any:
	import torch

	if args.image:
		return load_image_tensor(args.image, args.width, args.height)
	generator = torch.Generator(device="cpu")
	generator.manual_seed(args.seed)
	if args.random_input:
		return torch.rand((1, 3, args.height, args.width), generator=generator)
	return torch.zeros((1, 3, args.height, args.width), dtype=torch.float32)


def load_model(args: argparse.Namespace) -> Any:
	import torch

	attempt_load, DetectionModel, _make_anchors, _dist2bbox = import_reference(args.reference_root)
	if args.weights:
		model = attempt_load(str(args.weights), device=torch.device("cpu"), fuse=args.fuse)
	else:
		model = DetectionModel(str(args.yaml), ch=3, nc=args.classes)
	model.eval()
	return model


def normalise_outputs(output: Any) -> tuple[list[Any], list[list[Any]]]:
	if isinstance(output, tuple):
		decoded, raw = output
	else:
		decoded, raw = output, None

	if isinstance(decoded, list):
		decoded_branches = decoded
	else:
		decoded_branches = [decoded]

	if raw is None:
		raw_branches: list[list[Any]] = []
	elif raw and isinstance(raw[0], list):
		raw_branches = raw
	else:
		raw_branches = [raw]
	return decoded_branches, raw_branches


def split_raw_branch(raw_branch: Sequence[Any], reg_max: int, classes: int) -> tuple[Any, Any]:
	import torch

	batch = raw_branch[0].shape[0]
	no = reg_max * 4 + classes
	merged = torch.cat([level.view(batch, no, -1) for level in raw_branch], 2)
	box, cls = merged.split((reg_max * 4, classes), 1)
	return box.permute(0, 2, 1).contiguous(), cls.permute(0, 2, 1).contiguous()


def dfl_project(box_logits: Any, reg_max: int) -> Any:
	import torch

	batch, anchors, channels = box_logits.shape
	proj = torch.arange(reg_max, dtype=box_logits.dtype, device=box_logits.device)
	return box_logits.view(batch, anchors, 4, channels // 4).softmax(3).matmul(proj)


def simple_nms(boxes_xywh: np.ndarray, scores: np.ndarray, max_det: int, iou_thresh: float) -> np.ndarray:
	if boxes_xywh.size == 0 or scores.size == 0:
		return np.zeros((0, 6), dtype=np.float32)
	classes = scores.argmax(axis=1)
	conf = scores.max(axis=1)
	order = np.argsort(-conf)
	xyxy = np.empty_like(boxes_xywh)
	xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5
	xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5
	xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
	xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5

	kept: list[int] = []
	for idx in order:
		if conf[idx] <= 0.0:
			break
		suppressed = False
		for kept_idx in kept:
			if classes[idx] != classes[kept_idx]:
				continue
			xx1 = max(xyxy[idx, 0], xyxy[kept_idx, 0])
			yy1 = max(xyxy[idx, 1], xyxy[kept_idx, 1])
			xx2 = min(xyxy[idx, 2], xyxy[kept_idx, 2])
			yy2 = min(xyxy[idx, 3], xyxy[kept_idx, 3])
			inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
			area_a = max(0.0, xyxy[idx, 2] - xyxy[idx, 0]) * max(0.0, xyxy[idx, 3] - xyxy[idx, 1])
			area_b = max(0.0, xyxy[kept_idx, 2] - xyxy[kept_idx, 0]) * max(0.0, xyxy[kept_idx, 3] - xyxy[kept_idx, 1])
			iou = inter / max(area_a + area_b - inter, 1.0e-12)
			if iou > iou_thresh:
				suppressed = True
				break
		if not suppressed:
			kept.append(int(idx))
			if len(kept) >= max_det:
				break

	if not kept:
		return np.zeros((0, 6), dtype=np.float32)
	return np.column_stack([boxes_xywh[kept], conf[kept], classes[kept].astype(np.float32)]).astype(np.float32)


def generate_fixture(args: argparse.Namespace) -> dict[str, Any]:
	import torch

	_attempt_load, _DetectionModel, make_anchors, dist2bbox = import_reference(args.reference_root)
	model = load_model(args)
	input_tensor = make_input(args)
	with torch.no_grad():
		output = model(input_tensor)

	decoded_branches, raw_branches = normalise_outputs(output)
	fixture: dict[str, Any] = {
		"input": input_tensor.numpy().astype(np.float32),
		"meta": np.array(
			json.dumps(
				{
					"yaml": str(args.yaml),
					"weights": str(args.weights) if args.weights else None,
					"classes": args.classes,
					"reg_max": args.reg_max,
					"width": args.width,
					"height": args.height,
					"branches": len(decoded_branches),
				},
				sort_keys=True,
			)
		),
	}

	for branch_idx, decoded in enumerate(decoded_branches):
		decoded_np = decoded.detach().cpu().numpy().astype(np.float32)
		fixture[f"branch{branch_idx}/decoded_xywh_scores"] = decoded_np
		boxes = np.transpose(decoded_np[:, :4, :], (0, 2, 1))[0]
		scores = np.transpose(decoded_np[:, 4:, :], (0, 2, 1))[0]
		fixture[f"branch{branch_idx}/decoded_xywh"] = boxes.astype(np.float32)
		fixture[f"branch{branch_idx}/class_scores"] = scores.astype(np.float32)
		fixture[f"branch{branch_idx}/nms"] = simple_nms(boxes, scores, args.max_det, args.iou_thresh)

	for branch_idx, raw_branch in enumerate(raw_branches):
		for scale_idx, level in enumerate(raw_branch):
			fixture[f"branch{branch_idx}/scale{scale_idx}/raw_logits"] = level.detach().cpu().numpy().astype(np.float32)
		box_logits, cls_logits = split_raw_branch(raw_branch, args.reg_max, args.classes)
		fixture[f"branch{branch_idx}/raw_box_logits_flat"] = box_logits.detach().cpu().numpy().astype(np.float32)
		fixture[f"branch{branch_idx}/raw_class_logits_flat"] = cls_logits.detach().cpu().numpy().astype(np.float32)
		fixture[f"branch{branch_idx}/dfl_distances"] = dfl_project(box_logits, args.reg_max).detach().cpu().numpy().astype(np.float32)
		anchor_points, _stride_tensor = make_anchors(raw_branch, model.stride, 0.5)
		fixture[f"branch{branch_idx}/anchor_points"] = anchor_points.detach().cpu().numpy().astype(np.float32)
		fixture[f"branch{branch_idx}/strides"] = model.stride.detach().cpu().numpy().astype(np.float32)
		fixture[f"branch{branch_idx}/decoded_xyxy_grid"] = dist2bbox(
			torch.from_numpy(fixture[f"branch{branch_idx}/dfl_distances"]),
			anchor_points.unsqueeze(0),
			xywh=False,
		).numpy().astype(np.float32)

	return fixture


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
	parser.add_argument("--yaml", type=Path, required=True, help="reference YOLOv9 model YAML")
	parser.add_argument("--weights", type=Path, help="optional PyTorch checkpoint")
	parser.add_argument("--classes", type=int, default=80)
	parser.add_argument("--reg-max", type=int, default=16)
	parser.add_argument("--width", type=int, default=640)
	parser.add_argument("--height", type=int, default=640)
	parser.add_argument("--image", type=Path)
	parser.add_argument("--random-input", action="store_true")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--fuse", action="store_true", help="fuse Conv+BN while loading checkpoint")
	parser.add_argument("--max-det", type=int, default=300)
	parser.add_argument("--iou-thresh", type=float, default=0.45)
	parser.add_argument("--output", type=Path, required=True)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	if args.classes <= 0 or args.reg_max <= 0 or args.width <= 0 or args.height <= 0:
		parser.error("classes, reg-max, width, and height must be positive")
	try:
		fixture = generate_fixture(args)
	except Exception as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 2
	args.output.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(args.output, **fixture)
	print(f"wrote {args.output} with {len(fixture)} tensors")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
