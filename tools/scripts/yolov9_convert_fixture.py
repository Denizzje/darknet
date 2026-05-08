#!/usr/bin/env python3
"""Validate YOLOv9 PyTorch-to-Darknet tensor mappings before conversion.

This is a conversion/fixture skeleton, not a Darknet .weights writer.  It is
deliberately validation-first so bad or incomplete mappings fail before any
future writer can serialize unusable weights.

Mapping JSON may be either a list of entries or {"mappings": [...]}:

[
  {
    "source": "model.0.conv.weight",
    "target": "convolutional_0/weights",
    "source_shape": [16, 3, 3, 3],
    "target_shape": [16, 3, 3, 3],
    "transform": "copy",
    "role": "other"
  }
]

State manifest JSON, useful for fixtures without torch, may be either
{"name": [shape...]} or {"tensors": [{"name": "...", "shape": [...]}]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


VALID_TRANSFORMS = {
	"batchnorm",
	"conv2d_oihw",
	"copy",
	"linear",
}

VALID_ROLES = {
	"backbone",
	"ddetect_box_logits",
	"ddetect_class_logits",
	"neck",
	"other",
}


@dataclass(frozen=True)
class TensorSpec:
	name: str
	shape: tuple[int, ...]


@dataclass(frozen=True)
class MappingEntry:
	source: str
	target: str
	source_shape: tuple[int, ...] | None
	target_shape: tuple[int, ...] | None
	transform: str
	role: str


@dataclass
class ValidationReport:
	source_tensor_count: int
	mapping_count: int
	errors: list[str]
	warnings: list[str]
	unmapped_sources: list[str]

	@property
	def ok(self) -> bool:
		return not self.errors

	def to_json_dict(self, mappings: Sequence[MappingEntry]) -> dict[str, Any]:
		return {
			"ok": self.ok,
			"source_tensor_count": self.source_tensor_count,
			"mapping_count": self.mapping_count,
			"errors": self.errors,
			"warnings": self.warnings,
			"unmapped_sources": self.unmapped_sources,
			"validated_mappings": [
				{
					"source": entry.source,
					"target": entry.target,
					"source_shape": list(entry.source_shape) if entry.source_shape is not None else None,
					"target_shape": list(entry.target_shape) if entry.target_shape is not None else None,
					"transform": entry.transform,
					"role": entry.role,
				}
				for entry in mappings
			],
		}


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def parse_shape(value: Any, *, field: str) -> tuple[int, ...]:
	if not isinstance(value, list):
		raise ValueError(f"{field} must be a list of dimensions")
	shape: list[int] = []
	for dim in value:
		if not isinstance(dim, int) or dim < 0:
			raise ValueError(f"{field} contains invalid dimension {dim!r}")
		shape.append(dim)
	return tuple(shape)


def load_state_manifest(path: Path) -> dict[str, TensorSpec]:
	data = load_json(path)
	tensors: dict[str, TensorSpec] = {}

	if isinstance(data, Mapping) and "tensors" in data:
		raw_tensors = data["tensors"]
		if not isinstance(raw_tensors, list):
			raise ValueError("state manifest 'tensors' must be a list")
		for idx, item in enumerate(raw_tensors):
			if not isinstance(item, Mapping):
				raise ValueError(f"state manifest tensors[{idx}] must be an object")
			name = item.get("name")
			if not isinstance(name, str) or not name:
				raise ValueError(f"state manifest tensors[{idx}].name must be a non-empty string")
			shape = parse_shape(item.get("shape"), field=f"state manifest tensors[{idx}].shape")
			tensors[name] = TensorSpec(name=name, shape=shape)
	elif isinstance(data, Mapping):
		for name, shape_value in data.items():
			if not isinstance(name, str) or not name:
				raise ValueError("state manifest tensor names must be non-empty strings")
			shape = parse_shape(shape_value, field=f"state manifest {name}.shape")
			tensors[name] = TensorSpec(name=name, shape=shape)
	else:
		raise ValueError("state manifest must be an object")

	if not tensors:
		raise ValueError("state manifest did not contain any tensors")
	return tensors


def load_checkpoint_shapes(path: Path) -> dict[str, TensorSpec]:
	try:
		import torch  # type: ignore
	except ImportError as exc:
		raise RuntimeError("loading PyTorch checkpoints requires torch; use --state-manifest for fixture-only validation") from exc

	checkpoint = torch.load(path, map_location="cpu")
	if hasattr(checkpoint, "state_dict"):
		state_dict = checkpoint.state_dict()
	elif isinstance(checkpoint, Mapping) and "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
		state_dict = checkpoint["model"].state_dict()
	elif isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
		state_dict = checkpoint["state_dict"]
	elif isinstance(checkpoint, Mapping):
		state_dict = checkpoint
	else:
		raise ValueError("checkpoint does not look like a module or state_dict")

	tensors: dict[str, TensorSpec] = {}
	for name, tensor in state_dict.items():
		if hasattr(tensor, "shape"):
			tensors[str(name)] = TensorSpec(name=str(name), shape=tuple(int(dim) for dim in tensor.shape))

	if not tensors:
		raise ValueError("checkpoint did not expose any tensor-shaped state_dict entries")
	return tensors


def parse_mapping_entry(raw: Any, idx: int) -> MappingEntry:
	if not isinstance(raw, Mapping):
		raise ValueError(f"mapping[{idx}] must be an object")

	source = raw.get("source")
	target = raw.get("target")
	if not isinstance(source, str) or not source:
		raise ValueError(f"mapping[{idx}].source must be a non-empty string")
	if not isinstance(target, str) or not target:
		raise ValueError(f"mapping[{idx}].target must be a non-empty string")

	source_shape_value = raw.get("source_shape", raw.get("shape"))
	source_shape = parse_shape(source_shape_value, field=f"mapping[{idx}].source_shape") if source_shape_value is not None else None

	target_shape_value = raw.get("target_shape")
	target_shape = parse_shape(target_shape_value, field=f"mapping[{idx}].target_shape") if target_shape_value is not None else None

	transform = raw.get("transform", "copy")
	if not isinstance(transform, str) or transform not in VALID_TRANSFORMS:
		raise ValueError(f"mapping[{idx}].transform must be one of {sorted(VALID_TRANSFORMS)}")

	role = raw.get("role", "other")
	if role == "objectness":
		raise ValueError("YOLOv9 DDetect mappings must not contain an objectness role")
	if not isinstance(role, str) or role not in VALID_ROLES:
		raise ValueError(f"mapping[{idx}].role must be one of {sorted(VALID_ROLES)}")

	return MappingEntry(
		source=source,
		target=target,
		source_shape=source_shape,
		target_shape=target_shape,
		transform=transform,
		role=role,
	)


def load_mappings(path: Path) -> list[MappingEntry]:
	data = load_json(path)
	raw_mappings = data.get("mappings") if isinstance(data, Mapping) else data
	if not isinstance(raw_mappings, list) or not raw_mappings:
		raise ValueError("mapping file must contain a non-empty mapping list")
	return [parse_mapping_entry(raw, idx) for idx, raw in enumerate(raw_mappings)]


def validate_mappings(
	tensors: Mapping[str, TensorSpec],
	mappings: Sequence[MappingEntry],
	*,
	allow_unmapped_source: bool,
	num_classes: int | None,
	reg_max: int,
) -> ValidationReport:
	errors: list[str] = []
	warnings: list[str] = []

	source_names: set[str] = set()
	target_names: set[str] = set()
	for entry in mappings:
		if entry.source in source_names:
			errors.append(f"duplicate source tensor in mapping: {entry.source}")
		source_names.add(entry.source)

		if entry.target in target_names:
			errors.append(f"duplicate target tensor in mapping: {entry.target}")
		target_names.add(entry.target)

		spec = tensors.get(entry.source)
		if spec is None:
			errors.append(f"mapped source tensor is missing from state_dict: {entry.source}")
			continue

		if entry.source_shape is not None and spec.shape != entry.source_shape:
			errors.append(
				f"{entry.source} shape mismatch: mapping expects {entry.source_shape}, checkpoint has {spec.shape}")

		if entry.transform == "copy" and entry.target_shape is not None and spec.shape != entry.target_shape:
			errors.append(
				f"{entry.source} copy transform cannot produce target shape {entry.target_shape} from {spec.shape}")

		if entry.role == "ddetect_box_logits":
			expected_channels = 4 * reg_max
			if not spec.shape:
				errors.append(f"{entry.source} ddetect_box_logits tensor must have at least one dimension")
			elif spec.shape[0] != expected_channels:
				errors.append(
					f"{entry.source} ddetect_box_logits first dimension must be {expected_channels}, got {spec.shape[0]}")

		if entry.role == "ddetect_class_logits":
			if num_classes is None:
				warnings.append(f"{entry.source} class-logit shape cannot be fully validated without --num-classes")
			elif not spec.shape:
				errors.append(f"{entry.source} ddetect_class_logits tensor must have at least one dimension")
			elif spec.shape[0] != num_classes:
				errors.append(
					f"{entry.source} ddetect_class_logits first dimension must be {num_classes}, got {spec.shape[0]}")

	unmapped_sources = sorted(set(tensors) - source_names)
	if unmapped_sources and not allow_unmapped_source:
		errors.append(
			f"{len(unmapped_sources)} source tensors are not mapped; pass --allow-unmapped-source only for partial fixtures")

	return ValidationReport(
		source_tensor_count=len(tensors),
		mapping_count=len(mappings),
		errors=errors,
		warnings=warnings,
		unmapped_sources=unmapped_sources,
	)


def write_fixture_manifest(path: Path, report: ValidationReport, mappings: Sequence[MappingEntry]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(report.to_json_dict(mappings), handle, indent=2, sort_keys=True)
		handle.write("\n")


def run_self_test() -> int:
	tensors = {
		"model.head.cv2.0.2.weight": TensorSpec("model.head.cv2.0.2.weight", (64, 32, 1, 1)),
		"model.head.cv2.0.2.bias": TensorSpec("model.head.cv2.0.2.bias", (64,)),
		"model.head.cv3.0.2.weight": TensorSpec("model.head.cv3.0.2.weight", (3, 32, 1, 1)),
		"model.head.cv3.0.2.bias": TensorSpec("model.head.cv3.0.2.bias", (3,)),
	}
	mappings = [
		MappingEntry("model.head.cv2.0.2.weight", "ddetect/0/box/weight", (64, 32, 1, 1), (64, 32, 1, 1), "copy", "ddetect_box_logits"),
		MappingEntry("model.head.cv2.0.2.bias", "ddetect/0/box/bias", (64,), (64,), "copy", "ddetect_box_logits"),
		MappingEntry("model.head.cv3.0.2.weight", "ddetect/0/class/weight", (3, 32, 1, 1), (3, 32, 1, 1), "copy", "ddetect_class_logits"),
		MappingEntry("model.head.cv3.0.2.bias", "ddetect/0/class/bias", (3,), (3,), "copy", "ddetect_class_logits"),
	]
	report = validate_mappings(tensors, mappings, allow_unmapped_source=False, num_classes=3, reg_max=16)
	if not report.ok:
		print("self-test unexpectedly failed:", file=sys.stderr)
		for error in report.errors:
			print(f"  {error}", file=sys.stderr)
		return 1

	bad_mapping = [
		MappingEntry("model.head.cv3.0.2.weight", "ddetect/0/box/weight", (3, 32, 1, 1), None, "copy", "ddetect_box_logits"),
	]
	bad_report = validate_mappings(tensors, bad_mapping, allow_unmapped_source=True, num_classes=3, reg_max=16)
	if bad_report.ok:
		print("self-test expected the bad DDetect box mapping to fail", file=sys.stderr)
		return 1

	print("self-test passed")
	return 0


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	source = parser.add_mutually_exclusive_group()
	source.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint to inspect with torch.load")
	source.add_argument("--state-manifest", type=Path, help="JSON tensor-shape manifest for fixture validation")
	parser.add_argument("--mapping", type=Path, help="JSON source-to-target mapping file")
	parser.add_argument("--allow-unmapped-source", action="store_true", help="allow checkpoint tensors not listed in the mapping")
	parser.add_argument("--num-classes", type=int, help="validate ddetect_class_logits output channels")
	parser.add_argument("--reg-max", type=int, default=16, help="DFL reg_max used to validate ddetect_box_logits channels")
	parser.add_argument("--write-fixture-manifest", type=Path, help="write a validation manifest JSON after successful validation")
	parser.add_argument("--darknet-weights-out", type=Path, help="reserved for the future writer; currently refused after validation")
	parser.add_argument("--self-test", action="store_true", help="run built-in validation tests without reading files")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)

	if args.self_test:
		return run_self_test()

	if args.mapping is None:
		parser.error("--mapping is required unless --self-test is used")
	if args.checkpoint is None and args.state_manifest is None:
		parser.error("one of --checkpoint or --state-manifest is required unless --self-test is used")
	if args.reg_max <= 0:
		parser.error("--reg-max must be positive")
	if args.num_classes is not None and args.num_classes <= 0:
		parser.error("--num-classes must be positive")

	try:
		tensors = load_checkpoint_shapes(args.checkpoint) if args.checkpoint else load_state_manifest(args.state_manifest)
		mappings = load_mappings(args.mapping)
		report = validate_mappings(
			tensors,
			mappings,
			allow_unmapped_source=args.allow_unmapped_source,
			num_classes=args.num_classes,
			reg_max=args.reg_max,
		)
	except Exception as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 2

	if report.warnings:
		for warning in report.warnings:
			print(f"warning: {warning}", file=sys.stderr)

	if not report.ok:
		print("mapping validation failed:", file=sys.stderr)
		for error in report.errors:
			print(f"  {error}", file=sys.stderr)
		return 1

	if args.write_fixture_manifest:
		write_fixture_manifest(args.write_fixture_manifest, report, mappings)
		print(f"wrote validation fixture manifest: {args.write_fixture_manifest}")

	if args.darknet_weights_out:
		print(
			"error: Darknet .weights writing is not implemented in this skeleton; refusing to write after validation",
			file=sys.stderr,
		)
		return 2

	print(f"validated {report.mapping_count} mappings against {report.source_tensor_count} tensors")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
