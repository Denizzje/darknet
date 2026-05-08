#!/usr/bin/env python3
"""Validate YOLOv9 PyTorch-to-Darknet tensor mappings and write .weights.

The tool is intentionally staged:

* with ``--state-manifest`` it validates source tensor coverage and YOLOv9 head
  tensor shapes without importing torch;
* with ``--cfg`` it also validates that every weighted Darknet target slot is
  mapped exactly once;
* with ``--checkpoint --cfg --darknet-weights-out`` it serializes a Darknet
  ``.weights`` file in the order expected by ``load_weights()``.

Mapping JSON may be either a list of entries or ``{"mappings": [...]}``:

[
  {
    "source": "model.0.conv.weight",
    "target": "convolutional_0/weights",
    "source_shape": [16, 3, 3, 3],
    "target_shape": [16, 3, 3, 3],
    "transform": "conv2d_oihw",
    "role": "backbone"
  }
]

Supported Darknet target aliases are:

* ``layer_12/weights`` or ``12/weights`` for absolute Darknet layer indices;
* ``convolutional_0/weights`` for convolutional layer ordinals;
* ``connected_0/weights`` for connected layer ordinals.

For convolutional layers, Darknet load order is biases, optional batchnorm
scales/rolling_mean/rolling_variance, then weights.  PyTorch Conv2d OIHW
weights are already in Darknet's flattened order.

State manifest JSON, useful for fixtures without torch, may be either
``{"name": [shape...]}`` or ``{"tensors": [{"name": "...", "shape": [...]}]}``.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import tempfile
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
	"ddetect_aux_box_logits",
	"ddetect_aux_class_logits",
	"ddetect_box_logits",
	"ddetect_class_logits",
	"ddetect_main_box_logits",
	"ddetect_main_class_logits",
	"neck",
	"other",
}

PARAM_ALIASES = {
	"bias": "biases",
	"biases": "biases",
	"beta": "biases",
	"mean": "rolling_mean",
	"rolling_mean": "rolling_mean",
	"running_mean": "rolling_mean",
	"var": "rolling_variance",
	"variance": "rolling_variance",
	"rolling_var": "rolling_variance",
	"rolling_variance": "rolling_variance",
	"running_var": "rolling_variance",
	"running_variance": "rolling_variance",
	"scale": "scales",
	"scales": "scales",
	"gamma": "scales",
	"weight": "weights",
	"weights": "weights",
}

DARKNET_WEIGHTS_VERSION = (0, 2, 5)


@dataclass(frozen=True)
class TensorSpec:
	name: str
	shape: tuple[int, ...]
	array: Any | None = None


@dataclass(frozen=True)
class MappingEntry:
	source: str
	target: str
	source_shape: tuple[int, ...] | None
	target_shape: tuple[int, ...] | None
	transform: str
	role: str


@dataclass(frozen=True)
class TargetSlot:
	layer_index: int
	layer_type: str
	ordinal: int
	param: str
	expected_shape: tuple[int, ...] | None

	@property
	def canonical(self) -> str:
		return f"layer_{self.layer_index}/{self.param}"


@dataclass(frozen=True)
class CfgWeightedLayer:
	layer_index: int
	layer_type: str
	ordinal: int
	params: Mapping[str, str]

	@property
	def skipped(self) -> bool:
		return self.params.get("dontload", "0") not in {"0", "false", "False", ""}

	@property
	def share_layer(self) -> bool:
		return "share_index" in self.params

	def int_param(self, name: str, default: int) -> int:
		value = self.params.get(name)
		if value is None:
			return default
		try:
			return int(value)
		except ValueError as exc:
			raise ValueError(f"layer {self.layer_index} {name} must be an integer, got {value!r}") from exc

	def slots(self) -> list[TargetSlot]:
		if self.skipped or self.share_layer:
			return []

		if self.layer_type == "convolutional":
			filters = self.int_param("filters", 1)
			batch_normalize = self.int_param("batch_normalize", 0) != 0 or self.int_param("cbn", 0) != 0
			dontloadscales = self.int_param("dontloadscales", 0) != 0
			slots = [
				TargetSlot(self.layer_index, self.layer_type, self.ordinal, "biases", (filters,)),
			]
			if batch_normalize and not dontloadscales:
				slots.extend(
					[
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "scales", (filters,)),
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "rolling_mean", (filters,)),
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "rolling_variance", (filters,)),
					]
				)
			slots.append(TargetSlot(self.layer_index, self.layer_type, self.ordinal, "weights", None))
			return slots

		if self.layer_type == "connected":
			outputs = self.int_param("output", self.int_param("outputs", 1))
			batch_normalize = self.int_param("batch_normalize", 0) != 0
			slots = [
				TargetSlot(self.layer_index, self.layer_type, self.ordinal, "biases", (outputs,)),
				TargetSlot(self.layer_index, self.layer_type, self.ordinal, "weights", None),
			]
			if batch_normalize:
				slots.extend(
					[
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "scales", (outputs,)),
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "rolling_mean", (outputs,)),
						TargetSlot(self.layer_index, self.layer_type, self.ordinal, "rolling_variance", (outputs,)),
					]
				)
			return slots

		return []


@dataclass
class TargetValidation:
	errors: list[str]
	unused_targets: list[str]
	resolved_targets: dict[str, str]


@dataclass
class ValidationReport:
	source_tensor_count: int
	mapping_count: int
	errors: list[str]
	warnings: list[str]
	unmapped_sources: list[str]
	unused_targets: list[str]
	resolved_targets: dict[str, str]
	written_weights: str | None = None
	written_bytes: int | None = None

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
			"unused_targets": self.unused_targets,
			"resolved_targets": self.resolved_targets,
			"written_weights": self.written_weights,
			"written_bytes": self.written_bytes,
			"validated_mappings": [
				{
					"source": entry.source,
					"target": entry.target,
					"resolved_target": self.resolved_targets.get(entry.target),
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


def torch_load(path: Path) -> Any:
	try:
		import torch  # type: ignore
	except ImportError as exc:
		raise RuntimeError("loading PyTorch checkpoints requires torch; use --state-manifest for fixture-only validation") from exc

	try:
		return torch.load(path, map_location="cpu", weights_only=False)
	except TypeError:
		return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
	if hasattr(checkpoint, "state_dict"):
		return checkpoint.state_dict()
	if isinstance(checkpoint, Mapping):
		for key in ("ema", "model"):
			value = checkpoint.get(key)
			if hasattr(value, "state_dict"):
				return value.state_dict()
			if isinstance(value, Mapping):
				return value
		if isinstance(checkpoint.get("state_dict"), Mapping):
			return checkpoint["state_dict"]
		return checkpoint
	raise ValueError("checkpoint does not look like a module or state_dict")


def load_reference_model_state(path: Path, reference_yolov9_root: Path) -> Mapping[str, Any]:
	try:
		import torch  # type: ignore
	except ImportError as exc:
		raise RuntimeError("--fuse-reference-model requires torch") from exc

	root = reference_yolov9_root.resolve()
	if not (root / "models" / "experimental.py").exists():
		raise ValueError(f"{root} does not look like the YOLOv9 repo root")
	sys.path.insert(0, str(root))
	try:
		from models.experimental import attempt_load  # type: ignore

		model = attempt_load(str(path), device=torch.device("cpu"), fuse=True)
	finally:
		with contextlib_suppress_value_error():
			sys.path.remove(str(root))
	if hasattr(model, "module"):
		model = model.module
	return model.state_dict()


class contextlib_suppress_value_error:
	def __enter__(self) -> None:
		return None

	def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
		return exc_type is ValueError


def state_dict_to_specs(state_dict: Mapping[str, Any], *, include_arrays: bool) -> dict[str, TensorSpec]:
	tensors: dict[str, TensorSpec] = {}
	for name, tensor in state_dict.items():
		if not hasattr(tensor, "shape"):
			continue
		shape = tuple(int(dim) for dim in tensor.shape)
		array = None
		if include_arrays:
			if not hasattr(tensor, "detach"):
				raise ValueError(f"{name} is not a torch tensor and cannot be serialized")
			array = tensor.detach().cpu().float().contiguous().numpy()
		tensors[str(name)] = TensorSpec(name=str(name), shape=shape, array=array)

	if not tensors:
		raise ValueError("checkpoint did not expose any tensor-shaped state_dict entries")
	return tensors


def load_checkpoint_tensors(path: Path, *, include_arrays: bool, fuse_reference_model: bool, reference_yolov9_root: Path | None) -> dict[str, TensorSpec]:
	if fuse_reference_model:
		if reference_yolov9_root is None:
			raise ValueError("--fuse-reference-model requires --reference-yolov9-root")
		state_dict = load_reference_model_state(path, reference_yolov9_root)
	else:
		state_dict = extract_state_dict(torch_load(path))
	return state_dict_to_specs(state_dict, include_arrays=include_arrays)


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


def strip_cfg_comment(line: str) -> str:
	for marker in ("#", ";"):
		idx = line.find(marker)
		if idx >= 0:
			line = line[:idx]
	return line.strip()


def parse_darknet_cfg(path: Path) -> list[CfgWeightedLayer]:
	layers: list[CfgWeightedLayer] = []
	section_name: str | None = None
	params: dict[str, str] = {}
	layer_index = -1
	ordinals = {"convolutional": 0, "connected": 0}

	def flush() -> None:
		nonlocal params, section_name, layer_index
		if section_name is None or section_name == "net":
			return
		if section_name in ordinals:
			ordinal = ordinals[section_name]
			ordinals[section_name] += 1
			layers.append(CfgWeightedLayer(layer_index, section_name, ordinal, dict(params)))

	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = strip_cfg_comment(raw_line)
		if not line:
			continue
		if line.startswith("[") and line.endswith("]"):
			flush()
			section_name = line[1:-1].strip().lower()
			params = {}
			if section_name != "net":
				layer_index += 1
			continue
		if "=" in line and section_name is not None:
			key, value = line.split("=", 1)
			params[key.strip().lower()] = value.strip()
	flush()
	return layers


def normalise_param_name(value: str) -> str | None:
	return PARAM_ALIASES.get(value.strip().lower())


def parse_target_ref(value: str) -> tuple[str, int, str]:
	clean = value.strip().lower().replace("\\", "/")
	parts = [part for part in re.split(r"[/.:]+", clean) if part]
	if len(parts) < 2:
		raise ValueError("target must end with a parameter name")
	param = normalise_param_name(parts[-1])
	if param is None:
		raise ValueError(f"unknown target parameter {parts[-1]!r}")
	head = "_".join(parts[:-1])
	if re.fullmatch(r"\d+", head):
		return ("layer", int(head), param)
	match = re.fullmatch(r"(layer|darknet|convolutional|conv|connected|fc)[_-]?(\d+)", head)
	if match:
		namespace = match.group(1)
		if namespace == "darknet":
			namespace = "layer"
		if namespace == "conv":
			namespace = "convolutional"
		if namespace == "fc":
			namespace = "connected"
		return (namespace, int(match.group(2)), param)
	raise ValueError("target must identify a layer index or convolutional/connected ordinal")


def build_target_aliases(cfg_layers: Sequence[CfgWeightedLayer]) -> tuple[dict[tuple[str, int, str], TargetSlot], list[TargetSlot]]:
	aliases: dict[tuple[str, int, str], TargetSlot] = {}
	slots: list[TargetSlot] = []
	for layer in cfg_layers:
		for slot in layer.slots():
			slots.append(slot)
			aliases[("layer", slot.layer_index, slot.param)] = slot
			aliases[(slot.layer_type, slot.ordinal, slot.param)] = slot
	return aliases, slots


def validate_targets(
	mappings: Sequence[MappingEntry],
	cfg_layers: Sequence[CfgWeightedLayer] | None,
	*,
	allow_unmapped_target: bool,
) -> TargetValidation:
	if cfg_layers is None:
		return TargetValidation(errors=[], unused_targets=[], resolved_targets={})

	aliases, expected_slots = build_target_aliases(cfg_layers)
	errors: list[str] = []
	resolved_targets: dict[str, str] = {}
	seen_slots: dict[str, str] = {}

	for entry in mappings:
		try:
			ref = parse_target_ref(entry.target)
		except ValueError as exc:
			errors.append(f"{entry.target}: {exc}")
			continue
		slot = aliases.get(ref)
		if slot is None:
			errors.append(f"{entry.target}: target slot does not exist in cfg")
			continue
		resolved_targets[entry.target] = slot.canonical
		if slot.canonical in seen_slots:
			errors.append(f"duplicate target tensor in mapping: {entry.target} and {seen_slots[slot.canonical]} both map to {slot.canonical}")
		seen_slots[slot.canonical] = entry.target
		if slot.expected_shape is not None:
			if entry.target_shape is not None and entry.target_shape != slot.expected_shape:
				errors.append(f"{entry.target} target_shape mismatch: cfg expects {slot.expected_shape}, mapping has {entry.target_shape}")

	unused_targets = sorted(slot.canonical for slot in expected_slots if slot.canonical not in seen_slots)
	if unused_targets and not allow_unmapped_target:
		errors.append(f"{len(unused_targets)} Darknet target slots are not mapped")
	return TargetValidation(errors=errors, unused_targets=unused_targets, resolved_targets=resolved_targets)


def is_box_logit_role(role: str) -> bool:
	return role in {"ddetect_box_logits", "ddetect_aux_box_logits", "ddetect_main_box_logits"}


def is_class_logit_role(role: str) -> bool:
	return role in {"ddetect_class_logits", "ddetect_aux_class_logits", "ddetect_main_class_logits"}


def validate_mappings(
	tensors: Mapping[str, TensorSpec],
	mappings: Sequence[MappingEntry],
	*,
	allow_unmapped_source: bool,
	allow_unmapped_target: bool,
	cfg_layers: Sequence[CfgWeightedLayer] | None,
	num_classes: int | None,
	reg_max: int,
) -> ValidationReport:
	errors: list[str] = []
	warnings: list[str] = []

	source_names: set[str] = set()
	for entry in mappings:
		if entry.source in source_names:
			errors.append(f"duplicate source tensor in mapping: {entry.source}")
		source_names.add(entry.source)

		spec = tensors.get(entry.source)
		if spec is None:
			errors.append(f"mapped source tensor is missing from state_dict: {entry.source}")
			continue

		if entry.source_shape is not None and spec.shape != entry.source_shape:
			errors.append(f"{entry.source} shape mismatch: mapping expects {entry.source_shape}, checkpoint has {spec.shape}")

		if entry.transform == "copy" and entry.target_shape is not None and spec.shape != entry.target_shape:
			errors.append(f"{entry.source} copy transform cannot produce target shape {entry.target_shape} from {spec.shape}")

		if is_box_logit_role(entry.role):
			expected_channels = 4 * reg_max
			if not spec.shape:
				errors.append(f"{entry.source} {entry.role} tensor must have at least one dimension")
			elif spec.shape[0] != expected_channels:
				errors.append(f"{entry.source} {entry.role} first dimension must be {expected_channels}, got {spec.shape[0]}")

		if is_class_logit_role(entry.role):
			if num_classes is None:
				warnings.append(f"{entry.source} class-logit shape cannot be fully validated without --num-classes")
			elif not spec.shape:
				errors.append(f"{entry.source} {entry.role} tensor must have at least one dimension")
			elif spec.shape[0] != num_classes:
				errors.append(f"{entry.source} {entry.role} first dimension must be {num_classes}, got {spec.shape[0]}")

	unmapped_sources = sorted(set(tensors) - source_names)
	if unmapped_sources and not allow_unmapped_source:
		errors.append(f"{len(unmapped_sources)} source tensors are not mapped; pass --allow-unmapped-source only for partial fixtures")

	target_report = validate_targets(mappings, cfg_layers, allow_unmapped_target=allow_unmapped_target)
	errors.extend(target_report.errors)

	return ValidationReport(
		source_tensor_count=len(tensors),
		mapping_count=len(mappings),
		errors=errors,
		warnings=warnings,
		unmapped_sources=unmapped_sources,
		unused_targets=target_report.unused_targets,
		resolved_targets=target_report.resolved_targets,
	)


def write_report(path: Path, report: ValidationReport, mappings: Sequence[MappingEntry]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(report.to_json_dict(mappings), handle, indent=2, sort_keys=True)
		handle.write("\n")


def tensor_to_float32_array(spec: TensorSpec, entry: MappingEntry) -> Any:
	if spec.array is None:
		raise ValueError(f"{entry.source} has no loaded tensor data; use --checkpoint for writing")
	try:
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("writing Darknet weights requires numpy") from exc

	array = np.asarray(spec.array, dtype="<f4")
	if entry.target_shape is not None and tuple(int(dim) for dim in array.shape) != entry.target_shape:
		raise ValueError(f"{entry.source} cannot write {entry.target}: source data shape {array.shape} != target_shape {entry.target_shape}")
	if entry.transform in {"batchnorm", "conv2d_oihw", "copy", "linear"}:
		return np.ascontiguousarray(array.reshape(-1), dtype="<f4")
	raise ValueError(f"unsupported transform for writing: {entry.transform}")


def write_darknet_weights(
	path: Path,
	tensors: Mapping[str, TensorSpec],
	mappings: Sequence[MappingEntry],
	cfg_layers: Sequence[CfgWeightedLayer],
	resolved_targets: Mapping[str, str],
) -> int:
	by_canonical = {resolved_targets[entry.target]: entry for entry in mappings if entry.target in resolved_targets}
	path.parent.mkdir(parents=True, exist_ok=True)

	bytes_written = 0
	with path.open("wb") as handle:
		header = struct.pack("<iiiQ", *DARKNET_WEIGHTS_VERSION, 0)
		handle.write(header)
		bytes_written += len(header)
		for layer in cfg_layers:
			for slot in layer.slots():
				entry = by_canonical.get(slot.canonical)
				if entry is None:
					raise ValueError(f"missing mapping for {slot.canonical}")
				array = tensor_to_float32_array(tensors[entry.source], entry)
				if slot.expected_shape is not None and tuple(array.shape) != (slot.expected_shape[0],):
					raise ValueError(f"{entry.source} cannot write {slot.canonical}: expected {slot.expected_shape}, got {tuple(array.shape)}")
				handle.write(array.tobytes(order="C"))
				bytes_written += int(array.nbytes)
	return bytes_written


def run_self_test() -> int:
	try:
		import numpy as np
	except ImportError:
		print("self-test requires numpy", file=sys.stderr)
		return 2

	tensors = {
		"model.head.cv2.0.2.weight": TensorSpec("model.head.cv2.0.2.weight", (64, 32, 1, 1), np.zeros((64, 32, 1, 1), dtype=np.float32)),
		"model.head.cv2.0.2.bias": TensorSpec("model.head.cv2.0.2.bias", (64,), np.zeros((64,), dtype=np.float32)),
		"model.head.cv3.0.2.weight": TensorSpec("model.head.cv3.0.2.weight", (3, 32, 1, 1), np.zeros((3, 32, 1, 1), dtype=np.float32)),
		"model.head.cv3.0.2.bias": TensorSpec("model.head.cv3.0.2.bias", (3,), np.zeros((3,), dtype=np.float32)),
	}
	mappings = [
		MappingEntry("model.head.cv2.0.2.weight", "ddetect/0/box/weight", (64, 32, 1, 1), (64, 32, 1, 1), "copy", "ddetect_box_logits"),
		MappingEntry("model.head.cv2.0.2.bias", "ddetect/0/box/bias", (64,), (64,), "copy", "ddetect_box_logits"),
		MappingEntry("model.head.cv3.0.2.weight", "ddetect/0/class/weight", (3, 32, 1, 1), (3, 32, 1, 1), "copy", "ddetect_class_logits"),
		MappingEntry("model.head.cv3.0.2.bias", "ddetect/0/class/bias", (3,), (3,), "copy", "ddetect_class_logits"),
	]
	report = validate_mappings(
		tensors,
		mappings,
		allow_unmapped_source=False,
		allow_unmapped_target=False,
		cfg_layers=None,
		num_classes=3,
		reg_max=16,
	)
	if not report.ok:
		print("self-test unexpectedly failed:", file=sys.stderr)
		for error in report.errors:
			print(f"  {error}", file=sys.stderr)
		return 1

	bad_mapping = [
		MappingEntry("model.head.cv3.0.2.weight", "ddetect/0/box/weight", (3, 32, 1, 1), None, "copy", "ddetect_box_logits"),
	]
	bad_report = validate_mappings(
		tensors,
		bad_mapping,
		allow_unmapped_source=True,
		allow_unmapped_target=False,
		cfg_layers=None,
		num_classes=3,
		reg_max=16,
	)
	if bad_report.ok:
		print("self-test expected the bad DDetect box mapping to fail", file=sys.stderr)
		return 1

	with tempfile.TemporaryDirectory() as tmpdir:
		cfg_path = Path(tmpdir) / "tiny.cfg"
		cfg_path.write_text(
			"""
[net]
batch=1
subdivisions=1
width=1
height=1
channels=1

[convolutional]
batch_normalize=1
filters=2
size=1
stride=1
pad=1
activation=linear
""".lstrip(),
			encoding="utf-8",
		)
		cfg_layers = parse_darknet_cfg(cfg_path)
		write_tensors = {
			"b": TensorSpec("b", (2,), np.array([1, 2], dtype=np.float32)),
			"s": TensorSpec("s", (2,), np.array([3, 4], dtype=np.float32)),
			"m": TensorSpec("m", (2,), np.array([5, 6], dtype=np.float32)),
			"v": TensorSpec("v", (2,), np.array([7, 8], dtype=np.float32)),
			"w": TensorSpec("w", (2, 1, 1, 1), np.array([[[[9]]], [[[10]]]], dtype=np.float32)),
		}
		write_mappings = [
			MappingEntry("b", "convolutional_0/biases", (2,), (2,), "copy", "other"),
			MappingEntry("s", "convolutional_0/scales", (2,), (2,), "batchnorm", "other"),
			MappingEntry("m", "convolutional_0/rolling_mean", (2,), (2,), "batchnorm", "other"),
			MappingEntry("v", "convolutional_0/rolling_variance", (2,), (2,), "batchnorm", "other"),
			MappingEntry("w", "convolutional_0/weights", (2, 1, 1, 1), (2, 1, 1, 1), "conv2d_oihw", "other"),
		]
		write_report_obj = validate_mappings(
			write_tensors,
			write_mappings,
			allow_unmapped_source=False,
			allow_unmapped_target=False,
			cfg_layers=cfg_layers,
			num_classes=None,
			reg_max=16,
		)
		if not write_report_obj.ok:
			print("writer self-test validation unexpectedly failed:", file=sys.stderr)
			for error in write_report_obj.errors:
				print(f"  {error}", file=sys.stderr)
			return 1
		out_path = Path(tmpdir) / "tiny.weights"
		bytes_written = write_darknet_weights(out_path, write_tensors, write_mappings, cfg_layers, write_report_obj.resolved_targets)
		if bytes_written != 20 + 10 * 4:
			print(f"writer self-test wrote unexpected byte count: {bytes_written}", file=sys.stderr)
			return 1

	print("self-test passed")
	return 0


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	source = parser.add_mutually_exclusive_group()
	source.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint to inspect with torch.load")
	source.add_argument("--state-manifest", type=Path, help="JSON tensor-shape manifest for fixture validation")
	parser.add_argument("--mapping", type=Path, help="JSON source-to-target mapping file")
	parser.add_argument("--cfg", type=Path, help="Darknet cfg used for target-slot validation and .weights write order")
	parser.add_argument("--reference-yolov9-root", type=Path, help="reference_yolov9_repo/yolov9 root for --fuse-reference-model")
	parser.add_argument("--fuse-reference-model", action="store_true", help="load checkpoint via YOLOv9 attempt_load(..., fuse=True)")
	parser.add_argument("--allow-unmapped-source", action="store_true", help="allow checkpoint tensors not listed in the mapping")
	parser.add_argument("--allow-unmapped-target", action="store_true", help="allow cfg weighted target slots not listed in the mapping")
	parser.add_argument("--num-classes", type=int, help="validate ddetect class-logit output channels")
	parser.add_argument("--reg-max", type=int, default=16, help="DFL reg_max used to validate ddetect box-logit channels")
	parser.add_argument("--write-fixture-manifest", type=Path, help="deprecated alias for --conversion-report")
	parser.add_argument("--conversion-report", type=Path, help="write a conversion validation report JSON")
	parser.add_argument("--darknet-weights-out", type=Path, help="write a Darknet .weights file after strict validation")
	parser.add_argument("--self-test", action="store_true", help="run built-in validation and writer tests without reading inputs")
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
	if args.darknet_weights_out and args.state_manifest:
		parser.error("--darknet-weights-out requires --checkpoint, not --state-manifest")
	if args.darknet_weights_out and args.cfg is None:
		parser.error("--darknet-weights-out requires --cfg")
	if args.darknet_weights_out and args.allow_unmapped_target:
		parser.error("--darknet-weights-out cannot be used with --allow-unmapped-target")
	if args.reg_max <= 0:
		parser.error("--reg-max must be positive")
	if args.num_classes is not None and args.num_classes <= 0:
		parser.error("--num-classes must be positive")

	try:
		include_arrays = bool(args.darknet_weights_out)
		tensors = (
			load_checkpoint_tensors(
				args.checkpoint,
				include_arrays=include_arrays,
				fuse_reference_model=args.fuse_reference_model,
				reference_yolov9_root=args.reference_yolov9_root,
			)
			if args.checkpoint
			else load_state_manifest(args.state_manifest)
		)
		mappings = load_mappings(args.mapping)
		cfg_layers = parse_darknet_cfg(args.cfg) if args.cfg else None
		report = validate_mappings(
			tensors,
			mappings,
			allow_unmapped_source=args.allow_unmapped_source,
			allow_unmapped_target=args.allow_unmapped_target,
			cfg_layers=cfg_layers,
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
		if report.unmapped_sources:
			print(f"  unmapped source tensors: {len(report.unmapped_sources)}", file=sys.stderr)
		if report.unused_targets:
			print(f"  unused target slots: {len(report.unused_targets)}", file=sys.stderr)
		report_path = args.conversion_report or args.write_fixture_manifest
		if report_path:
			write_report(report_path, report, mappings)
			print(f"wrote failed conversion report: {report_path}", file=sys.stderr)
		return 1

	if args.darknet_weights_out:
		try:
			assert cfg_layers is not None
			bytes_written = write_darknet_weights(args.darknet_weights_out, tensors, mappings, cfg_layers, report.resolved_targets)
			report.written_weights = str(args.darknet_weights_out)
			report.written_bytes = bytes_written
		except Exception as exc:
			print(f"error: failed to write Darknet weights: {exc}", file=sys.stderr)
			return 2
		print(f"wrote Darknet weights: {args.darknet_weights_out} ({bytes_written} bytes)")

	report_path = args.conversion_report or args.write_fixture_manifest
	if report_path:
		write_report(report_path, report, mappings)
		print(f"wrote conversion report: {report_path}")

	target_note = f" and {len(report.resolved_targets)} cfg target slots" if args.cfg else ""
	print(f"validated {report.mapping_count} mappings against {report.source_tensor_count} tensors{target_note}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
