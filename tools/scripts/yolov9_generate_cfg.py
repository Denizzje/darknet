#!/usr/bin/env python3
"""Generate Darknet cfg files for the YOLOv9/GELAN detection topologies.

The generator intentionally supports the small detection model family used by
the local YOLOv9 training plan: GELAN-t/s and YOLOv9-t/s.  It reads the
reference YAML topology, expands modules into current Darknet primitives, and
tracks tensor shapes while generating the cfg so bad routes fail early.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
	import yaml
except ImportError:  # pragma: no cover - exercised only on minimal Python envs
	yaml = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_DIR = ROOT / "reference_yolov9_repo" / "yolov9" / "models" / "detect"
DEFAULT_OUTPUT_DIR = ROOT / "cfg"
DEFAULT_MODELS = ("gelan-t", "gelan-s", "yolov9-t", "yolov9-s")
REG_MAX = 16
BOX_LOGIT_CHANNELS = 4 * REG_MAX


class GenerateError(RuntimeError):
	"""Raised when the reference topology cannot be lowered safely."""


@dataclass(frozen=True)
class Shape:
	w: int
	h: int
	c: int

	def describe(self) -> str:
		return f"{self.w} x {self.h} x {self.c}"


@dataclass(frozen=True)
class Profile:
	name: str
	suffix: str
	classes: int
	width: int
	height: int
	batch: int
	subdivisions: int
	max_batches: int
	steps: tuple[int, int]
	dataset_comment: str | None = None
	training_gate: str | None = None


PROFILES: Mapping[str, Profile] = {
	"coco": Profile(
		name="coco",
		suffix="",
		classes=80,
		width=640,
		height=640,
		batch=64,
		subdivisions=1,
		max_batches=2000200,
		steps=(1600000, 1800000),
		dataset_comment="COCO-style 80-class detector defaults.",
	),
	"legogears": Profile(
		name="legogears",
		suffix="-legogears",
		classes=5,
		width=224,
		height=160,
		batch=64,
		subdivisions=1,
		max_batches=3000,
		steps=(2400, 2700),
		dataset_comment="test_training_set/LegoGears_v2/LegoGears.local.data",
		training_gate="./darknet detector train test_training_set/LegoGears_v2/LegoGears.local.data {cfg_path} -dont_show -map",
	),
}


@dataclass
class Section:
	kind: str
	options: list[tuple[str, int | float | str]]
	shape: Shape | None
	comment: str | None = None

	def render(self, idx: int) -> list[str]:
		lines: list[str] = []
		if self.comment:
			shape_suffix = "" if self.shape is None else f" -> {self.shape.describe()}"
			lines.append(f"# {idx} - {self.comment}{shape_suffix}")
		lines.append(f"[{self.kind}]")
		for key, value in self.options:
			lines.append(f"{key}={value}")
		return lines


def make_divisible(value: int | float, divisor: int) -> int:
	return int(math.ceil(float(value) / divisor) * divisor)


def load_reference_yaml(path: Path) -> dict[str, Any]:
	if yaml is None:
		raise GenerateError("PyYAML is required to read YOLOv9 reference YAML files")
	try:
		with path.open("r", encoding="utf-8") as handle:
			data = yaml.safe_load(handle)
	except OSError as exc:
		raise GenerateError(f"cannot read reference YAML {path}: {exc}") from exc
	if not isinstance(data, dict):
		raise GenerateError(f"{path} did not load as a YAML mapping")
	for key in ("backbone", "head"):
		if key not in data or not isinstance(data[key], list):
			raise GenerateError(f"{path} is missing a {key!r} list")
	return data


def as_int_list(value: Any, *, context: str) -> list[int]:
	if not isinstance(value, list) or not value:
		raise GenerateError(f"{context} must be a non-empty list")
	result: list[int] = []
	for item in value:
		if not isinstance(item, int):
			raise GenerateError(f"{context} contains non-integer source {item!r}")
		result.append(item)
	return result


def format_args(args: Sequence[Any]) -> str:
	return "[" + ", ".join(repr(arg) for arg in args) + "]"


class CfgBuilder:
	def __init__(self, *, model: str, profile: Profile, source_yaml: Path) -> None:
		self.model = model
		self.profile = profile
		self.source_yaml = source_yaml
		self.sections: list[Section] = []
		self.current_shape = Shape(profile.width, profile.height, 3)
		self.yaml_outputs: dict[int, int] = {}
		self.mapping_notes: list[str] = []

		if profile.width % 32 != 0 or profile.height % 32 != 0:
			raise GenerateError(
				f"{profile.name} dimensions must be divisible by 32, got {profile.width}x{profile.height}")

	def last_idx(self) -> int | None:
		return len(self.sections) - 1 if self.sections else None

	def add_section(
		self,
		kind: str,
		options: list[tuple[str, int | float | str]],
		shape: Shape | None,
		comment: str | None = None,
	) -> int:
		idx = len(self.sections)
		self.sections.append(Section(kind=kind, options=options, shape=shape, comment=comment))
		if shape is not None:
			self.current_shape = shape
		return idx

	def section_shape(self, idx: int) -> Shape:
		try:
			shape = self.sections[idx].shape
		except IndexError as exc:
			raise GenerateError(f"internal error: layer index {idx} is outside generated cfg") from exc
		if shape is None:
			raise GenerateError(f"layer {idx} does not produce image-shaped output")
		return shape

	def conv(
		self,
		filters: int,
		*,
		size: int = 1,
		stride: int = 1,
		groups: int = 1,
		batch_normalize: bool = True,
		activation: str = "swish",
		comment: str | None = None,
	) -> int:
		if filters <= 0:
			raise GenerateError(f"convolution filters must be positive, got {filters}")
		if size <= 0 or stride <= 0:
			raise GenerateError(f"convolution size/stride must be positive, got size={size}, stride={stride}")
		if groups <= 0:
			raise GenerateError(f"convolution groups must be positive, got {groups}")
		if self.current_shape.c % groups != 0:
			raise GenerateError(
				f"cannot group convolution input with {self.current_shape.c} channels into {groups} groups")
		if filters % groups != 0:
			raise GenerateError(f"cannot group convolution output with {filters} filters into {groups} groups")

		padding = size // 2
		out_w = (self.current_shape.w + 2 * padding - size) // stride + 1
		out_h = (self.current_shape.h + 2 * padding - size) // stride + 1
		if out_w <= 0 or out_h <= 0:
			raise GenerateError(
				f"convolution would produce invalid shape from {self.current_shape.describe()}")

		options: list[tuple[str, int | float | str]] = []
		if batch_normalize:
			options.append(("batch_normalize", 1))
		options.append(("filters", filters))
		if groups != 1:
			options.append(("groups", groups))
		options.extend([
			("size", size),
			("stride", stride),
			("pad", 1),
			("activation", activation),
		])
		return self.add_section("convolutional", options, Shape(out_w, out_h, filters), comment)

	def route(
		self,
		layers: Sequence[int],
		*,
		groups: int = 1,
		group_id: int = 0,
		comment: str | None = None,
	) -> int:
		if not layers:
			raise GenerateError("route requires at least one input layer")
		if groups <= 0:
			raise GenerateError(f"route groups must be positive, got {groups}")
		if group_id < 0 or group_id >= groups:
			raise GenerateError(f"route group_id={group_id} is outside [0, {groups})")

		shapes = [self.section_shape(layer) for layer in layers]
		first = shapes[0]
		for layer, shape in zip(layers[1:], shapes[1:]):
			if shape.w != first.w or shape.h != first.h:
				raise GenerateError(
					f"route shape mismatch: layer {layers[0]} has {first.describe()}, "
					f"layer {layer} has {shape.describe()}")
		channels = sum(shape.c for shape in shapes)
		if channels % groups != 0:
			raise GenerateError(f"route channels={channels} cannot be split into {groups} groups")

		options: list[tuple[str, int | float | str]] = [("layers", ",".join(str(layer) for layer in layers))]
		if groups != 1:
			options.extend([("groups", groups), ("group_id", group_id)])
		return self.add_section("route", options, Shape(first.w, first.h, channels // groups), comment)

	def shortcut(self, from_layer: int, *, activation: str = "linear", comment: str | None = None) -> int:
		from_shape = self.section_shape(from_layer)
		if from_shape != self.current_shape:
			raise GenerateError(
				f"shortcut shape mismatch: current {self.current_shape.describe()} and "
				f"from layer {from_layer} {from_shape.describe()}")
		return self.add_section(
			"shortcut",
			[("from", from_layer), ("activation", activation)],
			Shape(self.current_shape.w, self.current_shape.h, self.current_shape.c),
			comment,
		)

	def local_avgpool(self, *, size: int = 2, stride: int = 1, padding: int = 0, comment: str | None = None) -> int:
		out_w = (self.current_shape.w + padding - size) // stride + 1
		out_h = (self.current_shape.h + padding - size) // stride + 1
		if out_w <= 0 or out_h <= 0:
			raise GenerateError(f"local_avgpool would produce invalid shape from {self.current_shape.describe()}")
		return self.add_section(
			"local_avgpool",
			[("size", size), ("stride", stride), ("padding", padding)],
			Shape(out_w, out_h, self.current_shape.c),
			comment,
		)

	def maxpool(self, *, size: int = 5, stride: int = 1, comment: str | None = None) -> int:
		padding = size - 1
		out_w = (self.current_shape.w + padding - size) // stride + 1
		out_h = (self.current_shape.h + padding - size) // stride + 1
		if out_w <= 0 or out_h <= 0:
			raise GenerateError(f"maxpool would produce invalid shape from {self.current_shape.describe()}")
		return self.add_section(
			"maxpool",
			[("size", size), ("stride", stride)],
			Shape(out_w, out_h, self.current_shape.c),
			comment,
		)

	def upsample(self, *, stride: int = 2, comment: str | None = None) -> int:
		if stride <= 0:
			raise GenerateError(f"upsample stride must be positive, got {stride}")
		return self.add_section(
			"upsample",
			[("stride", stride)],
			Shape(self.current_shape.w * stride, self.current_shape.h * stride, self.current_shape.c),
			comment,
		)

	def resolve_source(self, source: int, yaml_idx: int) -> int | None:
		if source == -1:
			last = self.last_idx()
			if last is None:
				return None
			return last
		if source < 0:
			source = yaml_idx + source
		if source not in self.yaml_outputs:
			raise GenerateError(f"YAML layer {yaml_idx} references unavailable source {source}")
		return self.yaml_outputs[source]

	def ensure_source(self, source: int, yaml_idx: int, *, comment: str) -> None:
		source_idx = self.resolve_source(source, yaml_idx)
		if source_idx is None:
			return
		if source_idx == self.last_idx():
			return
		self.route([source_idx], comment=comment)

	def add_conv_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) < 1:
			raise GenerateError(f"YAML {yaml_idx} Conv requires at least output channels")
		filters = int(args[0])
		size = int(args[1]) if len(args) > 1 else 1
		stride = int(args[2]) if len(args) > 2 else 1
		groups = int(args[4]) if len(args) > 4 else 1
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} Conv")
		return self.conv(filters, size=size, stride=stride, groups=groups, comment=f"YAML {yaml_idx} Conv {format_args(args)}")

	def add_aconv_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) != 1:
			raise GenerateError(f"YAML {yaml_idx} AConv expects [channels], got {args!r}")
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} AConv")
		self.local_avgpool(comment=f"YAML {yaml_idx} AConv pre-pool")
		return self.conv(int(args[0]), size=3, stride=2, comment=f"YAML {yaml_idx} AConv conv {format_args(args)}")

	def add_elan1_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) != 3:
			raise GenerateError(f"YAML {yaml_idx} ELAN1 expects [c2, c3, c4], got {args!r}")
		c2, c3, c4 = (int(value) for value in args)
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} ELAN1")
		cv1 = self.conv(c3, size=1, comment=f"YAML {yaml_idx} ELAN1 cv1")
		chunk0 = self.route([cv1], groups=2, group_id=0, comment=f"YAML {yaml_idx} ELAN1 chunk 0")
		chunk1 = self.route([cv1], groups=2, group_id=1, comment=f"YAML {yaml_idx} ELAN1 chunk 1")
		cv2 = self.conv(c4, size=3, comment=f"YAML {yaml_idx} ELAN1 cv2")
		cv3 = self.conv(c4, size=3, comment=f"YAML {yaml_idx} ELAN1 cv3")
		self.route([chunk0, chunk1, cv2, cv3], comment=f"YAML {yaml_idx} ELAN1 concat")
		return self.conv(c2, size=1, comment=f"YAML {yaml_idx} ELAN1 cv4")

	def add_repconvn(self, c2: int, *, prefix: str, groups: int = 1) -> int:
		input_idx = self.last_idx()
		if input_idx is None:
			raise GenerateError("RepConvN cannot consume net input directly")
		self.conv(c2, size=3, groups=groups, activation="linear", comment=f"{prefix} RepConvN conv1 3x3")
		conv1 = self.last_idx()
		assert conv1 is not None
		self.route([input_idx], comment=f"{prefix} RepConvN conv2 input")
		self.conv(c2, size=1, groups=groups, activation="linear", comment=f"{prefix} RepConvN conv2 1x1")
		return self.shortcut(conv1, activation="swish", comment=f"{prefix} RepConvN conv1+conv2")

	def add_repnbottleneck(self, c2: int, *, prefix: str, shortcut: bool = True, groups: int = 1) -> int:
		input_idx = self.last_idx()
		if input_idx is None:
			raise GenerateError("RepNBottleneck cannot consume net input directly")
		hidden = c2
		self.add_repconvn(hidden, prefix=f"{prefix} cv1")
		self.conv(c2, size=3, groups=groups, comment=f"{prefix} cv2")
		if shortcut and self.section_shape(input_idx) == self.current_shape:
			return self.shortcut(input_idx, activation="linear", comment=f"{prefix} residual")
		return self.last_idx() if self.last_idx() is not None else input_idx

	def add_repncsp(self, c2: int, n: int, *, prefix: str, shortcut: bool = True, groups: int = 1) -> int:
		input_idx = self.last_idx()
		if input_idx is None:
			raise GenerateError("RepNCSP cannot consume net input directly")
		hidden = c2 // 2
		if hidden <= 0:
			raise GenerateError(f"{prefix} RepNCSP hidden channels must be positive")

		self.conv(hidden, size=1, comment=f"{prefix} RepNCSP cv1")
		for block_idx in range(n):
			self.add_repnbottleneck(hidden, prefix=f"{prefix} RepNCSP m.{block_idx}", shortcut=shortcut, groups=groups)
		path1 = self.last_idx()
		assert path1 is not None

		self.route([input_idx], comment=f"{prefix} RepNCSP cv2 input")
		path2 = self.conv(hidden, size=1, comment=f"{prefix} RepNCSP cv2")
		self.route([path1, path2], comment=f"{prefix} RepNCSP concat")
		return self.conv(c2, size=1, comment=f"{prefix} RepNCSP cv3")

	def add_repncspelan4_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) != 4:
			raise GenerateError(f"YAML {yaml_idx} RepNCSPELAN4 expects [c2, c3, c4, c5], got {args!r}")
		c2, c3, c4, c5 = (int(value) for value in args)
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} RepNCSPELAN4")
		cv1 = self.conv(c3, size=1, comment=f"YAML {yaml_idx} RepNCSPELAN4 cv1")
		chunk0 = self.route([cv1], groups=2, group_id=0, comment=f"YAML {yaml_idx} RepNCSPELAN4 chunk 0")
		chunk1 = self.route([cv1], groups=2, group_id=1, comment=f"YAML {yaml_idx} RepNCSPELAN4 chunk 1")
		self.add_repncsp(c4, c5, prefix=f"YAML {yaml_idx} RepNCSPELAN4 cv2")
		cv2_out = self.conv(c4, size=3, comment=f"YAML {yaml_idx} RepNCSPELAN4 cv2 conv")
		self.add_repncsp(c4, c5, prefix=f"YAML {yaml_idx} RepNCSPELAN4 cv3")
		cv3_out = self.conv(c4, size=3, comment=f"YAML {yaml_idx} RepNCSPELAN4 cv3 conv")
		self.route([chunk0, chunk1, cv2_out, cv3_out], comment=f"YAML {yaml_idx} RepNCSPELAN4 concat")
		return self.conv(c2, size=1, comment=f"YAML {yaml_idx} RepNCSPELAN4 cv4")

	def add_sppelan_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) != 2:
			raise GenerateError(f"YAML {yaml_idx} SPPELAN expects [c2, c3], got {args!r}")
		c2, c3 = (int(value) for value in args)
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} SPPELAN")
		cv1 = self.conv(c3, size=1, comment=f"YAML {yaml_idx} SPPELAN cv1")
		pool1 = self.maxpool(size=5, stride=1, comment=f"YAML {yaml_idx} SPPELAN maxpool 1")
		pool2 = self.maxpool(size=5, stride=1, comment=f"YAML {yaml_idx} SPPELAN maxpool 2")
		pool3 = self.maxpool(size=5, stride=1, comment=f"YAML {yaml_idx} SPPELAN maxpool 3")
		self.route([cv1, pool1, pool2, pool3], comment=f"YAML {yaml_idx} SPPELAN concat")
		return self.conv(c2, size=1, comment=f"YAML {yaml_idx} SPPELAN cv5")

	def add_concat_module(self, yaml_idx: int, sources: Sequence[int]) -> int:
		layers: list[int] = []
		for source in sources:
			resolved = self.resolve_source(source, yaml_idx)
			if resolved is None:
				raise GenerateError(f"YAML {yaml_idx} Concat cannot use net input directly")
			layers.append(resolved)
		return self.route(layers, comment=f"YAML {yaml_idx} Concat from {list(sources)}")

	def add_upsample_module(self, yaml_idx: int, source: int, args: Sequence[Any]) -> int:
		if len(args) < 2:
			raise GenerateError(f"YAML {yaml_idx} nn.Upsample expects scale args, got {args!r}")
		stride = int(args[1])
		self.ensure_source(source, yaml_idx, comment=f"route YAML {source} for YAML {yaml_idx} Upsample")
		return self.upsample(stride=stride, comment=f"YAML {yaml_idx} nn.Upsample x{stride}")

	def ddetect_hidden(self, first_channels: int) -> tuple[int, int]:
		box_hidden = make_divisible(max(first_channels // 4, BOX_LOGIT_CHANNELS, 16), 4)
		cls_hidden = max(first_channels, min(self.profile.classes * 2, 128))
		return box_hidden, cls_hidden

	def add_ddetect_head(
		self,
		*,
		yaml_idx: int,
		source_yaml: int,
		feature_idx: int,
		scale_name: str,
		box_hidden: int,
		cls_hidden: int,
	) -> int:
		feature_shape = self.section_shape(feature_idx)
		self.route([feature_idx], comment=f"YAML {yaml_idx} {scale_name} box branch input from YAML {source_yaml}")
		self.conv(box_hidden, size=3, comment=f"YAML {yaml_idx} {scale_name} box conv 1")
		self.conv(box_hidden, size=3, groups=4, comment=f"YAML {yaml_idx} {scale_name} box conv 2")
		box_logits = self.conv(
			BOX_LOGIT_CHANNELS,
			size=1,
			groups=4,
			batch_normalize=False,
			activation="linear",
			comment=f"YAML {yaml_idx} {scale_name} box logits",
		)

		self.route([feature_idx], comment=f"YAML {yaml_idx} {scale_name} class branch input from YAML {source_yaml}")
		self.conv(cls_hidden, size=3, comment=f"YAML {yaml_idx} {scale_name} class conv 1")
		self.conv(cls_hidden, size=3, comment=f"YAML {yaml_idx} {scale_name} class conv 2")
		class_logits = self.conv(
			self.profile.classes,
			size=1,
			batch_normalize=False,
			activation="linear",
			comment=f"YAML {yaml_idx} {scale_name} class logits",
		)

		out = self.route([box_logits, class_logits], comment=f"YAML {yaml_idx} {scale_name} logits concat")
		out_shape = self.section_shape(out)
		expected_channels = self.profile.classes + BOX_LOGIT_CHANNELS
		if out_shape.w != feature_shape.w or out_shape.h != feature_shape.h or out_shape.c != expected_channels:
			raise GenerateError(
				f"{scale_name} DDetect logits shape {out_shape.describe()} does not match "
				f"feature {feature_shape.describe()} + {expected_channels} output channels")
		return out

	def add_detect_module(self, yaml_idx: int, module: str, sources: Sequence[int]) -> int:
		source_layers: list[tuple[int, int, Shape]] = []
		for source in sources:
			resolved = self.resolve_source(source, yaml_idx)
			if resolved is None:
				raise GenerateError(f"YAML {yaml_idx} {module} cannot use net input directly")
			source_layers.append((source, resolved, self.section_shape(resolved)))

		if module == "DDetect":
			branch_count = 1
			inference_branch = 0
		elif module == "DualDDetect":
			branch_count = 2
			inference_branch = 1
		else:
			raise GenerateError(f"unsupported detect module {module!r}")
		if len(source_layers) % branch_count != 0:
			raise GenerateError(f"YAML {yaml_idx} {module} source count is not divisible by branch_count")

		scale_count = len(source_layers) // branch_count
		if scale_count != 3:
			raise GenerateError(f"YAML {yaml_idx} {module} expected three scales per branch, got {scale_count}")

		branch_hiddens: list[tuple[int, int]] = []
		for branch in range(branch_count):
			first_channels = source_layers[branch * scale_count][2].c
			branch_hiddens.append(self.ddetect_hidden(first_channels))

		detect_layers: list[int] = []
		scale_labels = ("3/8", "4/16", "5/32")
		for idx, (source_yaml, feature_idx, _shape) in enumerate(source_layers):
			branch = idx // scale_count
			scale = idx % scale_count
			prefix = "A" if branch_count == 2 and branch == 0 else "P"
			box_hidden, cls_hidden = branch_hiddens[branch]
			detect_layers.append(
				self.add_ddetect_head(
					yaml_idx=yaml_idx,
					source_yaml=source_yaml,
					feature_idx=feature_idx,
					scale_name=f"{prefix}{scale_labels[scale]}",
					box_hidden=box_hidden,
					cls_hidden=cls_hidden,
				)
			)

		self.validate_detect_strides(yaml_idx, source_layers, branch_count=branch_count, scale_count=scale_count)

		options: list[tuple[str, int | float | str]] = [
			("classes", self.profile.classes),
			("reg_max", REG_MAX),
			("layers", ",".join(str(layer) for layer in detect_layers)),
			("strides", "8,16,32"),
			("branch_count", branch_count),
			("inference_branch", inference_branch),
		]
		if branch_count > 1:
			options.append(("aux_loss_weight", 0.25))
		options.extend([
			("tal_topk", 10),
			("tal_alpha", 0.5),
			("tal_beta", 6.0),
			("box_normalizer", 7.5),
			("cls_normalizer", 0.5),
			("dfl_normalizer", 1.5),
			("nms_kind", "greedynms"),
			("beta_nms", 0.6),
		])
		if branch_count == 2:
			comment = (
				"YOLOv9 DualDDetect loss/output; layers are aux A3/A4/A5 first, "
				"main P3/P4/P5 second"
			)
		else:
			comment = "YOLOv9 DDetect loss/output"
		return self.add_section("yolov9", options, None, comment)

	def validate_detect_strides(
		self,
		yaml_idx: int,
		source_layers: Sequence[tuple[int, int, Shape]],
		*,
		branch_count: int,
		scale_count: int,
	) -> None:
		expected = (8, 16, 32)
		for branch in range(branch_count):
			for scale in range(scale_count):
				source_yaml, layer_idx, shape = source_layers[branch * scale_count + scale]
				stride = expected[scale]
				if self.profile.width // stride != shape.w or self.profile.height // stride != shape.h:
					raise GenerateError(
						f"YAML {yaml_idx} detect input YAML {source_yaml}/cfg {layer_idx} has "
						f"{shape.describe()}, expected stride {stride} from "
						f"{self.profile.width}x{self.profile.height}")

	def add_yaml_layer(self, yaml_idx: int, raw: Any) -> None:
		if not isinstance(raw, list) or len(raw) != 4:
			raise GenerateError(f"YAML layer {yaml_idx} must be [from, repeats, module, args], got {raw!r}")
		source, repeats, module, args = raw
		if repeats != 1:
			raise GenerateError(f"YAML layer {yaml_idx} uses repeats={repeats}; generator supports expanded n=1 layers")
		if not isinstance(module, str):
			raise GenerateError(f"YAML layer {yaml_idx} module must be a string, got {module!r}")
		if not isinstance(args, list):
			raise GenerateError(f"YAML layer {yaml_idx} args must be a list, got {args!r}")

		if module in {"Concat", "DDetect", "DualDDetect"}:
			sources = as_int_list(source, context=f"YAML {yaml_idx} {module} sources")
		elif isinstance(source, int):
			sources = [source]
		else:
			raise GenerateError(f"YAML {yaml_idx} {module} source must be an int, got {source!r}")

		if module == "Conv":
			out_idx = self.add_conv_module(yaml_idx, sources[0], args)
		elif module == "AConv":
			out_idx = self.add_aconv_module(yaml_idx, sources[0], args)
		elif module == "ELAN1":
			out_idx = self.add_elan1_module(yaml_idx, sources[0], args)
		elif module == "RepNCSPELAN4":
			out_idx = self.add_repncspelan4_module(yaml_idx, sources[0], args)
		elif module == "SPPELAN":
			out_idx = self.add_sppelan_module(yaml_idx, sources[0], args)
		elif module == "nn.Upsample":
			out_idx = self.add_upsample_module(yaml_idx, sources[0], args)
		elif module == "Concat":
			out_idx = self.add_concat_module(yaml_idx, sources)
		elif module in {"DDetect", "DualDDetect"}:
			out_idx = self.add_detect_module(yaml_idx, module, sources)
		else:
			raise GenerateError(f"YAML layer {yaml_idx} uses unsupported module {module!r}")

		self.yaml_outputs[yaml_idx] = out_idx

	def net_block(self) -> list[str]:
		p = self.profile
		return [
			"[net]",
			"# Testing",
			"#batch=1",
			"#subdivisions=1",
			"# Training",
			f"batch={p.batch}",
			f"subdivisions={p.subdivisions}",
			f"width={p.width}",
			f"height={p.height}",
			"channels=3",
			"momentum=0.9",
			"decay=0.0005",
			"angle=0",
			"saturation = 1.5",
			"exposure = 1.5",
			"hue=.1",
			"",
			"learning_rate=0.00261",
			"burn_in=1000",
			f"max_batches={p.max_batches}",
			"policy=steps",
			f"steps={p.steps[0]},{p.steps[1]}",
			"scales=.1,.1",
			"",
			"cutmix=0",
			"flip=0",
			"mixup=0",
			"mosaic=0",
			"use_cuda_graph=0",
		]

	def render(self, output_path: Path) -> str:
		source = self.source_yaml.relative_to(ROOT).as_posix()
		generator = Path(__file__).resolve().relative_to(ROOT).as_posix()
		lines = [
			f"# Generated Darknet cfg for {self.model} ({self.profile.name}).",
			f"# Generator: {generator}",
			f"# Source topology: {source}",
		]
		if self.profile.dataset_comment:
			lines.append(f"# Dataset/defaults: {self.profile.dataset_comment}")
		if self.profile.training_gate:
			cfg_path = output_path.relative_to(ROOT).as_posix()
			lines.append(f"# Training gate: {self.profile.training_gate.format(cfg_path=cfg_path)}")
		lines.extend([
			"#",
			"# Mapping notes:",
			"# - Conv, AConv, SPPELAN, routes, upsample, maxpool, and final [yolov9] use native Darknet layers.",
			"# - RepNCSPELAN4 expands RepNCSP/RepNBottleneck/RepConvN train-time branches with route+shortcut sums.",
			"# - DDetect/DualDDetect heads are expanded into box/class conv branches before the final [yolov9] layer.",
		])
		if self.model.startswith("yolov9"):
			lines.append("# - DualDDetect order is aux A3/A4/A5 first, main P3/P4/P5 second; inference_branch=1.")
		lines.append("")
		lines.extend(self.net_block())
		for idx, section in enumerate(self.sections):
			lines.append("")
			lines.extend(section.render(idx))
		lines.append("")
		return "\n".join(lines)


def build_cfg(model: str, profile: Profile, reference_dir: Path, output_path: Path) -> str:
	source_yaml = reference_dir / f"{model}.yaml"
	data = load_reference_yaml(source_yaml)
	builder = CfgBuilder(model=model, profile=profile, source_yaml=source_yaml)
	raw_layers = list(data["backbone"]) + list(data["head"])
	for yaml_idx, raw in enumerate(raw_layers):
		builder.add_yaml_layer(yaml_idx, raw)
	return builder.render(output_path)


def write_if_changed(path: Path, content: str) -> bool:
	if path.exists() and path.read_text(encoding="utf-8") == content:
		return False
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")
	return True


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--models",
		nargs="+",
		choices=DEFAULT_MODELS,
		default=list(DEFAULT_MODELS),
		help="model cfgs to generate",
	)
	parser.add_argument(
		"--profiles",
		nargs="+",
		choices=tuple(PROFILES),
		default=list(PROFILES),
		help="profile variants to generate",
	)
	parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR, help="directory containing reference YAML files")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="cfg output directory")
	parser.add_argument("--check-only", action="store_true", help="generate and validate in memory without writing cfg files")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)

	try:
		for model in args.models:
			for profile_name in args.profiles:
				profile = PROFILES[profile_name]
				output_path = args.output_dir / f"{model}{profile.suffix}.cfg"
				content = build_cfg(model, profile, args.reference_dir, output_path)
				if args.check_only:
					print(f"checked {output_path.relative_to(ROOT)}")
					continue
				changed = write_if_changed(output_path, content)
				verb = "wrote" if changed else "unchanged"
				print(f"{verb} {output_path.relative_to(ROOT)}")
	except GenerateError as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
