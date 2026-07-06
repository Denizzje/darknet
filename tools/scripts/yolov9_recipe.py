#!/usr/bin/env python3
"""Reference-derived YOLOv9/GELAN training recipes for Darknet cfg generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

try:
	import yaml
except ImportError:  # pragma: no cover - handled by the caller as well
	yaml = None


class ProfileLike(Protocol):
	name: str
	classes: int
	width: int
	height: int
	batch: int
	subdivisions: int
	burn_in: int
	max_batches: int
	learning_rate: str
	momentum: str
	decay: str
	policy: str
	steps: tuple[int, int] | None
	scales: str | None
	power: int | None
	flip: int
	mosaic: int
	hue: str
	saturation: str
	exposure: str
	max_chart_loss: int | None
	train_images: int | None
	epochs: int | None
	close_mosaic_epochs: int | None


@dataclass(frozen=True)
class Recipe:
	name: str
	output_suffix: str
	description: str
	sources: tuple[str, ...]


@dataclass(frozen=True)
class CompiledRecipe:
	name: str
	output_suffix: str
	description: str
	sources: tuple[str, ...]
	net_options: list[tuple[str, int | float | str]]
	manifest_training: dict[str, dict[str, Any]]
	manifest_augmentation: dict[str, dict[str, Any]]
	manifest_loss: dict[str, dict[str, Any]]
	updates_per_epoch: int | None = None
	epochs: int | None = None
	max_batches: int | None = None
	warmup_iterations: int | None = None
	close_mosaic_iteration: int | None = None


RECIPES: Mapping[str, Recipe] = {
	"paper_500e_close15": Recipe(
		name="paper_500e_close15",
		output_suffix="-yolov9-paper",
		description="YOLOv9 paper Appendix A / hyp.scratch-high recipe, 500 epochs with close-mosaic.",
		sources=(
			"reference_yolov9_repo/yolov9/data/hyps/hyp.scratch-high.yaml",
			"reference_yolov9_repo/yolov9/train.py",
			"reference_yolov9_repo/yolov9/train_dual.py",
			"YOLOv9 paper Appendix A",
		),
	),
	"repo_hyp_scratch_high": Recipe(
		name="repo_hyp_scratch_high",
		output_suffix="-yolov9-reference-scratch",
		description="YOLOv9 reference hyp.scratch-high recipe without paper close-mosaic override.",
		sources=(
			"reference_yolov9_repo/yolov9/data/hyps/hyp.scratch-high.yaml",
			"reference_yolov9_repo/yolov9/train.py",
			"reference_yolov9_repo/yolov9/train_dual.py",
		),
	),
	"legacy_darknet_comparison": Recipe(
		name="legacy_darknet_comparison",
		output_suffix="-darknet-legacy-comparison",
		description="Legacy Darknet comparison recipe preserving YOLOv4-style scheduler and augmentation fields.",
		sources=("tools/scripts/yolov9_generate_cfg.py historical profile defaults",),
	),
}


def recipe_names() -> tuple[str, ...]:
	return tuple(RECIPES)


def load_hyp_scratch_high(root: Path) -> dict[str, Any]:
	path = root / "reference_yolov9_repo" / "yolov9" / "data" / "hyps" / "hyp.scratch-high.yaml"
	if yaml is None:
		raise RuntimeError("PyYAML is required to load YOLOv9 recipe files")
	with path.open("r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle)
	if not isinstance(data, dict):
		raise RuntimeError(f"{path} did not load as a YAML mapping")
	return data


def _field(value: Any, source: str, status: str) -> dict[str, Any]:
	return {"value": value, "source": source, "status": status}


def _fmt_float(value: float) -> str:
	text = f"{value:.10g}"
	if "e" not in text and "." not in text:
		text += ".0"
	return text


def _updates_per_epoch(profile: ProfileLike) -> int | None:
	if profile.train_images is None:
		return None
	images_per_update = max(1, int(profile.batch))
	return int(math.ceil(profile.train_images / images_per_update))


def _compile_reference_recipe(recipe: Recipe, profile: ProfileLike, root: Path) -> CompiledRecipe:
	hyp = load_hyp_scratch_high(root)
	epochs = 500 if recipe.name == "paper_500e_close15" else (profile.epochs or 300)
	updates = _updates_per_epoch(profile)
	max_batches = profile.max_batches if updates is None else updates * epochs
	warmup_iterations = profile.burn_in
	if updates is not None:
		warmup_iterations = max(round(float(hyp["warmup_epochs"]) * updates), 100)

	close_mosaic_epochs = 15 if recipe.name == "paper_500e_close15" else (profile.close_mosaic_epochs or 0)
	close_mosaic_iteration = 0
	if updates is not None and close_mosaic_epochs:
		close_mosaic_iteration = max(0, max_batches - close_mosaic_epochs * updates)

	lr0 = float(hyp["lr0"])
	lrf = float(hyp["lrf"])
	final_lr = lr0 * lrf

	training = {
		"optimizer": _field("SGD", "YOLOv9 train.py smart_optimizer default and paper Appendix A", "exact"),
		"lr0": _field(lr0, "hyp.scratch-high.yaml lr0", "exact"),
		"lrf": _field(lrf, "hyp.scratch-high.yaml lrf", "exact"),
		"final_learning_rate": _field(final_lr, "lr0 * lrf", "exact"),
		"lr_policy": _field("linear_final", "train.py linear lf function", "exact"),
		"momentum": _field(float(hyp["momentum"]), "hyp.scratch-high.yaml momentum", "exact"),
		"weight_decay": _field(float(hyp["weight_decay"]), "hyp.scratch-high.yaml weight_decay", "exact"),
		"warmup_epochs": _field(float(hyp["warmup_epochs"]), "hyp.scratch-high.yaml warmup_epochs", "exact"),
		"warmup_iterations": _field(warmup_iterations, "warmup_epochs * updates_per_epoch", "exact_if_train_image_count_matches"),
		"warmup_momentum": _field(float(hyp["warmup_momentum"]), "hyp.scratch-high.yaml warmup_momentum", "exact"),
		"warmup_bias_lr": _field(float(hyp["warmup_bias_lr"]), "hyp.scratch-high.yaml warmup_bias_lr", "exact"),
		"warmup_nonbias_lr_start": _field(0.0, "train.py warmup interpolation for non-bias groups", "exact"),
		"epochs": _field(epochs, "YOLOv9 paper Appendix A" if recipe.name == "paper_500e_close15" else "generator profile/default", "exact"),
		"updates_per_epoch": _field(updates, "ceil(train_images / batch)", "exact_if_train_image_count_matches"),
		"max_batches": _field(max_batches, "updates_per_epoch * epochs", "exact_if_train_image_count_matches"),
		"close_mosaic_epochs": _field(close_mosaic_epochs, "YOLOv9 paper Appendix A", "exact" if close_mosaic_epochs else "not_enabled"),
		"close_mosaic_iteration": _field(close_mosaic_iteration, "max_batches - close_mosaic_epochs * updates_per_epoch", "exact_if_train_image_count_matches"),
	}

	loss = {
		"reg_max": _field(16, "models/yolo.py DDetect/DualDDetect", "exact"),
		"tal_topk": _field(10, "utils/loss_tal*.py", "exact"),
		"tal_alpha": _field(0.5, "utils/loss_tal*.py", "exact"),
		"tal_beta": _field(6.0, "utils/loss_tal*.py", "exact"),
		"box_gain": _field(float(hyp["box"]), "hyp.scratch-high.yaml box", "exact"),
		"cls_gain": _field(float(hyp["cls"]), "hyp.scratch-high.yaml cls", "exact"),
		"dfl_gain": _field(float(hyp["dfl"]), "hyp.scratch-high.yaml dfl", "exact"),
	}

	augmentation = {
		"augment_policy": _field("yolov9", "YOLOv9 reference dataloader", "implemented"),
		"hsv_h": _field(float(hyp["hsv_h"]), "hyp.scratch-high.yaml hsv_h", "implemented"),
		"hsv_s": _field(float(hyp["hsv_s"]), "hyp.scratch-high.yaml hsv_s", "implemented"),
		"hsv_v": _field(float(hyp["hsv_v"]), "hyp.scratch-high.yaml hsv_v", "implemented"),
		"degrees": _field(float(hyp["degrees"]), "hyp.scratch-high.yaml degrees", "implemented"),
		"translate": _field(float(hyp["translate"]), "hyp.scratch-high.yaml translate", "implemented"),
		"yolov9_scale": _field(float(hyp["scale"]), "hyp.scratch-high.yaml scale", "implemented"),
		"shear": _field(float(hyp["shear"]), "hyp.scratch-high.yaml shear", "implemented"),
		"perspective": _field(float(hyp["perspective"]), "hyp.scratch-high.yaml perspective", "implemented"),
		"flipud_prob": _field(float(hyp["flipud"]), "hyp.scratch-high.yaml flipud", "implemented"),
		"fliplr_prob": _field(float(hyp["fliplr"]), "hyp.scratch-high.yaml fliplr", "implemented"),
		"mosaic_prob": _field(float(hyp["mosaic"]), "hyp.scratch-high.yaml mosaic", "implemented"),
		"mixup_prob": _field(float(hyp["mixup"]), "hyp.scratch-high.yaml mixup", "implemented"),
		"copy_paste_prob": _field(float(hyp["copy_paste"]), "hyp.scratch-high.yaml copy_paste", "implemented"),
	}

	net_options: list[tuple[str, int | float | str]] = [
		("momentum", _fmt_float(float(hyp["momentum"]))),
		("decay", _fmt_float(float(hyp["weight_decay"]))),
		("augment_policy", "yolov9"),
		("hsv_h", _fmt_float(float(hyp["hsv_h"]))),
		("hsv_s", _fmt_float(float(hyp["hsv_s"]))),
		("hsv_v", _fmt_float(float(hyp["hsv_v"]))),
		("degrees", _fmt_float(float(hyp["degrees"]))),
		("translate", _fmt_float(float(hyp["translate"]))),
		("yolov9_scale", _fmt_float(float(hyp["scale"]))),
		("shear", _fmt_float(float(hyp["shear"]))),
		("perspective", _fmt_float(float(hyp["perspective"]))),
		("flipud_prob", _fmt_float(float(hyp["flipud"]))),
		("fliplr_prob", _fmt_float(float(hyp["fliplr"]))),
		("mosaic_prob", _fmt_float(float(hyp["mosaic"]))),
		("mixup_prob", _fmt_float(float(hyp["mixup"]))),
		("copy_paste_prob", _fmt_float(float(hyp["copy_paste"]))),
		("learning_rate", _fmt_float(lr0)),
		("lrf", _fmt_float(lrf)),
		("final_learning_rate", _fmt_float(final_lr)),
		("burn_in", warmup_iterations),
		("warmup_iterations", warmup_iterations),
		("warmup_bias_lr", _fmt_float(float(hyp["warmup_bias_lr"]))),
		("warmup_momentum", _fmt_float(float(hyp["warmup_momentum"]))),
		("warmup_nonbias_lr_start", _fmt_float(0.0)),
		("max_batches", max_batches),
		("policy", "linear_final"),
	]
	if close_mosaic_epochs:
		net_options.extend([
			("close_mosaic_epochs", close_mosaic_epochs),
			("close_mosaic_iteration", close_mosaic_iteration),
		])
	net_options.extend([
		("cutmix", 0),
		("use_cuda_graph", 0),
	])

	return CompiledRecipe(
		name=recipe.name,
		output_suffix=recipe.output_suffix,
		description=recipe.description,
		sources=recipe.sources,
		net_options=net_options,
		manifest_training=training,
		manifest_augmentation=augmentation,
		manifest_loss=loss,
		updates_per_epoch=updates,
		epochs=epochs,
		max_batches=max_batches,
		warmup_iterations=warmup_iterations,
		close_mosaic_iteration=close_mosaic_iteration,
	)


def _compile_legacy_recipe(recipe: Recipe, profile: ProfileLike) -> CompiledRecipe:
	training = {
		"lr0": _field(profile.learning_rate, "historical Darknet profile default", "legacy_comparison"),
		"lr_policy": _field(profile.policy, "historical Darknet profile default", "legacy_comparison"),
		"momentum": _field(profile.momentum, "historical Darknet profile default", "legacy_comparison"),
		"weight_decay": _field(profile.decay, "historical Darknet profile default", "legacy_comparison"),
		"burn_in": _field(profile.burn_in, "historical Darknet profile default", "legacy_comparison"),
		"max_batches": _field(profile.max_batches, "historical Darknet profile default", "legacy_comparison"),
	}
	augmentation = {
		"hue": _field(profile.hue, "historical Darknet profile default", "legacy_comparison"),
		"saturation": _field(profile.saturation, "historical Darknet profile default", "legacy_comparison"),
		"exposure": _field(profile.exposure, "historical Darknet profile default", "legacy_comparison"),
		"flip": _field(profile.flip, "historical Darknet profile default", "legacy_comparison"),
		"mosaic": _field(profile.mosaic, "historical Darknet profile default", "legacy_comparison"),
	}
	loss = {
		"reg_max": _field(16, "models/yolo.py DDetect/DualDDetect", "exact"),
		"tal_topk": _field(10, "utils/loss_tal*.py", "exact"),
		"tal_alpha": _field(0.5, "utils/loss_tal*.py", "exact"),
		"tal_beta": _field(6.0, "utils/loss_tal*.py", "exact"),
		"box_gain": _field(7.5, "YOLOv9 reference loss gain", "exact"),
		"cls_gain": _field(0.5, "YOLOv9 reference loss gain", "exact"),
		"dfl_gain": _field(1.5, "YOLOv9 reference loss gain", "exact"),
	}

	net_options: list[tuple[str, int | float | str]] = [
		("momentum", profile.momentum),
		("decay", profile.decay),
		("angle", 0),
		("saturation", profile.saturation),
		("exposure", profile.exposure),
		("hue", profile.hue),
		("learning_rate", profile.learning_rate),
		("burn_in", profile.burn_in),
		("max_batches", profile.max_batches),
		("policy", profile.policy),
	]
	if profile.steps is not None:
		net_options.append(("steps", f"{profile.steps[0]},{profile.steps[1]}"))
	if profile.scales is not None:
		net_options.append(("scales", profile.scales))
	if profile.power is not None:
		net_options.append(("power", profile.power))
	net_options.extend([
		("cutmix", 0),
		("flip", profile.flip),
		("mixup", 0),
		("mosaic", profile.mosaic),
		("use_cuda_graph", 0),
	])

	return CompiledRecipe(
		name=recipe.name,
		output_suffix=recipe.output_suffix,
		description=recipe.description,
		sources=recipe.sources,
		net_options=net_options,
		manifest_training=training,
		manifest_augmentation=augmentation,
		manifest_loss=loss,
		max_batches=profile.max_batches,
		warmup_iterations=profile.burn_in,
	)


def compile_recipe(name: str, profile: ProfileLike, root: Path) -> CompiledRecipe:
	if name not in RECIPES:
		raise KeyError(f"unknown recipe {name!r}")
	recipe = RECIPES[name]
	if name == "legacy_darknet_comparison":
		return _compile_legacy_recipe(recipe, profile)
	return _compile_reference_recipe(recipe, profile, root)


def dump_manifest(manifest: Mapping[str, Any]) -> str:
	if yaml is None:
		raise RuntimeError("PyYAML is required to write YOLOv9 recipe manifests")
	return yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=False)
