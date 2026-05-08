#!/usr/bin/env python3
"""Compare YOLOv9 PyTorch and Darknet parity fixture ``.npz`` files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def comparable_keys(left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile, pattern: str | None) -> list[str]:
	keys = sorted(set(left.files) & set(right.files))
	keys = [key for key in keys if key != "meta"]
	if pattern:
		keys = [key for key in keys if pattern in key]
	return keys


def compare_arrays(left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
	if left.shape != right.shape:
		return {"ok": False, "shape_left": left.shape, "shape_right": right.shape, "error": "shape mismatch"}
	diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
	max_abs = float(diff.max()) if diff.size else 0.0
	denom = np.maximum(np.abs(left.astype(np.float64)), atol)
	max_rel = float((diff / denom).max()) if diff.size else 0.0
	ok = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False))
	return {
		"ok": ok,
		"shape": left.shape,
		"max_abs": max_abs,
		"max_rel": max_rel,
		"mean_abs": float(diff.mean()) if diff.size else 0.0,
	}


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--pytorch", type=Path, required=True, help="PyTorch fixture .npz")
	parser.add_argument("--darknet", type=Path, required=True, help="Darknet fixture .npz")
	parser.add_argument("--rtol", type=float, default=1.0e-4)
	parser.add_argument("--atol", type=float, default=1.0e-5)
	parser.add_argument("--key-contains", help="compare only keys containing this substring")
	parser.add_argument("--allow-missing", action="store_true", help="do not fail when either fixture has extra keys")
	parser.add_argument("--report", type=Path, help="write JSON comparison report")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	if args.rtol < 0.0 or args.atol < 0.0:
		parser.error("rtol and atol must be non-negative")

	left = np.load(args.pytorch, allow_pickle=False)
	right = np.load(args.darknet, allow_pickle=False)
	keys = comparable_keys(left, right, args.key_contains)
	missing_left = sorted(set(right.files) - set(left.files))
	missing_right = sorted(set(left.files) - set(right.files))

	results: dict[str, Any] = {
		"ok": True,
		"pytorch": str(args.pytorch),
		"darknet": str(args.darknet),
		"rtol": args.rtol,
		"atol": args.atol,
		"compared_keys": keys,
		"missing_from_pytorch": missing_left,
		"missing_from_darknet": missing_right,
		"arrays": {},
	}

	if (missing_left or missing_right) and not args.allow_missing:
		results["ok"] = False

	for key in keys:
		comparison = compare_arrays(left[key], right[key], args.rtol, args.atol)
		results["arrays"][key] = comparison
		if not comparison["ok"]:
			results["ok"] = False

	if args.report:
		write_json(args.report, results)

	if not results["ok"]:
		print("parity comparison failed", file=sys.stderr)
		for key, comparison in results["arrays"].items():
			if not comparison["ok"]:
				print(f"  {key}: {comparison}", file=sys.stderr)
		if missing_left:
			print(f"  missing from PyTorch fixture: {len(missing_left)}", file=sys.stderr)
		if missing_right:
			print(f"  missing from Darknet fixture: {len(missing_right)}", file=sys.stderr)
		return 1

	print(f"compared {len(keys)} tensors successfully")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
