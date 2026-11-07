"""Small dependency-light regression checks; does not start model training.

Extracts pure helpers through AST so this can also run where Torch/sklearn are
unavailable. Real model smoke tests are a separate check.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps


SOURCE = Path(__file__).with_name("plant_study.py")


def macro_f1(y, pred, average="macro", zero_division=0):
    del average, zero_division
    scores = []
    for label in np.unique(np.concatenate([y, pred])):
        tp = np.sum((y == label) & (pred == label))
        fp = np.sum((y != label) & (pred == label))
        fn = np.sum((y == label) & (pred != label))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores))


def extract_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    ns = dict(__doc__="test", np=np, Image=Image, ImageOps=ImageOps,
              Path=Path, argparse=argparse, sys=sys, itertools=itertools,
              math=math, f1_score=macro_f1)
    functions = {"args_parse", "weight_grid", "best_weights", "model_score",
                 "select_recipe", "views", "LetterboxResize"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = value
    # ARCHES may be built by concatenating a literal tuple to CNN_ARCHES.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name not in ns for name in names) and all(name.isupper() for name in names):
                try:
                    exec(compile(ast.Module([node], type_ignores=[]), str(SOURCE), "exec"), ns)
                except NameError:
                    pass
    exec(compile(ast.Module(selected, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


class TensorStub(np.ndarray):
    def flip(self, dims):
        return np.flip(self, axis=dims).copy().view(TensorStub)

    def rot90(self, k, dims):
        return np.rot90(self, k, axes=dims).copy().view(TensorStub)

    def contiguous(self):
        return self.copy().view(TensorStub)


def main():
    ns = extract_helpers()
    for n in range(1, 5):
        grid = np.asarray(list(ns["weight_grid"](n)))
        assert grid.ndim == 2 and grid.shape[1] == n, (n, grid.shape)
        assert np.all(np.isfinite(grid)) and np.all(grid >= -1e-8)
        assert np.allclose(grid.sum(axis=1), 1)
        assert len(np.unique(grid.round(8), axis=0)) == len(grid)
        for onehot in np.eye(n):
            assert np.any(np.all(np.isclose(grid, onehot), axis=1))
    print("PASS weights: 1-4 branches, simplex, unique candidates, one-hot choices")

    y = np.tile(np.arange(5), 6)
    perfect = np.eye(5, dtype=np.float32)[y] * .95 + .01
    wrong = np.roll(perfect, 1, axis=1)
    uniform = np.full_like(perfect, .2)
    probs = [perfect, wrong, uniform, wrong[:, ::-1]]
    for metric in ("accuracy", "macro_f1"):
        weights, score = ns["best_weights"](probs, y, metric)
        assert len(weights) == 4 and np.isclose(score, 1), (weights, score)
    arches = ("efficientnet_b2", "convnext_tiny", "resnet50", "dinov2")
    predictions = {f"base|{arch}|plain": p for arch, p in zip(arches, probs)}
    folds = [np.arange(start, start + 10) for start in range(0, len(y), 10)]
    recipe, predicted = ns["select_recipe"](predictions, y, folds, arches, "accuracy")
    assert predicted.shape == perfect.shape and np.all(np.isfinite(predicted))
    assert np.allclose(predicted.sum(1), 1) and len(recipe["weights"]) == 4
    assert np.array_equal(predicted.argmax(1), y)
    print("PASS synthetic 4-branch recipe selection and normalized OOF output")

    x = np.arange(18).reshape(2, 1, 3, 3).view(TensorStub)
    assert len(ns["views"](x, False)) == 1
    for count in (4, 8):
        augmented = ns["views"](x, True, count)
        assert len(augmented) == count
        assert len({v.tobytes() for v in augmented}) == count
        assert all(v.shape == x.shape for v in augmented)
    print("PASS TTA: 4/8 distinct shape-preserving views")

    for size in (224, 288, 384):
        for source_size in ((120, 60), (60, 120), (80, 80), (1, 200), (200, 1)):
            result = ns["LetterboxResize"](size)(Image.new("RGB", source_size, (240, 10, 20)))
            assert isinstance(result, Image.Image) and result.size == (size, size)
            # The central source-colored region retains the source aspect ratio.
            pixels = np.asarray(result)
            rows, cols = np.where((pixels[..., 0] > 220) & (pixels[..., 1] < 30))
            assert len(rows)
            actual = (cols.max() - cols.min() + 1, rows.max() - rows.min() + 1)
            scale = size / max(source_size)
            expected = tuple(max(1, round(d * scale)) for d in source_size)
            assert all(abs(a - b) <= 1 for a, b in zip(actual, expected)), (actual, expected)
    print("PASS letterbox: size and aspect preservation for wide/tall/square/extreme images")

    original_argv = sys.argv
    try:
        sys.argv = ["colab_kernel_launcher.py", "--HistoryManager.hist_file=:memory:"]
        a = ns["args_parse"]()
        assert a.out == Path("/kaggle/working/plant_option_c_v3_optimized")
        explicit = ns["args_parse"](["--epochs", "4", "--folds", "3"])
        assert explicit.epochs == 4 and explicit.folds == 3
        ns["NOTEBOOK_CONFIG"] = {"epochs": 7}
        override = ns["args_parse"]()
        assert override.epochs == 7
    finally:
        sys.argv = original_argv
    print("PASS Colab flags ignored, explicit arguments accepted, notebook override honored, output path unchanged")


if __name__ == "__main__":
    main()
