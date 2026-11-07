"""Small data-flow checks for frozen-feature nested CV; no GPU/download needed."""
from pathlib import Path
import importlib.util
import tempfile

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


fragment = Path(__file__).with_name("dino_branch_fragment.py")
spec = importlib.util.spec_from_file_location("dino_fragment", fragment)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.np = np
mod.Path = Path

rng = np.random.default_rng(9)
y = np.repeat(np.arange(5), 24)
groups = np.repeat(np.arange(60), 2)
features = np.eye(5)[y] * 3 + rng.normal(0, .15, (len(y), 5))
test_features = np.eye(5) * 3
splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=42)
              .split(features, y, groups))

for tr, va in splits:
    assert not np.intersect1d(groups[tr], groups[va]).size
    inner, reason = mod._dino_inner_splits(y[tr], groups[tr], 3, 44, 5)
    assert reason is None and len(inner) >= 2
    for itr, iva in inner:
        assert not np.intersect1d(groups[tr[itr]], groups[tr[iva]]).size
        assert not np.intersect1d(tr[itr], va).size
        assert not np.intersect1d(tr[iva], va).size

with tempfile.TemporaryDirectory(dir=fragment.parent, prefix="dino_test_") as temp:
    oof, probs, report = mod._dino_fit_classifiers(
        features, y, test_features, splits, list("abcde"), groups=groups,
        c_values=(.01, .1, 1., 10.), model_dir=Path(temp))
    assert oof.shape == (120, 5) and probs.shape == (5, 5)
    np.testing.assert_allclose(oof.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(oof.argmax(axis=1), y)
    assert np.array_equal(probs.argmax(axis=1), np.arange(5))
    assert len(report["folds"]) == 3
    for fold, (tr, va) in enumerate(splits, 1):
        artifact = joblib.load(Path(temp) / f"dinov2_vits14_linear_fold{fold}.joblib")
        scaler = artifact["pipeline"].named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, features[tr].mean(axis=0), atol=1e-10)
        restored = mod._dino_aligned_proba(artifact["pipeline"], features[va], 5)
        np.testing.assert_allclose(restored, oof[va], atol=1e-7)

# Too few independent groups skips tuning without splitting group members.
inner, reason = mod._dino_inner_splits(np.arange(5), np.arange(5), 3, 42, 5)
assert not inner and reason

# The classifier's column order is not assumed to be the dataset order.
class ReorderedClassifier:
    classes_ = np.array([2, 0, 1])

    def predict_proba(self, x):
        return np.tile([.6, .3, .1], (len(x), 1))

aligned = mod._dino_aligned_proba(ReorderedClassifier(), np.zeros((2, 3)), 3)
np.testing.assert_allclose(aligned, [[.3, .1, .6], [.3, .1, .6]])
print("PASS: grouped inner CV, OOF/test probabilities, outer-train scaler, joblib, class alignment")
