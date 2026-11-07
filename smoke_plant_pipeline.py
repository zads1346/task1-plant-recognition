"""Synthetic CPU smoke test. Tiny stand-ins test plumbing, not model accuracy."""
from pathlib import Path
import sys
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_smoke_deps"))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

spec = importlib.util.spec_from_file_location("plant_study", ROOT / "plant_study.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)


class TinyCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(8, classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class TinyFrozenFeatures(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean((-2, -1)), x.std((-2, -1)),
                          x[:, :, :x.shape[-2] // 2, :].mean((-2, -1))], dim=1)


def main():
    root = ROOT / "_smoke_output"
    train, test = root / "data" / "train", root / "data" / "test"
    test.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    for label in range(5):
        directory = train / f"class{label}"
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            image = rng.integers(0, 150, (48 + 2 * i, 72 + label, 3), dtype=np.uint8)
            image[..., label % 3] += 60
            Image.fromarray(image).save(directory / f"{i}.png")
    ids = ["z.png", "a.png", "m.png"]
    for name in ids:
        Image.fromarray(rng.integers(0, 255, (51, 69, 3), dtype=np.uint8)).save(test / name)
    template = root / "template.csv"
    pd.DataFrame({"ID": ids, "Category": ["", "", ""]}).to_csv(template, index=False)
    output = root / "result"
    original_build, original_hub, original_argv = module.build_model, torch.hub.load, sys.argv
    module.build_model = lambda arch, n, pretrained, device: TinyCNN(n).to(device)
    torch.hub.load = lambda *args, **kwargs: TinyFrozenFeatures()
    sys.argv = ["plant_study.py", "--train", str(train), "--test", str(test),
                "--template", str(template), "--out", str(output),
                "--architectures", "efficientnet_b2", "--folds", "2", "--epochs", "1",
                "--freeze-epochs", "0", "--size", "96", "--progressive-size", "112",
                "--progressive-epochs", "1", "--dino-size", "112", "--workers", "0"]
    try:
        module.main()
    finally:
        module.build_model, torch.hub.load, sys.argv = original_build, original_hub, original_argv
    submission = pd.read_csv(output / "submission.csv")
    assert submission.columns.tolist() == ["ID", "Category"]
    assert submission.ID.tolist() == ids
    assert set(submission.Category).issubset({f"class{i}" for i in range(5)})
    probabilities = np.load(output / "final_test_probabilities.npy")
    assert probabilities.shape == (3, 5) and np.allclose(probabilities.sum(1), 1.)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["families"] == ["efficientnet_b2", "dinov2"]
    assert len(list((output / "models").glob("*.pth"))) == 4
    assert len(list((output / "models").glob("*.joblib"))) == 2
    assert {"base", "progressive"} == set(pd.read_csv(output / "fold_training.csv").stage)
    assert not (output / "submission.csv.tmp").exists()
    print("PASS: synthetic DINO heads + CNN base/progressive + ensemble + exact submission schema/order")


if __name__ == "__main__":
    main()
