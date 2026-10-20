"""One-off: convert MSCN artifacts written before format v3 to weights-only form.

    python -m scripts.migrate_models

Only for artifacts this project produced itself (they are unpickled once, here).
Fusion models are not migrated: their feature format changed and they are retrained.
"""
import glob

import torch

from src.estimator.training import FORMAT_VERSION, save

for path in glob.glob("models/**/mscn_s*.pt", recursive=True):
    b = torch.load(path, weights_only=False)
    if b.get("version") == FORMAT_VERSION:
        continue
    b["version"] = FORMAT_VERSION
    b["dims"] = list(b["dims"])
    save(b, path)
    print("migrated", path)
