import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))

import utils

nb_path = repo_root / "model_training.ipynb"
df = utils.load_metrics_from_notebook(str(nb_path))
if df is None:
    print("No metrics table found in model_training.ipynb (extraction helper returned None).")
    sys.exit(1)

out = df.to_dict(orient="records")
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)
with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Wrote models/metrics.json with", len(out), "rows.")
