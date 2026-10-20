"""Write numerical results as standard JSON, using null for undefined values."""
import json
import math
from pathlib import Path
import numpy as np

def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
