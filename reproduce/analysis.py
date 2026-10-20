"""Per-experiment summaries and explicitly paired, unit-level comparisons."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .cli import save_json, digest, source_signature


def paired_sign_flip(differences, seed=42, repetitions=10000):
    """Two-sided test of a mean paired difference under sign exchangeability."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Comparisons require at least two independent, finite unit differences")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    if len(d) <= 16:
        statistics = (abs(np.dot(signs, d) / len(d))
                      for signs in itertools.product([-1, 1], repeat=len(d)))
        extreme = sum(s >= observed - 1e-15 for s in statistics)
        p = extreme / 2**len(d)
        method = "exact paired sign-flip"
    else:
        extreme = 0
        for _ in range(repetitions):
            extreme += abs(np.mean(d * rng.choice([-1, 1], len(d)))) >= observed - 1e-15
        p = (extreme + 1) / (repetitions + 1)
        method = "Monte Carlo paired sign-flip"
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(repetitions)]
    return {"units": len(d), "mean_difference_a_minus_b": float(d.mean()),
            "bootstrap_95_percent_interval": np.quantile(means, [.025, .975]).tolist(),
            "p_value": p, "method": method, "seed": seed, "repetitions": repetitions}


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def summarize(config):
    root = Path(config["output_root"])
    results = {}
    rows = []
    for e in config["experiments"]:
        path = root / e["id"]
        if not (path / "complete.json").is_file():
            raise ValueError(f"Incomplete experiment: {e['id']}")
        manifest = json.loads((path / "manifest.json").read_text())
        settings = {k: v for k, v in config.items() if k not in {"experiments", "output_root"}}
        if manifest["settings"] != settings or manifest["source_sha256"] != source_signature():
            raise ValueError(f"Settings or source changed: {e['id']}")
        complete = json.loads((path / "complete.json").read_text())
        if digest(path / "result.json") != complete["artifacts"]["result.json"]:
            raise ValueError(f"Result changed: {e['id']}")
        r = json.loads((path / "result.json").read_text())
        if r["experiment"] != e:
            raise ValueError(f"Experiment configuration changed: {e['id']}")
        results[e["id"]] = r
        row = {"id": e["id"], "scene": e["scene"], "subject": e["subject"],
               "condition": e["condition"], "frames": r["num_frames"], "total_spikes": r["total_spikes"]}
        for source in ["oms_model", "raw_events"]:
            for metric in ["p", "p_chance", "enrichment", "N_obj", "N_bg", "N_tot",
                           "A_obj", "A_tot", "kept_frames", "selectivity_index"]:
                row[f"{source}.{metric}"] = (r[source] or {}).get(metric)
        rows.append(row)
    if not rows:
        raise ValueError("No experiments configured")
    comparisons = []
    for comparison in config["analysis"].get("comparisons", []):
        source = comparison.get("source", "oms_model")
        if source not in {"oms_model", "raw_events"}:
            raise ValueError("Comparison source must be oms_model or raw_events")
        units = {}
        seen = set()
        for pair in comparison["pairs"]:
            key = (pair["a"], pair["b"])
            if key in seen or key[0] == key[1]:
                raise ValueError("Repeated or self-paired experiments are not allowed")
            seen.add(key)
            a, b = results[key[0]][source], results[key[1]][source]
            if not a or not b or a["p"] is None or b["p"] is None:
                raise ValueError(f"Undefined p for pair {key}; no silent exclusion")
            units.setdefault(pair["unit"], []).append(a["p"] - b["p"])
        differences = [float(np.mean(values)) for values in units.values()]
        value = paired_sign_flip(differences, seed=comparison.get("seed", 42))
        value.update(name=comparison["name"], source=source, unit_definition=comparison["unit_definition"],
                     unit_differences=dict(zip(units, differences)), pairs=comparison["pairs"])
        comparisons.append(value)
    adjusted = holm([r["p_value"] for r in comparisons])
    for r, p in zip(comparisons, adjusted):
        r["p_value_holm"] = p
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(root / "statistics.json", {
        "metric": "p = response counts in the configured object region / total response counts",
        "comparisons": comparisons,
        "note": "Comparisons are user-specified. These tests are not asserted to reproduce the paper's original statistical analysis.",
    })
    print(f"Saved {root / 'summary.csv'} and {root / 'statistics.json'}")
