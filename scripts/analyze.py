"""Step 7: collect results and optionally compare explicitly paired independent units."""
import csv
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_utils.io import save_json
from common_utils.statistics import paired_sign_flip

RESULTS_ROOT = Path('results')
# Optional (result_a_path, result_b_path, independent_unit_label) triples.
# Repeat a unit label to average matched scene differences within that unit.
PAIRS = []


def run(results_root=RESULTS_ROOT, pairs=PAIRS):
    root = Path(results_root)
    paths = sorted(root.glob('*/result.json'))
    if not paths:
        raise ValueError('No experiment result.json files found')
    rows = []
    for path in paths:
        result = json.loads(path.read_text())
        row = dict(experiment=path.parent.name, scene=result['scene'],
                   subject=result['subject'], condition=result['condition'], frames=result['frames'],
                   objects=json.dumps(result['objects']))
        for source in ['oms_model', 'raw_events']:
            for key in ['p', 'p_chance', 'N_obj', 'N_bg', 'N_tot', 'A_obj', 'A_tot', 'kept_frames']:
                row[f'{source}.{key}'] = (result[source] or {}).get(key)
        rows.append(row)
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if pairs:
        units = {}
        seen = set()
        for a, b, unit in pairs:
            a, b = Path(a).resolve(), Path(b).resolve()
            key = frozenset((a, b))
            if a == b or key in seen:
                raise ValueError('Duplicate or self-paired result files')
            seen.add(key)
            values = []
            for path in [a, b]:
                metric = json.loads(path.read_text())['oms_model']
                if not metric or metric['p'] is None:
                    raise ValueError(f'Undefined object selectivity: {path}')
                values.append(metric['p'])
            units.setdefault(unit, []).append(values[0] - values[1])
        differences = {unit: float(np.mean(values)) for unit, values in units.items()}
        result = paired_sign_flip(list(differences.values()), seed=42)
        result.update(unit_differences=differences, pairs=[(str(a), str(b), unit) for a, b, unit in pairs])
        save_json(root / 'comparison.json', result)


if __name__ == '__main__':
    run()
