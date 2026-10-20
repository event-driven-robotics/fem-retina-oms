"""Step 6: score saved raw events and retina spikes against the union masks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_utils.io import save_json
from common_utils.utils import compute_object_selectivity

OUT = Path('results/example')


def run(out=OUT, vicinity_method='box', min_visibility=.5, exclude_edge_touch=False):
    out = Path(out)
    source = json.loads((out / 'source.json').read_text())
    motion = json.loads((out / 'motion.json').read_text())
    stimulus = json.loads((out / 'stimulus.json').read_text())
    retina = json.loads((out / 'retina_parameters.json').read_text())
    spikes = np.load(out / 'spikes.npy', mmap_mode='r')
    result = dict(scene=Path(source['image_path']).name, subject=source['subject'],
                  condition=motion['condition'], frames=len(spikes), objects=stimulus['objects'],
                  oms_model=None, raw_events=None, analysis=dict(vicinity_method=vicinity_method,
                  min_visibility=min_visibility, exclude_edge_touch=exclude_edge_touch,
                  ksize=retina['ksize']))
    if stimulus['objects']:
        masks = np.stack([np.asarray(Image.open(p)) for p in sorted((out / 'masks_union').glob('*.png'))])
        if masks.shape != spikes.shape:
            raise ValueError('Mask and spike dimensions must match')
        binned = np.load(out / 'binned_events.npy', mmap_mode='r')
        for label, response in [('oms_model', spikes), ('raw_events', binned.sum(axis=1))]:
            metrics = compute_object_selectivity(response, masks, ksize=retina['ksize'],
                min_visibility=min_visibility, exclude_edge_touch=exclude_edge_touch,
                vicinity_method=vicinity_method, return_more=True)
            result[label] = metrics[1] if isinstance(metrics, tuple) else metrics
    save_json(out / 'result.json', result)


if __name__ == '__main__':
    run()
