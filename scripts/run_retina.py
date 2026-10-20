"""Step 5: normalize saved event bins and compute raw OMS spikes."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_utils.io import save_json
from retina_model.oms_retina import normalize_bipolar_responces, run_oms_retina_

OUT = Path('results/example')


def run(out=OUT, ksize=39, tau_bp=.012, tau_am=.140, tau_var=.060,
        eps=.001, tau_syn=.005, tau_mem=.005, threshold=.3):
    out = Path(out)
    dt = json.loads((out / 'event_parameters.json').read_text())['dt']
    binned = np.load(out / 'binned_events.npy', mmap_mode='r')
    normalized = normalize_bipolar_responces(binned, ksize=ksize, dt=dt, tau_bp=tau_bp,
        tau_am=tau_am, tau_var=tau_var, eps=eps, return_only_normalized=True,
        output_path=out / 'normalized.npy')
    run_oms_retina_(normalized, dt=dt, tau_syn=tau_syn, tau_mem=tau_mem, threshold=threshold,
                   return_spikes_only=True, output_path=out / 'spikes.npy')
    save_json(out / 'retina_parameters.json', dict(dt=dt, ksize=ksize, tau_bp=tau_bp,
              tau_am=tau_am, tau_var=tau_var, eps=eps, tau_syn=tau_syn,
              tau_mem=tau_mem, threshold=threshold))


if __name__ == '__main__':
    run()
