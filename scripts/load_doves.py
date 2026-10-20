"""Step 1: edit the paths below, then load one DOVES image and recording."""
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye_movements.doves import load_doves
from common_utils.io import save_json

IMAGE_PATH = Path('/path/to/Doves/Images/imk01366.iml')
RECORDING_PATH = Path('/path/to/Doves/Fixations/imk01366.iml.mat')
SUBJECT = 0  # zero-based eye_data index
OUT = Path('results/example')


def run(image_path=IMAGE_PATH, recording_path=RECORDING_PATH, subject=SUBJECT,
        out=OUT, image_shape=(768, 1024), contrast_boost=False):
    image, eye = load_doves(image_path, recording_path, subject, image_shape, contrast_boost)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(out / 'background.png')
    np.save(out / 'raw_eye.npy', eye)
    save_json(out / 'source.json', dict(image_path=str(image_path),
              recording_path=str(recording_path), subject=subject, contrast_boost=contrast_boost))


if __name__ == '__main__':
    run()
