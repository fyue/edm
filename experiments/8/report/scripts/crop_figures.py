"""Crop the experiment figures for the progress report slides.

Removes the top title (which contains the experiment number) and the English
footer, and splits the FID-vs-K grid into one image per checkpoint.
Run from the EDM repository root with the edm environment (Pillow required).
"""

import os

from PIL import Image

EXP_ROOT = 'experiments/8'
OUT_DIR = os.path.join(EXP_ROOT, 'report', 'assets')

# Crop boxes are in source pixels: (left, top, right, bottom).
EXISTING_SRC = os.path.join(EXP_ROOT, '3_restart_k5_fid50k_3seeds/results/restart_k5_fid50k_3seed_overview.png')
EXISTING_BOX = (0, 142, None, 883)
UNIFORM_SRC = os.path.join(EXP_ROOT, '6_exact_10k_restart_k5_fid50k_1seed/results/restart_k5_fid25000_overview.png')
UNIFORM_BOX = (0, 146, None, 1017)
K_SWEEP_SRC = os.path.join(EXP_ROOT, '4_restart_k_sweep_10k_interval/results/fid_vs_k_by_checkpoint.png')
K_SWEEP_COLS = [(0, 795), (799, 1590), (1594, 2385), (2389, 3179)]
K_SWEEP_ROWS = [(283, 787), (815, 1327), (1351, 1873)]
K_SWEEP_NAMES = [['10k', '20k', '30k', '40k'], ['50k', '60k', '70k', '80k'], ['90k', '100k', None, None]]


def crop(src, box, dst):
    image = Image.open(src)
    left, top, right, bottom = box
    image.crop((left, top, right or image.size[0], bottom)).save(dst)
    print(f'{dst}: {src} {box}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    crop(EXISTING_SRC, EXISTING_BOX, os.path.join(OUT_DIR, 'existing_model_fid50k.png'))
    crop(UNIFORM_SRC, UNIFORM_BOX, os.path.join(OUT_DIR, 'uniform_checkpoints_fid25k.png'))
    for row, (top, bottom) in zip(K_SWEEP_NAMES, K_SWEEP_ROWS):
        for name, (left, right) in zip(row, K_SWEEP_COLS):
            if name is not None:
                crop(K_SWEEP_SRC, (left, top, right, bottom), os.path.join(OUT_DIR, f'k_sweep_{name}.png'))


if __name__ == '__main__':
    main()
