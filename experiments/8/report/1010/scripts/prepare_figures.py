"""Prepare the figures for the 1010 progress report slides.

Crops the top title (which contains the experiment number) and the footer from the
t_min comparison figure, and snapshots the K-sweep figures, which are rewritten while
the sweep is still running. Run from the EDM repository root (Pillow required).
"""

import os

from PIL import Image

EXP_ROOT = 'experiments/8'
OUT_DIR = os.path.join(EXP_ROOT, 'report', '1010', 'assets')

TMIN_SRC = os.path.join(EXP_ROOT, '8_restart_tmin_comparison/results/restart_tmin_fid50000.png')
TMIN_BOX = (0, 62, None, 655)  # (left, top, right, bottom) in source pixels
K_SWEEP_DIR = os.path.join(EXP_ROOT, '11_restart_fixed_main18_tmin014_k_sweep/runs/20k-140k/results')
K_SWEEP_FILES = ['fixed_main18_tmin014_fid5000.png', 'fixed_main18_tmin014_fid25000.png']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    image = Image.open(TMIN_SRC)
    left, top, right, bottom = TMIN_BOX
    dst = os.path.join(OUT_DIR, 'tmin_comparison_fid50k.png')
    image.crop((left, top, right or image.size[0], bottom)).save(dst)
    print(f'{dst}: {TMIN_SRC} {TMIN_BOX}')
    for name in K_SWEEP_FILES:
        dst = os.path.join(OUT_DIR, name)
        Image.open(os.path.join(K_SWEEP_DIR, name)).save(dst)
        print(f'{dst}: snapshot of {name}')


if __name__ == '__main__':
    main()
