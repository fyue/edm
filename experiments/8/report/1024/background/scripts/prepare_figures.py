"""Download the cited papers and crop their figures for the background slides.

Papers are fetched from arXiv into /tmp (not committed); only the cropped figures go to
assets/. Crop boxes are fractions of the rendered page (left, top, right, bottom), so they
do not depend on the rendering resolution. Run from the EDM repository root with the edm
environment (Pillow required); needs curl and pdftoppm.
"""

import os
import subprocess

from PIL import Image

OUT_DIR = os.path.join('experiments', '8', 'report', '1024', 'background', 'assets')
PAPER_DIR = '/tmp/deck_bg/papers'
DPI = 300

PAPERS = {
    'ddpm': '2006.11239',      # Ho et al., 2020, Denoising Diffusion Probabilistic Models
    'score_sde': '2011.13456', # Song et al., 2021, Score-Based Generative Modeling through SDEs
    'edm': '2206.00364',       # Karras et al., 2022, Elucidating the Design Space of Diffusion-Based Generative Models
    'restart': '2306.14878',   # Xu et al., 2023, Restart Sampling for Improving Generative Processes
}

# name: (paper, page, (left, top, right, bottom) as page fractions, dpi)
FIGURES = {
    'ddpm_fig2': ('ddpm', 2, (0.250, 0.085, 0.750, 0.151), DPI),
    'ddpm_fig1': ('ddpm', 1, (0.175, 0.589, 0.822, 0.887), DPI),
    'score_sde_fig2': ('score_sde', 4, (0.170, 0.097, 0.830, 0.270), DPI),
    'edm_fig1': ('edm', 2, (0.175, 0.081, 0.825, 0.159), DPI),
    'edm_fig2': ('edm', 4, (0.172, 0.088, 0.832, 0.200), DPI),
    'restart_fig1': ('restart', 3, (0.180, 0.090, 0.805, 0.276), DPI),
    'restart_fig5': ('restart', 9, (0.628, 0.150, 0.812, 0.285), 600),  # small figure
}


def fetch(arxiv_id):
    path = os.path.join(PAPER_DIR, f'{arxiv_id}.pdf')
    if not os.path.exists(path):
        os.makedirs(PAPER_DIR, exist_ok=True)
        subprocess.run(['curl', '-sSL', '-m', '120', '-o', path, f'https://arxiv.org/pdf/{arxiv_id}'], check=True)
    return path


def render(pdf, page, dpi):
    prefix = os.path.join(PAPER_DIR, f'{os.path.basename(pdf)[:-4]}_p{page}_{dpi}')
    png = f'{prefix}.png'
    if not os.path.exists(png):
        subprocess.run(['pdftoppm', '-r', str(dpi), '-f', str(page), '-l', str(page), '-png', '-singlefile', pdf, prefix], check=True)
    return png


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (paper, page, (left, top, right, bottom), dpi) in FIGURES.items():
        image = Image.open(render(fetch(PAPERS[paper]), page, dpi))
        w, h = image.size
        crop = image.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
        dst = os.path.join(OUT_DIR, f'{name}.png')
        crop.save(dst)
        print(f'{dst}: arXiv:{PAPERS[paper]} p{page} {crop.size}')


if __name__ == '__main__':
    main()
