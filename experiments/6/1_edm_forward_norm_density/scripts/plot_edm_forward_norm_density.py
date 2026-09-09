#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from training.dataset import ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EDM forward-noising norm density for an image dataset."
    )
    parser.add_argument("--data", default="datasets/cifar10-32x32.zip", help="Dataset ZIP or directory.")
    parser.add_argument("--out", default="experiments/6/1_edm_forward_norm_density/results/edm_forward_norm_density.png", help="Output PNG.")
    parser.add_argument("--max-images", type=int, default=5000, help="Number of images to use.")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for CPU processing.")
    parser.add_argument("--steps", type=int, default=256, help="Number of nonzero sigma values.")
    parser.add_argument("--sigma-min", type=float, default=0.002, help="Lowest nonzero EDM sigma.")
    parser.add_argument("--sigma-max", type=float, default=80.0, help="Highest EDM sigma.")
    parser.add_argument("--rho", type=float, default=7.0, help="EDM rho schedule exponent.")
    parser.add_argument("--norm-bins", type=int, default=240, help="Number of y-axis bins.")
    parser.add_argument("--norm-max", type=float, default=None, help="Upper y-axis limit.")
    parser.add_argument(
        "--mode",
        choices=["raw", "scaled"],
        default="scaled",
        help=(
            "raw plots ||x + sigma eps||/sqrt(dim). "
            "scaled divides by sqrt(sigma_data^2 + sigma^2) to show the density band."
        ),
    )
    parser.add_argument("--sigma-data", type=float, default=0.5, help="EDM sigma_data for scaled mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise.")
    return parser.parse_args()


def edm_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float) -> np.ndarray:
    ramp = np.linspace(0, 1, num_steps, dtype=np.float64)
    sigmas = (sigma_min ** (1 / rho) + ramp * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho
    return np.concatenate([[0.0], sigmas])


def load_images(path: str, max_images: int, batch: int) -> torch.Tensor:
    dataset = ImageFolderDataset(path=path, max_size=max_images, use_labels=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    chunks = []
    for images, _labels in loader:
        chunks.append(images.to(torch.float32) / 127.5 - 1)
    return torch.cat(chunks, dim=0)


def jet(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def save_heatmap(hist: np.ndarray, y_max: float, ylabel: str, title: str, out: Path) -> None:
    plot_w, plot_h = 900, 520
    left, right, top, bottom = 88, 24, 46, 64
    canvas_w = left + plot_w + right
    canvas_h = top + plot_h + bottom

    heat = jet(hist.T[::-1])
    heat_img = Image.fromarray(heat).resize((plot_w, plot_h), Image.Resampling.NEAREST)

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    img.paste(heat_img, (left, top))
    draw = ImageDraw.Draw(img)

    x0, y0 = left, top
    x1, y1 = left + plot_w, top + plot_h
    draw.rectangle([x0, y0, x1, y1], outline="black", width=1)

    for i in range(6):
        x = x0 + i * plot_w / 5
        step = round(i * (hist.shape[0] - 1) / 5)
        draw.line([(x, y1), (x, y1 + 5)], fill="black")
        draw.text((x - 12, y1 + 8), str(step), fill="black")

        y = y1 - i * plot_h / 5
        value = i * y_max / 5
        draw.line([(x0 - 5, y), (x0, y)], fill="black")
        draw.text((x0 - 56, y - 7), f"{value:.2g}", fill="black")

    draw.text((x0 + plot_w / 2 - 34, canvas_h - 28), "noise step", fill="black")
    draw.text((12, y0 + plot_h / 2 - 8), ylabel, fill="black")
    draw.text((x0 + 8, 16), title, fill="black")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    images = load_images(args.data, args.max_images, args.batch)
    sigmas = edm_sigmas(args.steps, args.sigma_min, args.sigma_max, args.rho)
    dim_sqrt = math.sqrt(float(images[0].numel()))

    if args.norm_max is None:
        args.norm_max = 1.4 if args.mode == "scaled" else max(2.0, args.sigma_max * 1.15)

    hist = np.zeros((len(sigmas), args.norm_bins), dtype=np.float64)
    y_edges = np.linspace(0, args.norm_max, args.norm_bins + 1, dtype=np.float64)

    for step_idx, sigma in enumerate(sigmas):
        for start in range(0, images.shape[0], args.batch):
            x0 = images[start : start + args.batch]
            xt = x0 + torch.randn_like(x0) * sigma
            norms = xt.flatten(1).norm(dim=1) / dim_sqrt
            if args.mode == "scaled":
                norms = norms / math.sqrt(args.sigma_data ** 2 + sigma ** 2)
            counts, _ = np.histogram(norms.numpy(), bins=y_edges)
            hist[step_idx] += counts

    hist = hist / np.maximum(hist.max(axis=1, keepdims=True), 1)

    ylabel = "scaled norm" if args.mode == "scaled" else "norm"
    out = Path(args.out)
    save_heatmap(
        hist=hist,
        y_max=args.norm_max,
        ylabel=ylabel,
        title=f"EDM forward noising norm density ({args.mode})",
        out=out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
