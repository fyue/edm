from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "2"
OUT = EXP_DIR / "results" / "conditional_generation_120k_vs_200k.png"

CHECKPOINTS = ["120k", "200k"]
CLASS_NAMES = [
    "0 airplane",
    "1 automobile",
    "2 bird",
    "3 cat",
    "4 deer",
    "5 dog",
    "6 frog",
    "7 horse",
    "8 ship",
    "9 truck",
]
SEEDS_TO_SHOW = list(range(8))

TILE = 64
SRC = 32
PAD = 8
LABEL_W = 150
HEADER_H = 46
ROW_H = TILE + PAD
CLASS_GAP = 14
COL_W = TILE + PAD


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    font = load_font(16)
    small = load_font(13)

    rows = len(CHECKPOINTS) * len(CLASS_NAMES)
    width = LABEL_W + len(SEEDS_TO_SHOW) * COL_W + PAD
    height = HEADER_H + rows * ROW_H + (len(CLASS_NAMES) - 1) * CLASS_GAP + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((PAD, 12), "Checkpoint / class", fill=(20, 20, 20), font=small)
    for i, seed in enumerate(SEEDS_TO_SHOW):
        x = LABEL_W + i * COL_W
        draw.text((x + 14, 12), f"seed {seed}", fill=(20, 20, 20), font=small)

    y = HEADER_H
    missing = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for ckpt in CHECKPOINTS:
            label = f"{ckpt}  {class_name}"
            draw.text((PAD, y + 20), label, fill=(20, 20, 20), font=font)
            for i, seed in enumerate(SEEDS_TO_SHOW):
                path = EXP_DIR / "generated" / ckpt / f"class{class_idx}" / f"{seed:06d}.png"
                x = LABEL_W + i * COL_W
                if path.exists():
                    img = Image.open(path).convert("RGB").resize((TILE, TILE), Image.Resampling.NEAREST)
                    canvas.paste(img, (x, y))
                else:
                    missing.append(str(path))
                    draw.rectangle((x, y, x + TILE - 1, y + TILE - 1), outline=(220, 50, 50), width=2)
                    draw.text((x + 8, y + 24), "missing", fill=(220, 50, 50), font=small)
            y += ROW_H
        if class_idx != len(CLASS_NAMES) - 1:
            y += CLASS_GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)

    if missing:
        missing_file = EXP_DIR / "logs" / "conditional-grid-missing.txt"
        missing_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise SystemExit(f"Missing {len(missing)} images. See {missing_file}")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
