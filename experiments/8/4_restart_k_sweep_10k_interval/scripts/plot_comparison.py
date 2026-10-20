#!/usr/bin/env python3
"""Draw Experiment 8-4 in the shared 8-11/8-12 style using compatible results."""

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[2]
CHECKPOINTS = tuple(f"{step}k" for step in range(20, 201, 20))
KS = tuple(range(9))
STAGES = (5000, 25000, 50000)
COLOR_LIMIT = 0.4
MISSING_COLOR = "#eeeeee"
NETWORK_DIR = (
    "experiments/0/1_cifar10_cond_edm_10k_retraining/training-runs/"
    "cifar10-cond-ddpmpp-edm-seed0-10k"
)
FID_REF = "fid-refs/cifar10-32x32.npz"
FIELDS = (
    "condition", "checkpoint", "num_images", "seeds", "restart_k", "main_steps",
    "restart_steps", "t_min", "t_max", "nfe", "fid", "network", "fid_ref",
    "source_experiment", "source_summary", "source_line", "source_condition",
)


def default_sources():
    """Highest-priority experiment first; use the 8-11 merged table if present."""
    group = REPO_ROOT / "experiments/8"
    baseline = group / "11_restart_fixed_main18_tmin014_k_sweep"
    merged = baseline / "results/restart_fid_summary.tsv"
    baseline_sources = [merged] if merged.is_file() else sorted(
        baseline.glob("runs/*/results/restart_fid_summary.tsv")
    )
    return [
        ("8-4", [EXP_ROOT / "results" / relative for relative in (
            "restart_fid_summary.tsv", "fid25k/restart_fid_summary.tsv",
            "fid50k/restart_fid_summary.tsv",
        )]),
        ("8-6", [group / "6_exact_10k_restart_k5_fid50k_1seed/results/restart_fid_summary.tsv"]),
        ("8-8", [group / "8_restart_tmin_comparison/results/restart_fid_summary.tsv"]),
        ("8-11", baseline_sources),
    ]


def same_repo_path(value, relative):
    # Absolute paths differ between PCs; compare the repository-relative identity.
    return value == relative or value.endswith("/" + relative)


def compatible_row(row):
    checkpoint = row["checkpoint"]
    if checkpoint not in CHECKPOINTS:
        return None
    k = int(row["restart_k"])
    count = int(row["num_images"])
    if k not in KS or count not in STAGES or int(row["main_steps"]) != 18:
        return None
    network = f"{NETWORK_DIR}/network-snapshot-{int(checkpoint[:-1]) * 1000:06d}.pkl"
    if not same_repo_path(row["network"], network) or not same_repo_path(row["fid_ref"], FID_REF):
        return None
    if row["seeds"] != f"0-{count - 1}" or int(row["nfe"]) != 35 + 4 * k:
        return None
    restart_steps = int(row["restart_steps"])
    if k:
        if restart_steps != 3 or float(row["t_min"]) != 0.06 or float(row["t_max"]) != 0.30:
            return None
    elif restart_steps != 0:
        return None
    # Older tables do not record these fields; their experiment defaults apply.
    for field, expected in (("rho", 7), ("restart_noise", 1), ("S_churn", 0),
                            ("gen_batch", 64), ("fid_batch", 64)):
        if row.get(field) not in (None, "") and float(row[field]) != expected:
            return None
    fid = float(row["fid"])
    if not math.isfinite(fid) or fid < 0:
        raise ValueError(f"Invalid FID: {row['fid']!r}")
    return dict(
        condition=f"{checkpoint}-k{k}", checkpoint=checkpoint, num_images=count,
        seeds=row["seeds"], restart_k=k, main_steps=18, restart_steps=restart_steps,
        t_min=0.06 if k else 0, t_max=0.30 if k else 0, nfe=35 + 4 * k,
        fid=fid, network=network, fid_ref=FID_REF, source_condition=row["condition"],
    )


def load_groups(sources=None):
    selected = {}
    for experiment, paths in default_sources() if sources is None else sources:
        within_source = {}
        for path in paths:
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for line, raw in enumerate(csv.DictReader(handle, delimiter="\t"), start=2):
                    try:
                        row = compatible_row(raw)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"{path}:{line}: {error}") from error
                    if row is None:
                        continue
                    key = row["checkpoint"], row["num_images"], row["restart_k"]
                    if key in within_source:
                        previous = within_source[key]
                        raise ValueError(
                            f"Duplicate compatible result in {experiment}: {key}; "
                            f"{previous['source_summary']}:{previous['source_line']} and {path}:{line}"
                        )
                    try:
                        source_path = str(path.relative_to(REPO_ROOT))
                    except ValueError:
                        source_path = str(path)
                    row.update(source_experiment=experiment, source_summary=source_path,
                               source_line=line)
                    within_source[key] = row
        for key, row in within_source.items():
            selected.setdefault(key, row)
    if not selected:
        raise ValueError("No compatible Experiment 8-4 results found")
    groups = {}
    for (checkpoint, count, k), row in selected.items():
        groups.setdefault((checkpoint, count), {})[k] = row
    return groups


def make_figure(groups, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    colormap = LinearSegmentedColormap.from_list(
        "fid_delta", ["#3b4cc0", "#ffffff", "#b40426"], N=257)
    colormap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, clip=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrix = np.full((len(CHECKPOINTS), len(KS)), np.nan)
    best_ks = np.full(len(CHECKPOINTS), np.nan)
    colors = plt.get_cmap("tab10")
    for i, checkpoint in enumerate(CHECKPOINTS):
        group = groups.get((checkpoint, count), {})
        if not group:
            continue
        ys = [group[k]["fid"] if k in group else np.nan for k in KS]
        axes[0].plot(KS, ys, marker="o", label=checkpoint, color=colors(i))
        if 0 in group:
            for j, k in enumerate(KS):
                if k in group:
                    matrix[i, j] = group[k]["fid"] - group[0]["fid"]
        if set(group) == set(KS):
            best_ks[i] = min(KS, key=lambda k: (group[k]["fid"], k))
    axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{count}, main=18, t_min=0.06")
    axes[0].set_xticks(KS)
    if axes[0].lines:
        axes[0].legend(fontsize=8)

    heatmap = axes[1].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=colormap, norm=norm)
    axes[1].set_xticks(range(len(KS)), KS)
    axes[1].set_yticks(range(len(CHECKPOINTS)), CHECKPOINTS)
    axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    below = bool(np.any(matrix < -COLOR_LIMIT))
    above = bool(np.any(matrix > COLOR_LIMIT))
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    figure.colorbar(heatmap, ax=axes[1], ticks=np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 9),
                   format=FormatStrFormatter("%.1f"), extend=extend)

    training = [int(checkpoint[:-1]) for checkpoint in CHECKPOINTS]
    axes[2].plot(training, best_ks, marker="o")
    axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
    axes[2].set_xticks(training, CHECKPOINTS)
    axes[2].set_yticks(KS)
    axes[2].set_ylim(-0.4, 8.4)
    if np.isnan(best_ks).all():
        axes[2].text(0.5, 0.5, "No complete K sweep", transform=axes[2].transAxes,
                     ha="center", va="center", color="dimgray")
    figure.tight_layout()
    return figure


def write_summary(groups, stages, outdir):
    path = outdir / "comparison_fid_summary.tsv"
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=outdir, prefix=".comparison-", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for count in stages:
            for checkpoint in CHECKPOINTS:
                for _, row in sorted(groups.get((checkpoint, count), {}).items()):
                    writer.writerow(row)
    temporary.replace(path)
    print(path)


def main(argv=None, sources=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fid-stage", choices=("all", "5000", "25000", "50000"), default="all")
    parser.add_argument("--outdir", type=Path, default=EXP_ROOT / "results")
    args = parser.parse_args(argv)
    groups = load_groups(sources)
    stages = STAGES if args.fid_stage == "all" else (int(args.fid_stage),)
    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(EXP_ROOT / "logs/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for count in stages:
        completed = sum(len(group) for (checkpoint, stage), group in groups.items() if stage == count)
        print(f"FID-{count}: {completed}/90 completed results")
        figure = make_figure(groups, count)
        try:
            for extension in ("png", "svg"):
                path = args.outdir / f"fixed_main18_tmin006_fid{count}.{extension}"
                temporary = path.with_name(f".{path.stem}.tmp.{extension}")
                figure.savefig(temporary, dpi=180)
                temporary.replace(path)
                print(path)
        finally:
            plt.close(figure)
    write_summary(groups, stages, args.outdir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
