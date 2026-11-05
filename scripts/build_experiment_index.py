#!/usr/bin/env python3
"""Build experiments/INDEX.md from the per-experiment README files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SECTIONS = ("目的", "Purpose")
MAX_SUMMARY_CHARS = 140


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown index of experiments/<group>/<name>/README.md (CPU only)."
    )
    parser.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--out", type=Path, default=None, help="default: <experiments>/INDEX.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the index is missing or out of date",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def truncate(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def first_paragraph(lines: list[str]) -> str:
    paragraph: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            if paragraph:
                break
            continue
        if in_code:
            continue
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        if not stripped or stripped.startswith(("|", "<", "---")):
            if paragraph:
                break
            continue
        paragraph.append(stripped.lstrip("-* ").strip() if not paragraph else stripped)
    return " ".join(paragraph)


def parse_readme(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Prefer the text right after the H1; fall back to the purpose section.
    body_start = next((i + 1 for i, line in enumerate(lines) if line.startswith("# ")), 0)
    head: list[str] = []
    for line in lines[body_start:]:
        if line.startswith("## "):
            break
        head.append(line)
    summary = first_paragraph(head)
    if not summary:
        for i, line in enumerate(lines):
            if line.startswith("## ") and line[3:].strip().startswith(SUMMARY_SECTIONS):
                summary = first_paragraph(lines[i + 1 :])
                break
    return clean_inline(title), truncate(clean_inline(summary))


def experiment_id(group: str, name: str) -> str:
    match = re.match(r"(\d+)[_-]", name)
    return f"{group}-{match.group(1)}" if match else f"{group}/{name}"


def has_results(exp_dir: Path) -> str:
    # A results/ holding only README.md (e.g. a not-yet-run plan) does not count.
    # Multi-PC experiments keep their outputs under runs/<name>/results/.
    for results in [exp_dir / "results", *sorted(exp_dir.glob("runs/*/results"))]:
        if results.is_dir() and any(p.name != "README.md" for p in results.iterdir()):
            return "✓"
    return ""


def build_index(experiments: Path) -> str:
    out: list[str] = [
        "# 実験インデックス",
        "",
        "`python scripts/build_experiment_index.py` が自動生成する。手で編集しない。",
        "全体の地図と読む順序は [README.md](README.md)、グループごとの問いと結論は各グループの `README.md` を参照する。",
        "概要列は各実験READMEのH1直後の段落（無ければ「目的」で始まる節の先頭段落）を切り詰めたもの。",
        "results列は `results/` または `runs/*/results/` にREADME以外の出力があれば✓。完了を意味しない。",
        "",
    ]

    # The complete Slack archive is local-only; index its versioned reference note instead.
    groups = sorted(
        (p for p in experiments.iterdir() if p.is_dir() and p.name != "slack_origin"),
        key=lambda p: natural_key(p.name),
    )
    for group in groups:
        rel_group = group.relative_to(experiments).as_posix()
        if (group / "README.md").exists() and not any(
            (child / "README.md").exists() for child in group.iterdir() if child.is_dir()
        ):
            # A group that is itself a single experiment (e.g. the 200k baseline) or a standalone note.
            title, summary = parse_readme(group / "README.md")
            out += [f"## {rel_group}", "", f"- [{title or rel_group}]({rel_group}/README.md)：{summary}", ""]
            continue

        children = sorted((p for p in group.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name))
        experiments_rows: list[str] = []
        materials: list[str] = [
            f"[{p.name}]({rel_group}/{p.name})"
            for p in sorted(group.glob("*.md"), key=lambda p: natural_key(p.name))
            if p.name != "README.md"
        ]
        for child in children:
            rel = child.relative_to(experiments).as_posix()
            readme = child / "README.md"
            # Stage directories (e.g. 4/stage0/) hold experiments one level deeper.
            nested = sorted(
                (p for p in child.iterdir() if p.is_dir() and (p / "README.md").exists()),
                key=lambda p: natural_key(p.name),
            )
            if readme.exists():
                title, summary = parse_readme(readme)
                exp_id = experiment_id(rel_group, child.name)
                experiments_rows.append(
                    f"| [{exp_id}]({rel}/README.md) | {title} | {summary} | {has_results(child)} |"
                )
            elif nested:
                for sub in nested:
                    title, summary = parse_readme(sub / "README.md")
                    exp_id = experiment_id(f"{rel_group}/{child.name}", sub.name)
                    experiments_rows.append(
                        f"| [{exp_id}]({rel}/{sub.name}/README.md) | {title} | {summary} | {has_results(sub)} |"
                    )
            else:
                materials.append(f"`{rel}/`")
        if not experiments_rows and not materials:
            continue

        heading = f"## {rel_group}"
        if (group / "README.md").exists():
            heading += f"（[グループREADME]({rel_group}/README.md)）"
        out += [heading, ""]
        if experiments_rows:
            out += ["| ID | タイトル | 概要 | results |", "|---|---|---|:-:|", *experiments_rows, ""]
        if materials:
            out += ["資料（実験READMEなし）：" + "、".join(materials), ""]

    notes = sorted(p.name for p in experiments.glob("*.md") if p.name not in ("INDEX.md", "README.md"))
    if notes:
        out += ["## 直下の資料", ""] + [f"- [{name}]({name})" for name in notes] + [""]
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    experiments = args.experiments.resolve()
    out_path = args.out or experiments / "INDEX.md"
    text = build_index(experiments)
    if args.check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if current != text:
            print(f"{out_path} is out of date; run scripts/build_experiment_index.py", file=sys.stderr)
            return 1
        return 0
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
