from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


def format_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.{precision}g}"


def infer_name(identifier: str) -> str:
    candidate = identifier
    path = Path(identifier).expanduser()
    if path.exists():
        candidate = path.name
    else:
        parts = identifier.replace("\\", "/").rstrip("/").split("/")
        if parts:
            tail = parts[-1]
            candidate = tail or identifier
    return candidate or identifier


def slugify(name: str) -> str:
    if not name:
        return "model"
    chars: list[str] = []
    for ch in name:
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        elif ch == ".":
            chars.append("_")
        else:
            chars.append("-")
    slug = "".join(chars).strip("-_")
    return slug or "model"


def display_path(path: Path) -> str:
    text = str(path)
    if not path.is_absolute() and not text.startswith("./"):
        text = f"./{text}"
    return text


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_markdown_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("|", "\\|")
    if isinstance(value, (np.floating, float)):
        return format_float(float(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return format_float(float(value.item()))
        return "tensor"
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    return str(value)


def write_markdown_report(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    columns: Sequence[str] | None = None,
    summary_lines: Sequence[str] | None = None,
) -> None:
    if columns is None:
        columns = list(df.columns)
    lines: list[str] = [f"# {title}", ""]

    if summary_lines:
        lines.extend(summary_lines)
        lines.append("")

    header = " | ".join(columns)
    divider = " | ".join(["---"] * len(columns))
    lines.append(f"| {header} |")
    lines.append(f"| {divider} |")

    for _, row in df[columns].iterrows():
        cells = [format_markdown_cell(row[col]) for col in columns]
        lines.append(f"| {' | '.join(cells)} |")

    path.write_text("\n".join(lines), encoding="utf-8")
