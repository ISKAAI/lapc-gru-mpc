"""Classify recursively collected highway MF4 logs without moving source files.

Default split policy:
  * all normal days except the newest two -> train_normal
  * second-newest normal day              -> val_normal
  * newest normal day                     -> test_normal
  * older abnormal/异常 logs   -> stress_dev
  * newest abnormal/异常 logs  -> stress_final (frozen final test)

Only files containing usable ``LCC active && ALC inactive`` segments are emitted
to a split. Byte-identical files found under conflicting normal/abnormal labels
are excluded from every split to prevent leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from asammdf import MDF

try:
    from data.channels import CHANNELS
except ModuleNotFoundError:  # Allow: python data/classify_highway_data.py
    from channels import CHANNELS


DEFAULT_DATA_ROOT = Path("/Users/kai/project/train/data")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "processed" / "highway_data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_label(relative_path: Path) -> str:
    parts = set(relative_path.parts)
    if "正常" in parts:
        return "normal"
    if "abnormal" in parts or "abormal" in parts or "异常" in parts:
        return "abnormal"
    return "unlabelled"


def scene_name(relative_path: Path) -> str:
    markers = {"正常", "abnormal", "abormal", "异常"}
    parts = list(relative_path.parts[:-1])
    for index, part in enumerate(parts):
        if part in markers:
            scene = "/".join(parts[index + 1 :])
            return scene or "unspecified"
    return "unlabelled"


def recording_date(path: Path) -> str:
    """Return the ISO recording date embedded in an XCP filename."""
    match = re.search(r"20\d{2}-\d{2}-\d{2}", path.name)
    if match is None:
        raise ValueError(f"Cannot determine recording date from filename: {path}")
    return match.group(0)


def active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(active, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], binary, [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def inspect_mf4(
    path: Path,
    trim: int = 20,
    min_len: int = 100,
    settle_threshold: float = 0.5,
    jump_threshold: float = 0.15,
) -> dict[str, object]:
    """Read required signals and reproduce the usable-segment policy causally."""
    try:
        mdf = MDF(path)
        signals = {}
        vx_timestamps = None
        for friendly_name, channel_name in CHANNELS.items():
            signal = mdf.get(channel_name, group=1)
            signals[friendly_name] = np.asarray(signal.samples, dtype=float)
            if friendly_name == "vx":
                vx_timestamps = np.asarray(signal.timestamps, dtype=float)
    except Exception as exc:  # Manifest should preserve failures instead of aborting.
        return {
            "read_ok": False,
            "read_error": f"{type(exc).__name__}: {exc}",
            "eligible_segments": 0,
            "eligible_samples": 0,
        }

    lengths = [len(values) for values in signals.values()]
    n = min(lengths, default=0)
    signals = {key: values[:n] for key, values in signals.items()}
    same_length = len(set(lengths)) == 1
    finite = all(np.isfinite(values).all() for values in signals.values())
    dt_median = (
        float(np.median(np.diff(vx_timestamps[:n])))
        if vx_timestamps is not None and n > 1
        else np.nan
    )

    if n == 0 or not finite:
        return {
            "read_ok": True,
            "read_error": "",
            "raw_samples": n,
            "same_length": same_length,
            "finite": finite,
            "dt_median": dt_median,
            "active_samples": 0,
            "eligible_segments": 0,
            "eligible_samples": 0,
        }

    active = (signals["lcc_active"] != 0) & (signals["alc_active"] == 0)
    eligible = []
    for start, end in active_runs(active):
        stop = max(start, end - trim)
        segment = {key: values[start:stop] for key, values in signals.items()}
        if len(segment["c0"]) == 0:
            continue

        jumps = np.flatnonzero(np.abs(np.diff(segment["c0"])) > jump_threshold)
        if len(jumps):
            segment = {key: values[: jumps[0] + 1] for key, values in segment.items()}

        settled = np.flatnonzero(np.abs(segment["c0"]) < settle_threshold)
        if len(settled) == 0:
            continue
        segment = {key: values[settled[0] :] for key, values in segment.items()}

        if len(segment["c0"]) < min_len:
            continue
        if min(segment["vx"]) < 2.0 or max(np.abs(segment["c0"])) > 2.0:
            continue
        eligible.append(segment)

    result: dict[str, object] = {
        "read_ok": True,
        "read_error": "",
        "raw_samples": n,
        "same_length": same_length,
        "finite": finite,
        "dt_median": dt_median,
        "active_samples": int(active.sum()),
        "eligible_segments": len(eligible),
        "eligible_samples": sum(len(segment["vx"]) for segment in eligible),
    }
    if eligible:
        vx = np.concatenate([segment["vx"] for segment in eligible])
        c0 = np.concatenate([segment["c0"] for segment in eligible])
        steer = np.concatenate([segment["steer"] for segment in eligible])
        result.update(
            vx_min=float(vx.min()),
            vx_median=float(np.median(vx)),
            vx_max=float(vx.max()),
            abs_c0_max=float(np.abs(c0).max()),
            c0_rms=float(np.sqrt(np.mean(np.square(c0)))),
            steer_rms=float(np.sqrt(np.mean(np.square(steer)))),
        )
    return result


def classify(data_root: Path, output_dir: Path) -> list[dict[str, object]]:
    paths = sorted(data_root.rglob("*.MF4"))
    if not paths:
        raise FileNotFoundError(f"No MF4 files found below {data_root}")

    rows = []
    for path in paths:
        relative = path.relative_to(data_root)
        row: dict[str, object] = {
            "path": str(path),
            "relative_path": str(relative),
            "day": relative.parts[0],
            "recording_date": recording_date(path),
            "source_label": source_label(relative),
            "scene": scene_name(relative),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        row.update(inspect_mf4(path))
        rows.append(row)

    hash_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        hash_groups[str(row["sha256"])].append(row)

    for group in hash_groups.values():
        labels = {str(row["source_label"]) for row in group}
        conflict = "normal" in labels and "abnormal" in labels
        canonical_path = min(str(row["path"]) for row in group)
        for row in group:
            row["duplicate_count"] = len(group)
            row["duplicate_of"] = "" if str(row["path"]) == canonical_path else canonical_path
            row["label_conflict"] = conflict

    normal_days = sorted(
        {str(row["recording_date"]) for row in rows if row["source_label"] == "normal"}
    )
    normal_split = {}
    if len(normal_days) >= 3:
        for day in normal_days[:-2]:
            normal_split[day] = "train_normal"
        normal_split[normal_days[-2]] = "val_normal"
        normal_split[normal_days[-1]] = "test_normal"
    elif len(normal_days) == 2:
        normal_split[normal_days[0]] = "train_normal"
        normal_split[normal_days[1]] = "test_normal"
    elif normal_days:
        normal_split[normal_days[0]] = "train_normal"

    seen_hashes = set()
    for row in rows:
        reason = ""
        split = ""
        digest = str(row["sha256"])
        if not row["read_ok"]:
            reason = "read_error"
        elif bool(row["label_conflict"]):
            reason = "hash_label_conflict"
        elif digest in seen_hashes:
            reason = "duplicate_content"
        elif int(row["eligible_segments"]) == 0:
            reason = "no_eligible_lcc_segment"
        elif row["source_label"] == "normal":
            split = normal_split.get(str(row["recording_date"]), "test_normal")
        elif row["source_label"] == "abnormal":
            split = (
                "stress_final"
                if str(row["recording_date"]) == normal_days[-1]
                else "stress_dev"
            )
        else:
            reason = "unlabelled_source"
        row["split"] = split
        row["exclude_reason"] = reason
        seen_hashes.add(digest)

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_names = ["train_normal", "val_normal", "test_normal", "stress_dev", "stress_final"]
    for split_name in split_names:
        selected = [str(row["path"]) for row in rows if row["split"] == split_name]
        (output_dir / f"{split_name}.txt").write_text(
            "".join(f"{path}\n" for path in selected), encoding="utf-8"
        )
    all_abnormal = [
        str(row["path"]) for row in rows if row["split"] in {"stress_dev", "stress_final"}
    ]
    (output_dir / "stress_abnormal.txt").write_text(
        "".join(f"{path}\n" for path in all_abnormal), encoding="utf-8"
    )
    excluded = [row for row in rows if row["exclude_reason"]]
    with (output_dir / "excluded.txt").open("w", encoding="utf-8") as stream:
        for row in excluded:
            stream.write(f"{row['exclude_reason']}\t{row['path']}\n")

    summary = {
        "data_root": str(data_root),
        "mf4_files": len(rows),
        "unique_hashes": len(hash_groups),
        "source_labels": dict(Counter(str(row["source_label"]) for row in rows)),
        "splits": dict(Counter(str(row["split"]) for row in rows if row["split"])),
        "excluded": dict(Counter(str(row["exclude_reason"]) for row in rows if row["exclude_reason"])),
        "split_eligible_samples": {
            split_name: sum(
                int(row["eligible_samples"]) for row in rows if row["split"] == split_name
            )
            for split_name in split_names
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = classify(args.data_root, args.output_dir)
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest -> {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
