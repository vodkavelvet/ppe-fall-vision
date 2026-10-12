#!/usr/bin/env python
"""Validasi modul fitur terhadap data nyata, memakai kode produksi.

Analisis sebelumnya (``analyse_candidates.py``) menghitung fitur dengan kodenya
sendiri. Skrip ini mengulang pengukuran kuncinya lewat :mod:`pfv.fall.features`,
sehingga yang diverifikasi adalah kode yang benar-benar dipakai pipeline. Kalau
angkanya berbeda, berarti ada kekeliruan di salah satu dari keduanya.

Dua pertanyaan yang dijawab:

1. Apakah angka per subjek dari modul produksi sama dengan hasil analisis?
2. Apakah **kombinasi** fitur benar-benar menutup lebih banyak subjek daripada
   fitur terbaik secara tunggal? Analisis sebelumnya menunjukkan ketiga fitur
   gagal pada subjek yang berbeda, dan hanya subjek 8 yang gagal di ketiganya.

Memakai keypoint tersimpan, tanpa inference ulang.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pfv.data.caucafall import ADL_ACTIVITIES, FALL_ACTIVITIES
from pfv.fall import compute_static
from pfv.perception.base import PersonPose

FALL_LOWER = [a.lower() for a in FALL_ACTIVITIES]
ADL_LOWER = [a.lower() for a in ADL_ACTIVITIES]

#: Fitur yang diharapkan TURUN saat orang tergeletak.
LOWER_WHEN_FALLEN = (
    "trunk_length_norm",
    "hip_knee_dy_norm",
    "hip_ankle_dy_norm",
    "keypoint_spread_y_norm",
    "bbox_aspect_ratio",
)


def collect(pose_dir: Path, keypoint_conf: float) -> Dict[Tuple[int, str], Dict[str, Dict[str, List[float]]]]:
    out: Dict[Tuple[int, str], Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(pose_dir.glob("*.npz")):
        subject = int(path.stem[1:3])
        activity = path.stem.split("_", 1)[1].replace("_", " ")
        data = np.load(path)
        kp_all, bb_all = data["keypoints"], data["bboxes"]
        detected, flags = data["detected"], data["fall_flags"]

        bucket = out.setdefault((subject, activity), {"upright": {}, "fallen": {}})
        for i in range(len(detected)):
            if not detected[i]:
                continue
            pose = PersonPose(
                bbox=tuple(float(v) for v in bb_all[i]),
                keypoints=kp_all[i],
                confidence=1.0,
                keypoint_conf=keypoint_conf,
            )
            group = "fallen" if flags[i] else "upright"
            for name, value in compute_static(pose).as_dict().items():
                if value is not None:
                    bucket[group].setdefault(name, []).append(value)
    return out


def med(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def separates(
    stats, subject: int, activity: str, feature: str
) -> Optional[bool]:
    """Apakah median tergeletak lebih rendah dari ADL TERENDAH subjek yang sama?"""
    bucket = stats.get((subject, activity))
    if bucket is None:
        return None
    fall_value = med(bucket["fallen"].get(feature, []))
    adl_values = [
        med(stats[(subject, a)]["upright"].get(feature, []))
        for a in ADL_LOWER
        if (subject, a) in stats
    ]
    adl_values = [v for v in adl_values if v is not None]
    if fall_value is None or not adl_values:
        return None
    return fall_value < min(adl_values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--keypoint-conf", type=float, default=0.5)
    args = parser.parse_args()

    stats = collect(args.pose_dir, args.keypoint_conf)
    if not stats:
        print(f"tidak ada .npz di {args.pose_dir}")
        return 2

    print("=" * 96)
    print("PEMISAHAN PER SUBJEK, dihitung lewat pfv.fall.features (kode produksi)")
    print("=" * 96)
    header = f"{'aktivitas':16s} | " + " | ".join(
        f"{f.replace('_norm','').replace('_dy',''):>12s}" for f in LOWER_WHEN_FALLEN
    )
    print(header)
    print("-" * len(header))

    for activity in FALL_LOWER:
        cells = []
        for feature in LOWER_WHEN_FALLEN:
            results = [separates(stats, s, activity, feature) for s in range(1, 11)]
            results = [r for r in results if r is not None]
            cells.append(f"{sum(results):2d}/{len(results):<2d}      " if results else f"{'-':>12s}")
        print(f"{activity:16s} | " + " | ".join(c[:12] for c in cells))

    print()
    print("=" * 96)
    print("APAKAH KOMBINASI MENUTUP LEBIH BANYAK? Jatuh ke depan, per subjek")
    print("=" * 96)
    combo = ("trunk_length_norm", "hip_knee_dy_norm", "hip_ankle_dy_norm")
    print(f"{'subjek':>7s} | " + " | ".join(f"{f.replace('_norm',''):>18s}" for f in combo) + " | gabungan")
    print("-" * 86)

    n_any = n_total = 0
    per_feature = {f: 0 for f in combo}
    for subject in range(1, 11):
        flags = [separates(stats, subject, "fall forward", f) for f in combo]
        if all(f is None for f in flags):
            continue
        n_total += 1
        any_ok = any(bool(f) for f in flags)
        n_any += any_ok
        for f, ok in zip(combo, flags):
            per_feature[f] += bool(ok)
        cells = [f"{'ya' if ok else 'TIDAK':>18s}" for ok in flags]
        print(f"{subject:7d} | " + " | ".join(cells) + f" | {'ya' if any_ok else 'TIDAK'}")

    print()
    for f in combo:
        print(f"{f:24s}: {per_feature[f]}/{n_total} subjek")
    print(f"{'kombinasi (salah satu)':24s}: {n_any}/{n_total} subjek")

    print()
    print("Catatan: 'gabungan' di sini hanya disjungsi dari pemisahan per fitur, bukan")
    print("klasifier. Ambang sesungguhnya dicari Task 7b dengan split per subjek, dan")
    print("Task 6 menambahkan persistence yang kemungkinan menutup sisanya.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
