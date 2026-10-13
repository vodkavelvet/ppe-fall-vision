#!/usr/bin/env python
"""Usulkan nilai AWAL ambang postur, hanya dari subjek latih.

Task 6 butuh ambang bawaan agar state machine bisa dijalankan. Pencarian ambang
yang sesungguhnya adalah Task 7b (grid search dengan split per subjek). Skrip ini
hanya memberi titik awal yang masuk akal, dengan satu aturan keras:

    HANYA subjek 1-6 yang dibaca. Subjek 7-8 (validasi) dan 9-10 (uji) tidak
    disentuh, dan skrip menolak berjalan bila diminta.

Aturan pemilihan sengaja sederhana dan konservatif. Untuk tiap fitur dipilih
ambang yang menangkap frame tergeletak sebanyak mungkin DENGAN laju salah-positif
per fitur tidak lebih dari ``--max-fpr`` pada frame non-jatuh (ADL dan frame
sebelum jatuh). Karena keputusan akhir memakai suara dari beberapa fitur, laju
salah-positif gabungan akan jauh lebih rendah daripada angka per fitur ini.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pfv.fall import compute_static
from pfv.perception.base import PersonPose

TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6)

#: (nama fitur, True bila TERGELETAK berarti nilai LEBIH TINGGI dari ambang)
FEATURES: Tuple[Tuple[str, bool], ...] = (
    ("trunk_angle_deg", True),
    ("keypoint_spread_y_norm", False),
    ("hip_ankle_dy_norm", False),
    ("hip_knee_dy_norm", False),
    ("trunk_length_norm", False),
)


def collect(pose_dir: Path, keypoint_conf: float) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {
        "lying": {n: [] for n, _ in FEATURES},
        "other": {n: [] for n, _ in FEATURES},
    }
    for path in sorted(pose_dir.glob("*.npz")):
        subject = int(path.stem[1:3])
        if subject not in TRAIN_SUBJECTS:
            continue  # penjaga: subjek lain tidak pernah dimuat
        data = np.load(path)
        kp_all, bb_all = data["keypoints"], data["bboxes"]
        detected, flags = data["detected"], data["fall_flags"]
        for i in range(len(detected)):
            if not detected[i]:
                continue
            pose = PersonPose(
                bbox=tuple(float(v) for v in bb_all[i]),
                keypoints=kp_all[i],
                confidence=1.0,
                keypoint_conf=keypoint_conf,
            )
            feats = compute_static(pose).as_dict()
            group = "lying" if flags[i] else "other"
            for name, _ in FEATURES:
                if feats.get(name) is not None:
                    out[group][name].append(float(feats[name]))
    return out


def best_threshold(
    lying: np.ndarray, other: np.ndarray, higher_is_lying: bool, max_fpr: float
) -> Optional[Tuple[float, float, float]]:
    """Ambang dengan TPR tertinggi di bawah batas FPR. Kembalikan (ambang, TPR, FPR)."""
    if lying.size == 0 or other.size == 0:
        return None
    candidates = np.unique(np.quantile(np.concatenate([lying, other]), np.linspace(0.01, 0.99, 197)))
    best: Optional[Tuple[float, float, float]] = None
    for t in candidates:
        if higher_is_lying:
            tpr, fpr = float((lying >= t).mean()), float((other >= t).mean())
        else:
            tpr, fpr = float((lying <= t).mean()), float((other <= t).mean())
        if fpr <= max_fpr and (best is None or tpr > best[1]):
            best = (float(t), tpr, fpr)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--keypoint-conf", type=float, default=0.5)
    parser.add_argument("--max-fpr", type=float, default=0.10)
    args = parser.parse_args()

    data = collect(args.pose_dir, args.keypoint_conf)
    n_lying = max(len(v) for v in data["lying"].values())
    n_other = max(len(v) for v in data["other"].values())
    print(f"Subjek latih {TRAIN_SUBJECTS}: ~{n_lying} frame tergeletak, ~{n_other} frame lain")
    print(f"Batas FPR per fitur: {args.max_fpr:.0%}\n")
    print(f"{'fitur':26s} {'arah':>6s} {'ambang':>9s} {'TPR':>7s} {'FPR':>7s}")
    print("-" * 60)
    for name, higher in FEATURES:
        result = best_threshold(
            np.asarray(data["lying"][name]), np.asarray(data["other"][name]), higher, args.max_fpr
        )
        arrow = ">=" if higher else "<="
        if result is None:
            print(f"{name:26s} {arrow:>6s} {'-':>9s}")
            continue
        t, tpr, fpr = result
        print(f"{name:26s} {arrow:>6s} {t:9.3f} {tpr:6.0%} {fpr:6.0%}")

    print("\nIni titik awal, bukan hasil akhir. Task 7b mencari ambang dengan grid search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
