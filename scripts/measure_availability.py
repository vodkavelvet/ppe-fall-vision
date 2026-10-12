#!/usr/bin/env python
"""Seberapa sering tiap fitur benar-benar tersedia?

Demo ``pfv features`` menunjukkan hanya 50-65 persen frame yang punya ketujuh
fitur per frame. Itu penting: fitur dengan daya pisah terbaik tidak berguna kalau
jarang tersedia, dan Task 6 harus tahu fitur mana yang bisa diandalkan hadir.

Dipisah antara frame tegak dan tergeletak, karena ketersediaan justru bisa turun
tepat saat dibutuhkan -- orang yang tergeletak sering punya anggota badan
tertutup tubuhnya sendiri.

Memakai keypoint tersimpan, tanpa inference ulang.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

from pfv.data.caucafall import FALL_ACTIVITIES
from pfv.fall import compute_static
from pfv.perception.base import PersonPose

FALL_LOWER = {a.lower() for a in FALL_ACTIVITIES}

FEATURES = (
    "bbox_aspect_ratio",
    "trunk_angle_deg",
    "trunk_length_norm",
    "hip_knee_dy_norm",
    "hip_ankle_dy_norm",
    "head_hip_dy_norm",
    "keypoint_spread_y_norm",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--keypoint-conf", type=float, default=0.5)
    args = parser.parse_args()

    counts: Dict[str, Dict[str, int]] = {
        group: defaultdict(int) for group in ("upright", "fallen")
    }
    totals: Dict[str, int] = defaultdict(int)
    complete: Dict[str, int] = defaultdict(int)

    for path in sorted(args.pose_dir.glob("*.npz")):
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
                keypoint_conf=args.keypoint_conf,
            )
            group = "fallen" if flags[i] else "upright"
            totals[group] += 1
            feats = compute_static(pose)
            if feats.n_available == len(FEATURES):
                complete[group] += 1
            for name, value in feats.as_dict().items():
                if value is not None:
                    counts[group][name] += 1

    print("=" * 74)
    print(f"KETERSEDIAAN FITUR (ambang keypoint {args.keypoint_conf})")
    print("=" * 74)
    print(f"{'fitur':24s} {'tegak':>12s} {'tergeletak':>12s} {'selisih':>10s}")
    print("-" * 74)

    for name in FEATURES:
        up = 100.0 * counts["upright"][name] / max(1, totals["upright"])
        fa = 100.0 * counts["fallen"][name] / max(1, totals["fallen"])
        print(f"{name:24s} {up:11.0f}% {fa:11.0f}% {fa - up:+9.0f}%")

    print("-" * 74)
    for group in ("upright", "fallen"):
        pct = 100.0 * complete[group] / max(1, totals[group])
        print(f"{'ketujuh fitur lengkap':24s} {group:>12s} {pct:10.0f}%  "
              f"({totals[group]} frame)")

    print()
    print("Fitur yang ketersediaannya TURUN saat tergeletak paling berisiko: nilainya")
    print("hilang justru pada frame yang ingin dideteksi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
