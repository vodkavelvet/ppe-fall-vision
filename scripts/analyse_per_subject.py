#!/usr/bin/env python
"""Apakah temuan arah jatuh konsisten PER SUBJEK, atau hanya artefak agregat?

Analisis sebelumnya (``analyse_fall_direction.py``) memakai median yang digabung
antar klip. Itu tidak cukup untuk diklaim: margin bisa berasal dari dua atau tiga
subjek saja sementara sisanya tidak menunjukkan apa pun.

Skrip ini menghitung per (subjek, aktivitas), lalu melaporkan **berapa dari 10
subjek** yang menunjukkan pemisahan pada arah yang diharapkan, beserta sebaran
nilainya. Itu yang bisa dipertanggungjawabkan.

Arah yang diharapkan saat orang tergeletak:
    rasio aspek   turun   (kotak jadi mendatar)
    sudut trunk   naik    (trunk jadi mendatar)
    panjang trunk turun   (trunk memendek karena perspektif)

Memakai keypoint tersimpan, tanpa inference ulang.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pfv.data.caucafall import ADL_ACTIVITIES, FALL_ACTIVITIES

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
KEYPOINT_CONF = 0.5

FALL_LOWER = [a.lower() for a in FALL_ACTIVITIES]
ADL_LOWER = [a.lower() for a in ADL_ACTIVITIES]

#: Arah perubahan yang diharapkan saat tergeletak, per fitur.
EXPECTED_SIGN = {"aspect_ratio": -1, "trunk_angle": +1, "trunk_norm": -1}


def centre(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    if kp[a, 2] < KEYPOINT_CONF or kp[b, 2] < KEYPOINT_CONF:
        return None
    return kp[[a, b], :2].mean(axis=0)


def frame_features(kp: np.ndarray, bbox: np.ndarray) -> Optional[Dict[str, float]]:
    shoulder = centre(kp, L_SH, R_SH)
    hip = centre(kp, L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None

    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if w <= 1 or h <= 1:
        return None

    vec = hip - shoulder
    # Dinormalisasi oleh DIAGONAL kotak, bukan tingginya: tinggi ikut runtuh saat
    # orang tergeletak, sehingga pemendekan trunk akan tersembunyi oleh pembaginya.
    diag = float(np.hypot(w, h))

    return {
        "aspect_ratio": h / w,
        "trunk_angle": float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))),
        "trunk_norm": float(np.hypot(*vec)) / diag,
    }


def collect(pose_dir: Path) -> Dict[Tuple[int, str], Dict[str, Dict[str, List[float]]]]:
    """Kunci ``(subject_id, activity)`` -> grup -> fitur -> daftar nilai."""
    out: Dict[Tuple[int, str], Dict[str, Dict[str, List[float]]]] = {}

    for path in sorted(pose_dir.glob("*.npz")):
        # clip_id: s03_fall_forward
        stem = path.stem
        subject = int(stem[1:3])
        activity = stem.split("_", 1)[1].replace("_", " ")

        data = np.load(path)
        keypoints, bboxes = data["keypoints"], data["bboxes"]
        detected, flags = data["detected"], data["fall_flags"]

        bucket = out.setdefault((subject, activity), {"upright": {}, "fallen": {}})
        for i in range(len(detected)):
            if not detected[i]:
                continue
            feats = frame_features(keypoints[i], bboxes[i])
            if feats is None:
                continue
            group = "fallen" if flags[i] else "upright"
            for name, value in feats.items():
                bucket[group].setdefault(name, []).append(value)

    return out


def med(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    args = parser.parse_args()

    stats = collect(args.pose_dir)
    if not stats:
        print(f"tidak ada .npz di {args.pose_dir}. Jalankan scripts/extract_pose.sh dulu.")
        return 2

    print("=" * 94)
    print("KONSISTENSI PER SUBJEK: berapa dari 10 subjek yang menunjukkan pemisahan?")
    print("=" * 94)
    print("Pemisahan = perubahan median tegak -> tergeletak mengikuti arah yang diharapkan")
    print("(rasio aspek turun, sudut trunk naik, panjang trunk turun).\n")

    header = f"{'aktivitas':16s} | " + " | ".join(
        f"{label:>26s}" for label in ("rasio aspek", "sudut trunk", "panjang trunk")
    )
    print(header)
    print(f"{'':16s} | " + " | ".join(f"{'n/10':>6s} {'median delta':>12s} {'min':>6s}" for _ in range(3)))
    print("-" * len(header))

    for activity in FALL_LOWER:
        cells = []
        for feature in ("aspect_ratio", "trunk_angle", "trunk_norm"):
            deltas: List[float] = []
            for subject in range(1, 11):
                bucket = stats.get((subject, activity))
                if bucket is None:
                    continue
                u, f = med(bucket["upright"].get(feature, [])), med(
                    bucket["fallen"].get(feature, [])
                )
                if u is None or f is None or u == 0:
                    continue
                deltas.append((f - u) / u * 100.0)

            if not deltas:
                cells.append(f"{'-':>6s} {'-':>12s} {'-':>6s}")
                continue

            sign = EXPECTED_SIGN[feature]
            n_ok = sum(1 for d in deltas if d * sign > 0)
            # "min" di sini = delta paling LEMAH ke arah yang diharapkan, yaitu
            # subjek terburuk. Itu angka yang menentukan apakah fitur bisa dipakai.
            weakest = min(d * sign for d in deltas)
            cells.append(
                f"{n_ok:>2d}/{len(deltas):<3d} {statistics.median(deltas):>+11.0f}% "
                f"{weakest:>+5.0f}%"
            )

        print(f"{activity:16s} | " + " | ".join(cells))

    print()
    print("=" * 94)
    print("PEMISAHAN DARI ADL: apakah ada ambang tunggal yang bekerja per subjek?")
    print("=" * 94)
    print("Untuk tiap subjek: nilai median saat jatuh ke depan versus median ADL")
    print("terdekat milik SUBJEK ITU SENDIRI. Perbandingan antar subjek tidak sah")
    print("karena tinggi badan dan jarak ke kamera berbeda.\n")

    print(f"{'subjek':>7s} | {'rasio aspek':>28s} | {'panjang trunk':>28s}")
    print(f"{'':>7s} | {'jatuh':>8s} {'ADL terdekat':>13s} {'ok':>4s} | "
          f"{'jatuh':>8s} {'ADL terdekat':>13s} {'ok':>4s}")
    print("-" * 72)

    summary = {"aspect_ratio": 0, "trunk_norm": 0}
    n_subjects = 0
    for subject in range(1, 11):
        ff = stats.get((subject, "fall forward"))
        if ff is None:
            continue
        n_subjects += 1
        cells = []
        for feature in ("aspect_ratio", "trunk_norm"):
            fall_value = med(ff["fallen"].get(feature, []))
            adl_values = [
                med(stats[(subject, a)]["upright"].get(feature, []))
                for a in ADL_LOWER
                if (subject, a) in stats
            ]
            adl_values = [v for v in adl_values if v is not None]
            if fall_value is None or not adl_values:
                cells.append(f"{'-':>8s} {'-':>13s} {'-':>4s}")
                continue
            # Diharapkan lebih rendah dari semua ADL, jadi pembandingnya ADL terendah.
            nearest = min(adl_values)
            ok = fall_value < nearest
            summary[feature] += ok
            cells.append(f"{fall_value:8.2f} {nearest:13.2f} {'ya' if ok else 'TIDAK':>4s}")
        print(f"{subject:7d} | " + " | ".join(cells))

    print()
    for feature, label in (("aspect_ratio", "rasio aspek"), ("trunk_norm", "panjang trunk")):
        print(f"{label:16s}: {summary[feature]}/{n_subjects} subjek terpisah dari ADL-nya sendiri")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
