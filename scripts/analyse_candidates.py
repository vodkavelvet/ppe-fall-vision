#!/usr/bin/env python
"""Cari fitur yang memisahkan jatuh ke depan dari ADL, per subjek.

Analisis per subjek menunjukkan tidak satu pun dari tiga fitur awal menyelesaikan
jatuh searah sumbu kamera: rasio aspek hanya memisahkan pada 3 dari 10 subjek,
panjang trunk 7 dari 10.

Hipotesis untuk kandidat baru. Yang membedakan "jatuh ke depan lalu tergeletak"
dari "membungkuk mengambil objek" bukan orientasi trunk, melainkan **di mana
anggota badan berada relatif satu terhadap lain**:

    tergeletak  -> pinggul, lutut, dan pergelangan kaki semuanya di lantai
    membungkuk  -> kaki di lantai tetapi pinggul tetap TINGGI
    berlutut    -> lutut di lantai, pinggul sedang, kepala tegak

Jadi selisih tinggi pinggul terhadap lutut atau pergelangan kaki seharusnya
mendekati nol untuk semua posisi tergeletak, termasuk yang searah kamera, dan
tetap besar untuk membungkuk.

Semua dinormalisasi oleh diagonal kotak, bukan tingginya: tinggi kotak ikut
runtuh saat orang tergeletak dan akan menyembunyikan efek yang dicari.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pfv.data.caucafall import ADL_ACTIVITIES, FALL_ACTIVITIES

NOSE = 0
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16
KEYPOINT_CONF = 0.5

FALL_LOWER = [a.lower() for a in FALL_ACTIVITIES]
ADL_LOWER = [a.lower() for a in ADL_ACTIVITIES]

#: Semua kandidat diharapkan TURUN saat orang tergeletak.
CANDIDATES = (
    "trunk_norm",
    "hip_knee_dy",
    "hip_ankle_dy",
    "head_hip_dy",
    "kp_spread_y",
)


def centre(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    if kp[a, 2] < KEYPOINT_CONF or kp[b, 2] < KEYPOINT_CONF:
        return None
    return kp[[a, b], :2].mean(axis=0)


def any_valid_y(kp: np.ndarray, *indices: int) -> Optional[float]:
    """Rata-rata y dari keypoint yang valid. Pergelangan kaki sering tertutup,
    jadi satu sisi saja sudah cukup di sini -- berbeda dari pusat trunk yang
    butuh keduanya agar sudutnya tidak miring."""
    ys = [float(kp[i, 1]) for i in indices if kp[i, 2] >= KEYPOINT_CONF]
    return sum(ys) / len(ys) if ys else None


def frame_features(kp: np.ndarray, bbox: np.ndarray) -> Optional[Dict[str, float]]:
    shoulder = centre(kp, L_SH, R_SH)
    hip = centre(kp, L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None

    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if w <= 1 or h <= 1:
        return None
    diag = float(np.hypot(w, h))

    feats: Dict[str, float] = {
        "trunk_norm": float(np.hypot(*(hip - shoulder))) / diag,
    }

    knee_y = any_valid_y(kp, L_KNEE, R_KNEE)
    if knee_y is not None:
        feats["hip_knee_dy"] = abs(knee_y - float(hip[1])) / diag

    ankle_y = any_valid_y(kp, L_ANK, R_ANK)
    if ankle_y is not None:
        feats["hip_ankle_dy"] = abs(ankle_y - float(hip[1])) / diag

    if kp[NOSE, 2] >= KEYPOINT_CONF:
        feats["head_hip_dy"] = abs(float(kp[NOSE, 1]) - float(hip[1])) / diag

    valid_y = kp[kp[:, 2] >= KEYPOINT_CONF, 1]
    if valid_y.size >= 4:
        feats["kp_spread_y"] = float(valid_y.max() - valid_y.min()) / diag

    return feats


def collect(pose_dir: Path) -> Dict[Tuple[int, str], Dict[str, Dict[str, List[float]]]]:
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
            feats = frame_features(kp_all[i], bb_all[i])
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
        print(f"tidak ada .npz di {args.pose_dir}")
        return 2

    print("=" * 88)
    print("PEMISAHAN DARI ADL PER SUBJEK, untuk tiap kandidat fitur")
    print("=" * 88)
    print("Dihitung per subjek: median saat tergeletak harus lebih rendah dari")
    print("median ADL TERENDAH milik subjek itu sendiri. Perbandingan antar subjek")
    print("tidak sah karena tinggi badan dan jarak kamera berbeda.\n")

    header = f"{'aktivitas':16s} | " + " | ".join(f"{c:>13s}" for c in CANDIDATES)
    print(header)
    print("-" * len(header))

    for activity in FALL_LOWER:
        cells = []
        for feature in CANDIDATES:
            n_ok = n_total = 0
            margins: List[float] = []
            for subject in range(1, 11):
                bucket = stats.get((subject, activity))
                if bucket is None:
                    continue
                fall_value = med(bucket["fallen"].get(feature, []))
                adl_values = [
                    med(stats[(subject, a)]["upright"].get(feature, []))
                    for a in ADL_LOWER
                    if (subject, a) in stats
                ]
                adl_values = [v for v in adl_values if v is not None]
                if fall_value is None or not adl_values:
                    continue
                nearest = min(adl_values)
                n_total += 1
                if fall_value < nearest:
                    n_ok += 1
                if nearest > 0:
                    margins.append((nearest - fall_value) / nearest * 100.0)
            if n_total == 0:
                cells.append(f"{'-':>13s}")
            else:
                m = statistics.median(margins) if margins else 0.0
                cells.append(f"{n_ok:2d}/{n_total:<2d} {m:>+6.0f}%")
        print(f"{activity:16s} | " + " | ".join(cells))

    print()
    print("Angka kiri = subjek yang terpisah; kanan = median margin relatif")
    print("terhadap ADL terdekat. Margin positif berarti nilai jatuh lebih rendah.")

    print()
    print("=" * 88)
    print("RINCIAN JATUH KE DEPAN, kandidat terbaik versus yang lama")
    print("=" * 88)
    show = ("trunk_norm", "hip_knee_dy", "hip_ankle_dy")
    print(f"{'subjek':>7s} | " + " | ".join(f"{s:>22s}" for s in show))
    print(f"{'':>7s} | " + " | ".join(f"{'jatuh':>7s} {'ADL':>7s} {'ok':>5s}" for _ in show))
    print("-" * 80)
    for subject in range(1, 11):
        bucket = stats.get((subject, "fall forward"))
        if bucket is None:
            continue
        cells = []
        for feature in show:
            fall_value = med(bucket["fallen"].get(feature, []))
            adl_values = [
                med(stats[(subject, a)]["upright"].get(feature, []))
                for a in ADL_LOWER
                if (subject, a) in stats
            ]
            adl_values = [v for v in adl_values if v is not None]
            if fall_value is None or not adl_values:
                cells.append(f"{'-':>7s} {'-':>7s} {'-':>5s}")
                continue
            nearest = min(adl_values)
            cells.append(
                f"{fall_value:7.3f} {nearest:7.3f} {'ya' if fall_value < nearest else 'TIDAK':>5s}"
            )
        print(f"{subject:7d} | " + " | ".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
