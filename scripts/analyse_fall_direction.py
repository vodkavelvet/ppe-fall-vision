#!/usr/bin/env python
"""Uji apakah fitur geometri skeleton memulihkan jatuh searah sumbu kamera.

**Temuan yang diuji.** Ekstraksi pose atas 100 klip menunjukkan rasio aspek
bounding box runtuh dengan bersih untuk jatuh ke samping dan ke belakang (2,09
menjadi 0,54), tetapi hampir tidak berubah untuk jatuh ke depan (2,32 menjadi
1,88). Lebih buruk lagi, nilai 1,88 itu tidak bisa dibedakan dari aktivitas
harian: berlutut 1,82, mengambil objek 1,80, duduk 1,89.

**Hipotesisnya.** Jatuh ke depan terjadi searah sumbu kamera, sehingga tubuh
tidak melebar di bidang gambar melainkan MEMENDEK. Kalau benar, maka panjang
trunk -- jarak piksel dari pusat bahu ke pusat pinggul, dinormalisasi agar tidak
bergantung jarak kamera -- seharusnya menyusut tajam justru pada kasus yang
gagal ditangkap rasio aspek.

Dijalankan atas keypoint yang sudah tersimpan, jadi tidak ada inference ulang.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pfv.data.caucafall import ADL_ACTIVITIES, FALL_ACTIVITIES

# Indeks keypoint COCO.
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
KEYPOINT_CONF = 0.5


def centre(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    """Titik tengah dua keypoint, atau ``None`` bila salah satu tidak valid."""
    if kp[a, 2] < KEYPOINT_CONF or kp[b, 2] < KEYPOINT_CONF:
        return None
    return (kp[[a, b], :2]).mean(axis=0)


def frame_features(kp: np.ndarray, bbox: np.ndarray) -> Optional[Dict[str, float]]:
    """Hitung fitur satu frame. ``None`` bila torso tidak lengkap."""
    shoulder = centre(kp, L_SH, R_SH)
    hip = centre(kp, L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None

    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if w <= 1 or h <= 1:
        return None

    vec = hip - shoulder
    trunk_len = float(np.hypot(*vec))
    # Dinormalisasi oleh DIAGONAL kotak, bukan tingginya. Memakai tinggi akan
    # membuat normalisasinya ikut runtuh saat orang tergeletak, sehingga
    # pemendekan trunk tersembunyi oleh pembaginya sendiri.
    diag = float(np.hypot(w, h))

    # Sudut terhadap vertikal: 0 derajat tegak, 90 derajat mendatar.
    angle = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))

    return {
        "aspect_ratio": h / w,
        "trunk_norm": trunk_len / diag,
        "trunk_angle": angle,
    }


def collect(pose_dir: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Kumpulkan fitur per aktivitas, dipisah antara frame tegak dan tergeletak."""
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for path in sorted(pose_dir.glob("*.npz")):
        data = np.load(path)
        keypoints = data["keypoints"]
        bboxes = data["bboxes"]
        detected = data["detected"]
        flags = data["fall_flags"]

        # clip_id berbentuk s03_fall_forward -> aktivitas "fall forward"
        activity = path.stem.split("_", 1)[1].replace("_", " ")
        bucket = out.setdefault(activity, {"upright": {}, "fallen": {}})

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


def fmt(value: Optional[float], width: int = 6, digits: int = 2) -> str:
    return f"{value:{width}.{digits}f}" if value is not None else " " * (width - 1) + "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    args = parser.parse_args()

    stats = collect(args.pose_dir)
    if not stats:
        print(f"tidak ada .npz di {args.pose_dir}. Jalankan scripts/extract_pose.sh dulu.")
        return 2

    def order(name: str) -> tuple:
        canonical = name.title().replace("Fall ", "Fall ")
        return (canonical not in FALL_ACTIVITIES, name)

    print("=" * 92)
    print("PERBANDINGAN FITUR: rasio aspek kotak versus geometri skeleton")
    print("=" * 92)
    header = (
        f"{'aktivitas':16s} | {'rasio aspek':>21s} | {'panjang trunk (norm)':>22s} "
        f"| {'sudut trunk (deg)':>20s}"
    )
    print(header)
    print(
        f"{'':16s} | {'tegak':>7s} {'jatuh':>7s} {'delta':>5s} "
        f"| {'tegak':>7s} {'jatuh':>7s} {'delta':>5s} | {'tegak':>6s} {'jatuh':>6s} {'delta':>5s}"
    )
    print("-" * 92)

    for activity in sorted(stats, key=order):
        up = stats[activity]["upright"]
        fa = stats[activity]["fallen"]

        def pair(name: str) -> tuple:
            u, f = med(up.get(name, [])), med(fa.get(name, []))
            if u is None or f is None or u == 0:
                return u, f, None
            return u, f, (f - u) / u * 100.0

        ar_u, ar_f, ar_d = pair("aspect_ratio")
        tl_u, tl_f, tl_d = pair("trunk_norm")
        ta_u, ta_f, ta_d = pair("trunk_angle")

        def pct(value: Optional[float]) -> str:
            return f"{value:+4.0f}%" if value is not None else "   -"

        print(
            f"{activity:16s} | {fmt(ar_u)} {fmt(ar_f)} {pct(ar_d)} "
            f"| {fmt(tl_u, 7, 3)} {fmt(tl_f, 7, 3)} {pct(tl_d)} "
            f"| {fmt(ta_u, 6, 1)} {fmt(ta_f, 6, 1)} {pct(ta_d)}"
        )

    print()
    print("INTERPRETASI")
    print("-" * 92)

    lateral = ["fall left", "fall right", "fall backwards"]
    axial = ["fall forward"]

    def mean_delta(activities: List[str], feature: str) -> Optional[float]:
        deltas = []
        for a in activities:
            if a not in stats:
                continue
            u, f = med(stats[a]["upright"].get(feature, [])), med(
                stats[a]["fallen"].get(feature, [])
            )
            if u and f:
                deltas.append((f - u) / u * 100.0)
        return statistics.mean(deltas) if deltas else None

    for feature, label in (
            ("aspect_ratio", "rasio aspek"),
            ("trunk_norm", "panjang trunk"),
            ("trunk_angle", "sudut trunk"),
    ):
        lat = mean_delta(lateral, feature)
        ax = mean_delta(axial, feature)
        if lat is None or ax is None:
            continue
        ratio = abs(ax) / abs(lat) if lat else float("nan")
        print(
            f"{label:16s}: jatuh menyamping {lat:+6.0f}%  |  jatuh searah kamera "
            f"{ax:+6.0f}%  |  respons tersisa {ratio * 100:3.0f}%"
        )

    print()
    print("Kolom 'respons tersisa' adalah ukuran kerusakan fitur: makin kecil,")
    print("makin fitur itu gagal pada jatuh searah sumbu kamera.")

    print()
    print("PEMBEDAAN DARI ADL PENGGANGGU (median)")
    print("-" * 92)
    confounders = ["kneel", "pick up object", "sit down"]
    for feature, label in (
            ("aspect_ratio", "rasio aspek"),
            ("trunk_norm", "panjang trunk"),
            ("trunk_angle", "sudut trunk"),
    ):
        ff = med(stats.get("fall forward", {}).get("fallen", {}).get(feature, []))
        conf_values = [
            med(stats[c]["upright"].get(feature, [])) for c in confounders if c in stats
        ]
        conf_values = [v for v in conf_values if v is not None]
        if ff is None or not conf_values:
            continue
        lo, hi = min(conf_values), max(conf_values)
        inside = lo <= ff <= hi
        verdict = "TUMPANG TINDIH" if inside else "terpisah"
        print(
            f"{label:16s}: jatuh ke depan {ff:7.2f}  |  rentang ADL "
            f"{lo:6.2f}-{hi:6.2f}  |  {verdict}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
