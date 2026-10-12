#!/usr/bin/env python
"""Jalankan pose atas seluruh CAUCAFall, simpan hasilnya, dan laporkan statistik.

Dua tujuan.

**Pertama, menjawab pertanyaan yang menentukan Task 5:** seberapa sering torso
lengkap terdeteksi, dan apakah ``bbox_aspect_ratio`` benar-benar memisahkan
tegak dari tergeletak untuk **semua arah jatuh**. Probe awal menemukan bahwa pada
`Subject.3/Fall forward` rasio saat terjatuh justru LEBIH TINGGI daripada saat
tegak -- kalau itu berlaku umum untuk jatuh ke arah kamera, rasio kotak tidak
boleh menjadi fitur utama.

**Kedua, menyimpan keypoint ke disk** agar Task 5-7 bisa beriterasi tanpa
menjalankan ulang inference. Ini versi lokal dan sederhana dari Task 16: satu
``.npz`` per klip, hanya ORANG TERBAIK per frame. Itu cukup karena audit
menunjukkan CAUCAFall punya tepat satu orang per frame, tetapi berarti berkas ini
TIDAK bisa dipakai untuk menguji pertukaran ID antar orang.

Pemakaian::

    .venv/bin/python scripts/extract_pose_caucafall.py "Dataset CAUCAFall/CAUCAFall"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pfv.config import load_config
from pfv.data.caucafall import FALL_ACTIVITIES, NATIVE_FPS, discover_clips
from pfv.io import ImageSequenceSource
from pfv.perception.base import N_KEYPOINTS
from pfv.perception.yolo_pose import YoloPoseEstimator


def extract_clip(clip, estimator, resize) -> Dict[str, np.ndarray]:
    """Jalankan pose atas satu klip, ambil orang dengan confidence tertinggi."""
    n = clip.n_frames
    keypoints = np.zeros((n, N_KEYPOINTS, 3), dtype=np.float32)
    bboxes = np.zeros((n, 4), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    n_people = np.zeros(n, dtype=np.int16)

    with ImageSequenceSource(clip.image_paths, fps=NATIVE_FPS, resize=resize) as src:
        for frame in src:
            poses = estimator.estimate(frame.image)
            i = frame.index
            n_people[i] = len(poses)
            if not poses:
                continue
            best = max(poses, key=lambda p: p.confidence)
            keypoints[i] = best.keypoints
            bboxes[i] = best.bbox
            confidence[i] = best.confidence
            detected[i] = True

    return {
        "keypoints": keypoints,
        "bboxes": bboxes,
        "confidence": confidence,
        "detected": detected,
        "n_people": n_people,
        "fall_flags": np.asarray(clip.fall_flags, dtype=bool),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah klip")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    estimator = YoloPoseEstimator(
        model=cfg.pose.model,
        models_dir=cfg.pose.models_dir,
        device=cfg.pose.device,
        conf=cfg.pose.conf,
        keypoint_conf=cfg.pose.keypoint_conf,
        imgsz=cfg.pose.imgsz,
    )
    estimator.load()
    print(f"device: {estimator.device} | bobot: {estimator.weights_path}")

    clips = discover_clips(args.root)
    if args.limit:
        clips = clips[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    per_activity: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    started = time.perf_counter()
    total_frames = 0

    for n_done, clip in enumerate(clips, 1):
        path = args.out / f"{clip.clip_id}.npz"
        if args.skip_existing and path.exists():
            data = dict(np.load(path))
        else:
            data = extract_clip(clip, estimator, cfg.video.resize)
            np.savez_compressed(path, **data)

        total_frames += clip.n_frames
        stats = per_activity[clip.activity]

        detected = data["detected"]
        kp = data["keypoints"]
        flags = data["fall_flags"]
        bboxes = data["bboxes"]

        stats["detect_rate"].append(float(detected.mean()))

        # Torso lengkap: kedua bahu (5, 6) dan kedua pinggul (11, 12) valid.
        torso = (kp[:, [5, 6, 11, 12], 2] >= cfg.pose.keypoint_conf).all(axis=1) & detected
        stats["torso_rate"].append(float(torso.mean()))

        knees = (kp[:, [13, 14], 2] >= cfg.pose.keypoint_conf).any(axis=1) & detected
        stats["knee_rate"].append(float(knees.mean()))

        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        ok = detected & (widths > 1)
        ratios = np.where(ok, heights / np.maximum(widths, 1e-6), np.nan)

        upright = ratios[ok & ~flags]
        fallen = ratios[ok & flags]
        if upright.size:
            stats["ratio_upright"].append(float(np.median(upright)))
        if fallen.size:
            stats["ratio_fallen"].append(float(np.median(fallen)))
            # Apakah rasio kotak benar-benar MEMISAHKAN pada klip ini?
            if upright.size:
                stats["separates"].append(
                    float(np.median(fallen)) < float(np.median(upright))
                )

        if n_done % 10 == 0 or n_done == len(clips):
            rate = total_frames / max(1e-9, time.perf_counter() - started)
            print(f"  {n_done}/{len(clips)} klip, {total_frames} frame, {rate:.0f} fps")

    _report(per_activity, args.out)
    return 0


def _report(per_activity: Dict[str, Dict[str, list]], out: Path) -> None:
    def med(values: list) -> str:
        return f"{statistics.median(values):.2f}" if values else "-"

    def pct(values: list) -> str:
        return f"{100 * statistics.median(values):.0f}%" if values else "-"

    print()
    header = (
        f"{'aktivitas':16s} {'n':>3s} {'ada org':>8s} {'torso':>7s} {'lutut':>7s} "
        f"{'rasio tegak':>12s} {'rasio jatuh':>12s} {'memisahkan':>11s}"
    )
    print(header)
    print("-" * len(header))

    for activity in sorted(per_activity, key=lambda a: (a not in FALL_ACTIVITIES, a)):
        s = per_activity[activity]
        sep = s["separates"]
        sep_text = f"{sum(sep)}/{len(sep)}" if sep else "-"
        print(
            f"{activity:16s} {len(s['detect_rate']):3d} {pct(s['detect_rate']):>8s} "
            f"{pct(s['torso_rate']):>7s} {pct(s['knee_rate']):>7s} "
            f"{med(s['ratio_upright']):>12s} {med(s['ratio_fallen']):>12s} "
            f"{sep_text:>11s}"
        )

    print()
    print("torso      = kedua bahu DAN kedua pinggul valid -> prasyarat trunk_angle")
    print("memisahkan = jumlah klip yang rasio-jatuh-nya < rasio-tegak-nya")
    print(f"keypoint tersimpan di {out}/")


if __name__ == "__main__":
    raise SystemExit(main())
