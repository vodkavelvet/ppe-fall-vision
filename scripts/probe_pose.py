#!/usr/bin/env python
"""Probe pose estimation atas klip CAUCAFall nyata.

Tujuannya bukan demo, tetapi menjawab satu pertanyaan yang menentukan kelayakan
Task 5: **seberapa sering torso lengkap terdeteksi?** ``trunk_angle`` butuh kedua
bahu dan kedua pinggul sekaligus. Kalau angkanya rendah di dataset ini, fitur
utama deteksi jatuh tidak akan tersedia dan rencana Task 5-6 harus disesuaikan.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import List

from pfv.config import load_config
from pfv.data.caucafall import NATIVE_FPS, load_clip
from pfv.io import ImageSequenceSource
from pfv.perception import PersonPose
from pfv.perception.yolo_pose import YoloPoseEstimator

ROOT = Path("Dataset CAUCAFall/CAUCAFall")
CLIPS = [
    "Subject.1/Fall backwards",
    "Subject.3/Fall forward",
    "Subject.5/Kneel",
    "Subject.7/Walk",
]


def main() -> int:
    cfg = load_config()
    estimator = YoloPoseEstimator(
        model=cfg.pose.model,
        models_dir=cfg.pose.models_dir,
        device=cfg.pose.device,
        conf=cfg.pose.conf,
        keypoint_conf=cfg.pose.keypoint_conf,
        imgsz=cfg.pose.imgsz, rotations=cfg.pose.rotations, merge_iou=cfg.pose.merge_iou, rotation_conf=cfg.pose.rotation_conf,
    )
    estimator.load()
    print(f"device: {estimator.device}")
    print(f"bobot : {estimator.weights_path} "
          f"({estimator.weights_path.stat().st_size / 1e6:.1f} MB)")
    print()

    header = (
        f"{'klip':28s} {'frame':>6s} {'ada org':>8s} {'torso':>7s} "
        f"{'kepala':>7s} {'lutut':>7s} {'kp valid':>9s} {'>1 org':>7s}"
    )
    print(header)
    print("-" * len(header))

    for rel in CLIPS:
        clip = load_clip(ROOT / rel)
        with ImageSequenceSource(
            clip.image_paths, fps=NATIVE_FPS, resize=cfg.video.resize
        ) as src:
            n = n_person = n_torso = n_head = n_knee = n_multi = 0
            valid_counts: List[int] = []
            ratios_standing: List[float] = []
            ratios_fallen: List[float] = []

            for frame in src:
                n += 1
                poses: List[PersonPose] = estimator.estimate(frame.image)
                if not poses:
                    continue
                n_person += 1
                if len(poses) > 1:
                    n_multi += 1

                # Ambil orang dengan confidence tertinggi.
                p = max(poses, key=lambda q: q.confidence)
                valid_counts.append(p.n_valid())
                if p.has_full_torso():
                    n_torso += 1
                if any(p.valid[i] for i in (0, 1, 2, 3, 4)):
                    n_head += 1
                if any(p.valid[i] for i in (13, 14)):
                    n_knee += 1

                ar = p.bbox_aspect_ratio
                if ar is not None:
                    (ratios_fallen if clip.fall_flags[frame.index] else ratios_standing).append(ar)

        pct = lambda a, b: f"{100.0 * a / b:.0f}%" if b else "-"  # noqa: E731
        print(
            f"{rel:28s} {n:6d} {pct(n_person, n):>8s} {pct(n_torso, n):>7s} "
            f"{pct(n_head, n):>7s} {pct(n_knee, n):>7s} "
            f"{statistics.median(valid_counts) if valid_counts else 0:9.0f} "
            f"{pct(n_multi, n):>7s}"
        )

        if ratios_standing:
            print(f"{'':28s}   rasio kotak tegak  : median {statistics.median(ratios_standing):.2f}")
        if ratios_fallen:
            print(f"{'':28s}   rasio kotak terjatuh: median {statistics.median(ratios_fallen):.2f}")

    print()
    print("torso = kedua bahu DAN kedua pinggul valid -> prasyarat trunk_angle (Task 5)")
    print("kp valid = median jumlah keypoint valid dari 17")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
