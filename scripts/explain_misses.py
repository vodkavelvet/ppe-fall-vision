#!/usr/bin/env python
"""Kenapa sebuah klip jatuh tidak menghasilkan alert? Pisahkan tiga penyebab.

Tidak semua kelewatan sama, dan hanya satu jenis yang bisa diperbaiki dengan
menyetel ambang:

* STRUKTURAL  -- periode tergeletak pada label lebih pendek dari ``persistence_s``
                 (ditambah waktu yang dibutuhkan untuk terbaca). Algoritma apa pun
                 tidak mungkin menangkapnya tanpa menurunkan persistence.
* POSE        -- orang tidak terdeteksi pada sebagian besar periode tergeletak.
                 State machine tidak bisa memperbaiki masukan yang tidak ada.
* POSTUR      -- orang terdeteksi tetapi fiturnya tidak memenuhi syarat tergeletak.
                 Inilah satu-satunya yang menjadi urusan ambang (Task 7b).

Hanya subjek latih; penjaga yang sama dengan ``replay_fall.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from pfv.config import load_config
from pfv.fall import classify_posture, compute_static
from pfv.fall.replay import clip_activity, clip_subject, replay_clip

TRAIN = (1, 2, 3, 4, 5, 6)
FALL = ("fall backwards", "fall forward", "fall left", "fall right", "fall sitting")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    args = parser.parse_args()
    cfg = load_config()
    persistence = cfg.fall.timing.persistence_s

    print(f"persistence_s = {persistence}\n")
    print(f"{'klip':24s} {'tergeletak':>10s} {'orang terdeteksi':>17s} {'postur LYING':>13s}  penyebab")
    print("-" * 92)

    counts = {"STRUKTURAL": 0, "POSE": 0, "POSTUR": 0}
    n_missed = 0
    for path in sorted(args.pose_dir.glob("*.npz")):
        if clip_subject(path) not in TRAIN or clip_activity(path) not in FALL:
            continue
        flags = np.load(path)["fall_flags"]
        detected_in_label: List[bool] = []
        lying_in_label: List[bool] = []
        alerted = False

        from pfv.fall import FallMonitor

        monitor = FallMonitor(cfg.fall, max_gap_s=cfg.tracking.max_gap_s)
        for step in replay_clip(path, keypoint_conf=cfg.pose.keypoint_conf):
            for d in monitor.update(step.tracker, step.tracked, step.timestamp_s).values():
                alerted = alerted or d.alert
            if step.fall_label:
                detected_in_label.append(bool(step.tracked))
                if step.tracked:
                    r = classify_posture(compute_static(step.tracked[0].pose), cfg.fall.posture)
                    lying_in_label.append(r.posture.value == "lying")
                else:
                    lying_in_label.append(False)

        if alerted:
            continue
        n_missed += 1
        duration = flags.sum() / 20.0
        det = float(np.mean(detected_in_label)) if detected_in_label else 0.0
        lying = float(np.mean(lying_in_label)) if lying_in_label else 0.0

        if duration < persistence + 0.5:
            cause = "STRUKTURAL"
        elif det < 0.7:
            cause = "POSE"
        else:
            cause = "POSTUR"
        counts[cause] += 1
        print(f"{path.stem:24s} {duration:9.1f}s {det:16.0%} {lying:12.0%}  {cause}")

    print("-" * 92)
    print(f"kelewatan: {n_missed}  ->  " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("\nHanya POSTUR yang bisa diperbaiki dengan menyetel ambang (Task 7b).")
    print("STRUKTURAL adalah batas atas recall akibat klip yang dipotong, bukan kegagalan algoritma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
