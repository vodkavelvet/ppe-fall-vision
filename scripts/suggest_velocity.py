#!/usr/bin/env python
"""Usulkan ambang kecepatan turun pinggul untuk memicu FALLING. Subjek latih saja.

Kecepatan dihitung lewat tracker yang sama dengan pipeline nyata (putar ulang
keypoint tersimpan), dengan jendela pendek. Dibandingkan dua kelompok:

* jendela di sekitar onset jatuh (10 frame sebelum sampai 15 frame sesudah)
* semua frame ADL, ditambah frame jauh sebelum jatuh

Hanya subjek 1-6 yang dibaca; subjek lain tidak pernah dimuat.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from pfv.fall import compute_temporal
from pfv.fall.replay import clip_activity, clip_subject, replay_clip

TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6)
FALL_ACTIVITIES = {"fall backwards", "fall forward", "fall left", "fall right", "fall sitting"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--window-s", type=float, default=0.5)
    args = parser.parse_args()

    near_onset: List[float] = []
    elsewhere: List[float] = []
    adl_peak: List[float] = []

    for path in sorted(args.pose_dir.glob("*.npz")):
        if clip_subject(path) not in TRAIN_SUBJECTS:
            continue
        activity = clip_activity(path)
        is_fall_clip = activity in FALL_ACTIVITIES
        flags = np.load(path)["fall_flags"]
        onset = int(np.argmax(flags)) if flags.any() else None

        clip_peak = 0.0
        for step in replay_clip(path):
            if not step.tracked:
                continue
            state = step.tracker.state(step.tracked[0].track_id)
            v = compute_temporal(state, window_s=args.window_s).hip_velocity_norm
            if v is None:
                continue
            if is_fall_clip and onset is not None and onset - 10 <= step.frame_index <= onset + 15:
                near_onset.append(v)
            else:
                elsewhere.append(v)
            if not is_fall_clip:
                clip_peak = max(clip_peak, v)
        if not is_fall_clip:
            adl_peak.append(clip_peak)

    near, other = np.asarray(near_onset), np.asarray(elsewhere)
    print(f"jendela {args.window_s}s | sekitar onset: {near.size} frame | lainnya: {other.size} frame")
    print(f"kecepatan turun (diagonal-kotak/detik), positif = turun\n")
    for label, arr in (("sekitar onset jatuh", near), ("frame lain", other)):
        q = np.quantile(arr, [0.5, 0.9, 0.99, 1.0])
        print(f"{label:22s} median {q[0]:+.2f}  p90 {q[1]:+.2f}  p99 {q[2]:+.2f}  maks {q[3]:+.2f}")
    print(f"puncak per klip ADL    median {np.median(adl_peak):+.2f}  maks {max(adl_peak):+.2f}\n")

    print(f"{'ambang':>8s} {'frame onset terlewat':>22s} {'frame lain terpicu':>20s}")
    for t in (0.1, 0.15, 0.2, 0.3, 0.4, 0.6):
        print(f"{t:8.2f} {np.mean(near >= t):21.0%} {np.mean(other >= t):19.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
