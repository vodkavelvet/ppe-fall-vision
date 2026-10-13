#!/usr/bin/env python
"""Apakah simulasi atas fitur ter-cache menghasilkan alert yang SAMA dengan
FallMonitor yang memakai tracker nyata?

Seluruh tuning dan evaluasi bergantung pada simulasi cepat. Kalau simulasi itu
menyimpang dari pipeline, angka yang dilaporkan mengukur sistem yang tidak ada.
Dibandingkan pada subjek latih saja: waktu alert pertama dan jumlah alert, klip
demi klip. Perbedaan sekecil apa pun dianggap kegagalan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pfv.config import load_config
from pfv.evaluation.cache import build_clip_data
from pfv.evaluation.search import TRAIN_SUBJECTS, assert_no_test_subjects
from pfv.evaluation.simulate import simulate_clip
from pfv.fall import FallMonitor
from pfv.fall.replay import clip_subject, replay_clip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    args = parser.parse_args()
    assert_no_test_subjects(TRAIN_SUBJECTS)

    cfg = load_config()
    t = cfg.tracking
    tk = dict(
        window_s=t.window_s, track_activation_threshold=t.track_activation_threshold,
        lost_track_buffer=t.lost_track_buffer,
        minimum_matching_threshold=t.minimum_matching_threshold,
        minimum_consecutive_frames=t.minimum_consecutive_frames,
        prune_after_s=t.prune_after_s,
    )

    n = mismatches = 0
    for path in sorted(args.pose_dir.glob("*.npz")):
        if clip_subject(path) not in TRAIN_SUBJECTS:
            continue
        n += 1

        monitor = FallMonitor(cfg.fall, max_gap_s=t.max_gap_s)
        real = []
        for step in replay_clip(path, keypoint_conf=cfg.pose.keypoint_conf, tracker_kwargs=tk):
            for d in monitor.update(step.tracker, step.tracked, step.timestamp_s).values():
                if d.alert:
                    real.append(round(d.timestamp_s, 6))

        sim = simulate_clip(build_clip_data(path, cfg), cfg.fall)
        simulated = [] if sim.first_alert_s is None else [round(sim.first_alert_s, 6)]

        if bool(real) != bool(simulated) or (real and abs(real[0] - simulated[0]) > 1e-6) \
                or len(real) != sim.n_alerts:
            mismatches += 1
            print(f"BEDA {path.stem}: pipeline {real} vs simulasi {sim.n_alerts} alert {simulated}")

    print(f"{n} klip subjek latih dibandingkan, {mismatches} berbeda")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
