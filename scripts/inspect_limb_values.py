#!/usr/bin/env python
"""Nilai indikator anggota badan pada klip jatuh yang ditolak kelompoknya.

Untuk memutuskan apakah penolakan itu karena ambang yang terlalu ketat, atau
karena estimasi keypoint kaki yang memang salah (kaki tertutup tubuh, pose
menebak). Hanya subjek latih.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from pfv.config import load_config
from pfv.fall import compute_static
from pfv.fall.replay import clip_subject, replay_clip

TRAIN = (1, 2, 3, 4, 5, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    args = parser.parse_args()
    cfg = load_config()
    thr = cfg.fall.posture

    print(f"ambang: hip_ankle<={thr.hip_ankle_dy_max}  hip_knee<={thr.hip_knee_dy_max}\n")
    for name in args.clips:
        path = args.pose_dir / f"{name}.npz"
        if clip_subject(path) not in TRAIN:
            print(f"DITOLAK {name}: bukan subjek latih")
            continue
        data = np.load(path)
        kp_all = data["keypoints"]
        conf = cfg.pose.keypoint_conf

        rows = {"hip_ankle": [], "hip_knee": [], "ankle_conf": [], "knee_conf": []}
        for step in replay_clip(path, keypoint_conf=conf):
            if not step.fall_label or not step.tracked:
                continue
            f = compute_static(step.tracked[0].pose)
            if f.hip_ankle_dy_norm is not None:
                rows["hip_ankle"].append(f.hip_ankle_dy_norm)
            if f.hip_knee_dy_norm is not None:
                rows["hip_knee"].append(f.hip_knee_dy_norm)
            kp = kp_all[step.frame_index]
            rows["ankle_conf"].append(float(kp[[15, 16], 2].max()))
            rows["knee_conf"].append(float(kp[[13, 14], 2].max()))

        def q(v, p):
            return float(np.quantile(v, p)) if v else float("nan")

        print(f"=== {name} (frame berlabel jatuh dengan orang terdeteksi: {len(rows['ankle_conf'])})")
        for key in ("hip_ankle", "hip_knee"):
            v = rows[key]
            print(f"  {key:10s} tersedia {len(v):3d}  p25 {q(v,.25):.3f}  median {q(v,.5):.3f}  "
                  f"p75 {q(v,.75):.3f}")
        print(f"  confidence kaki maks: median {statistics.median(rows['ankle_conf']):.2f} | "
              f"lutut maks: median {statistics.median(rows['knee_conf']):.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
