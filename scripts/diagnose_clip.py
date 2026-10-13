#!/usr/bin/env python
"""Diagnosis satu klip: indikator mana yang memilih apa, dan kapan.

Untuk memahami KENAPA suatu klip menghasilkan alarm palsu atau terlewat, sebelum
memutuskan apakah perbaikannya ada di ambang atau di fitur. Hanya membaca;
tidak mengubah apa pun.

Penjaga yang sama dengan ``replay_fall.py``: subjek held-out ditolak.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pfv.config import load_config
from pfv.fall import classify_posture, compute_static
from pfv.fall.replay import clip_subject, replay_clip

HELDOUT = (7, 8, 9, 10)
SYMBOL = {"lying": "L", "upright": ".", "unknown": "?"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", help="mis. s02_sit_down")
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--include-heldout", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    for name in args.clips:
        path = args.pose_dir / f"{name}.npz"
        if clip_subject(path) in HELDOUT and not args.include_heldout:
            print(f"DITOLAK {name}: subjek held-out")
            continue

        import numpy as np

        flags = np.load(path)["fall_flags"]
        timeline = []
        votes = Counter()
        n = 0
        raw_values = {"trunk_angle_deg": [], "keypoint_spread_y_norm": [], "hip_ankle_dy_norm": [],
                      "hip_knee_dy_norm": [], "trunk_length_norm": []}
        for step in replay_clip(path, keypoint_conf=cfg.pose.keypoint_conf):
            if not step.tracked:
                timeline.append(" ")
                continue
            static = compute_static(step.tracked[0].pose)
            r = classify_posture(static, cfg.fall.posture)
            timeline.append(SYMBOL[r.posture.value])
            n += 1
            for k, v in r.votes.items():
                if v:
                    votes[k] += 1
            for k in raw_values:
                val = getattr(static, k)
                if val is not None and r.posture.value == "lying":
                    raw_values[k].append(val)

        print(f"=== {name} | {len(timeline)} frame | label fall: {int(flags.sum())} frame ===")
        # Satu karakter per 5 frame (0,25 detik): L bila mayoritas L.
        line, truth = "", ""
        for i in range(0, len(timeline), 5):
            chunk = timeline[i:i + 5]
            c = Counter(chunk)
            line += max(c, key=c.get) if chunk else " "
            truth += "#" if flags[i:i + 5].mean() >= 0.5 else "-"
        print(f"postur  : {line}")
        print(f"label   : {truth}")
        print("L=tergeletak  .=tegak  ?=tak tahu  |  # = label jatuh")
        print("suara tergeletak per indikator (porsi frame): " +
              ", ".join(f"{k}={votes[k]/max(1,n):.0%}" for k in
                        ("trunk_angle", "keypoint_spread", "hip_ankle", "hip_knee", "trunk_length")))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
