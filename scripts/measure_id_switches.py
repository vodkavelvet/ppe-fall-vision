#!/usr/bin/env python
"""Ukur seberapa sering satu orang mendapat lebih dari satu track_id.

CAUCAFall hanya berisi satu orang per frame, jadi jumlah ID yang sehat per klip
adalah 1. Lebih dari itu berarti track terputus lalu diberi identitas baru.

Ini penting untuk deteksi jatuh karena orang yang jatuh mengubah bentuk kotaknya
drastis dalam beberapa frame. ByteTrack mencocokkan lewat IoU kotak, jadi
perubahan mendadak bisa memutus track tepat pada saat yang paling penting --
state machine Task 6 lalu kehilangan riwayatnya (FALLING tidak pernah sampai
FALLEN) dan lock pelanggaran Task 9 ter-reset.

Memakai bbox yang sudah tersimpan oleh extract_pose_caucafall.py, jadi tanpa
inference ulang.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from pfv.data.caucafall import FALL_ACTIVITIES, NATIVE_FPS
from pfv.io.frames import Frame, FrameTransform
from pfv.perception.base import PersonPose
from pfv.tracking import PersonTracker

POSE_DIR = Path("data/interim/caucafall/pose")

# Nama aktivitas dari nama berkas sudah huruf kecil ("fall backwards"), sedangkan
# FALL_ACTIVITIES berhuruf kapital di awal ("Fall backwards"). Membandingkan
# lewat .title() tidak pernah cocok ("Fall Backwards"), jadi samakan di sini.
FALL_LOWER = {a.lower() for a in FALL_ACTIVITIES}


def run_clip(npz_path: Path, **tracker_kwargs):
    data = np.load(npz_path)
    kp, bb, conf, det = data["keypoints"], data["bboxes"], data["confidence"], data["detected"]
    flags = data["fall_flags"]

    tracker = PersonTracker(fps=NATIVE_FPS, **tracker_kwargs)
    ids_over_time = []
    for i in range(len(det)):
        poses = []
        if det[i]:
            poses = [PersonPose(tuple(bb[i]), kp[i], float(conf[i]))]
        frame = Frame(
            index=i,
            image=np.zeros((2, 2, 3), np.uint8),
            timestamp_s=i / NATIVE_FPS,
            transform=FrameTransform(),
            source_size=(720, 480),
        )
        out = tracker.update(poses, frame)
        ids_over_time.append(out[0].track_id if out else None)
    return ids_over_time, flags


def first_switch_relative_to_fall(ids, flags):
    """Frame (relatif terhadap onset jatuh) saat ID berganti, atau None."""
    seen = None
    onset = int(np.argmax(flags)) if flags.any() else None
    for i, tid in enumerate(ids):
        if tid is None:
            continue
        if seen is None:
            seen = tid
        elif tid != seen:
            return (i - onset) if onset is not None else i
    return None


def main() -> int:
    configs = {
        "default (buffer 30)": {"lost_track_buffer": 30},
        "buffer 90": {"lost_track_buffer": 90},
        # Nilai LEBIH TINGGI = lebih permisif (ambang pada jarak 1-IoU). Percobaan
        # pertama dengan 0.5 memperburuk hasil jadi 85/100 karena arahnya terbalik.
        "buffer 30, match 0.9": {"lost_track_buffer": 30, "minimum_matching_threshold": 0.9},
        "buffer 30, match 0.95": {"lost_track_buffer": 30, "minimum_matching_threshold": 0.95},
        "buffer 30, match 0.99": {"lost_track_buffer": 30, "minimum_matching_threshold": 0.99},
    }

    for name, kwargs in configs.items():
        per_activity = defaultdict(list)
        switch_offsets = []
        for path in sorted(POSE_DIR.glob("*.npz")):
            activity = path.stem.split("_", 1)[1].replace("_", " ")
            ids, flags = run_clip(path, **kwargs)
            distinct = len({t for t in ids if t is not None})
            per_activity[activity].append(distinct)
            if distinct > 1 and activity in FALL_LOWER:
                off = first_switch_relative_to_fall(ids, flags)
                if off is not None:
                    switch_offsets.append(off)

        print(f"\n=== {name} ===")
        total_clips = sum(len(v) for v in per_activity.values())
        broken = sum(1 for v in per_activity.values() for d in v if d > 1)
        print(f"klip dengan >1 ID: {broken}/{total_clips}")
        for act in sorted(per_activity, key=lambda a: (a not in FALL_LOWER, a)):
            v = per_activity[act]
            print(f"  {act:16s} klip bermasalah {sum(1 for d in v if d > 1):2d}/{len(v)}  "
                  f"maks ID {max(v)}")
        if switch_offsets:
            before = sum(1 for o in switch_offsets if o < 0)
            near = sum(1 for o in switch_offsets if -5 <= o <= 20)
            print(f"  ID berganti pada klip jatuh: {len(switch_offsets)} kali; "
                  f"{near} di sekitar onset (-5..+20 frame); {before} sebelum onset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
