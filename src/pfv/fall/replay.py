"""Putar ulang keypoint tersimpan melalui tracker, tanpa inference.

Dipakai skrip analisis, demo, dan harness evaluasi Task 7b. Memutar ulang lewat
:class:`~pfv.tracking.PersonTracker` yang SAMA dengan pipeline nyata, bukan lewat
satu riwayat buatan, sehingga pemecahan ID (7 dari 100 klip pada ambang 0,95)
ikut terwakili. Mengevaluasi detektor pada riwayat yang lebih bersih daripada
yang akan dilihatnya di pipeline akan melambungkan hasil.

Hanya numpy: tidak ada OpenCV dan tidak ada model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from pfv.io.frames import Frame, FrameTransform
from pfv.perception.base import PersonPose
from pfv.tracking import PersonTracker, TrackedPerson

NATIVE_FPS = 20.0


@dataclass(frozen=True)
class ReplayStep:
    """Satu frame yang diputar ulang."""

    frame_index: int
    timestamp_s: float
    tracked: List[TrackedPerson]
    tracker: PersonTracker
    fall_label: bool
    """Ground truth: frame ini berada dalam periode tergeletak menurut label."""


def clip_subject(path: Path) -> int:
    """Nomor subjek dari nama berkas ``s03_fall_forward.npz``."""
    return int(Path(path).stem[1:3])


def clip_activity(path: Path) -> str:
    """Nama aktivitas huruf kecil, mis. ``'fall forward'``."""
    return Path(path).stem.split("_", 1)[1].replace("_", " ")


def replay_clip(
    npz_path: Path,
    *,
    keypoint_conf: float = 0.5,
    fps: float = NATIVE_FPS,
    tracker_kwargs: Optional[Dict[str, object]] = None,
) -> Iterator[ReplayStep]:
    """Putar ulang satu klip tersimpan, satu langkah per frame.

    Frame tanpa deteksi tetap diteruskan ke tracker dengan daftar kosong, persis
    seperti pipeline nyata; itulah yang membuat track bisa dipulihkan.
    """
    data = np.load(npz_path)
    keypoints, bboxes = data["keypoints"], data["bboxes"]
    confidence, detected = data["confidence"], data["detected"]
    flags = data["fall_flags"]

    tracker = PersonTracker(fps=fps, **(tracker_kwargs or {}))
    blank = np.zeros((2, 2, 3), dtype=np.uint8)

    for i in range(len(detected)):
        poses: List[PersonPose] = []
        if "frame_offsets" in data:
            start, stop = data["frame_offsets"][i:i + 2]
            poses = [PersonPose(
                bbox=tuple(float(v) for v in data["all_bboxes"][j]),
                keypoints=data["all_keypoints"][j],
                confidence=float(data["all_confidence"][j]),
                keypoint_conf=keypoint_conf,
            ) for j in range(int(start), int(stop))]
        elif detected[i]:
            poses = [PersonPose(
                bbox=tuple(float(v) for v in bboxes[i]),
                keypoints=keypoints[i], confidence=float(confidence[i]),
                keypoint_conf=keypoint_conf,
            )]
        frame = Frame(
            index=i,
            image=blank,
            timestamp_s=i / fps,
            transform=FrameTransform(),
            source_size=(720, 480),
        )
        yield ReplayStep(
            frame_index=i,
            timestamp_s=i / fps,
            tracked=tracker.update(poses, frame),
            tracker=tracker,
            fall_label=bool(flags[i]),
        )
