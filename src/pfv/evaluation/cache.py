"""Cache fitur per klip, supaya grid search tidak mengulang tracker ribuan kali.

Putar ulang melalui tracker memakan waktu sekitar satu detik per klip. Grid search
atas ratusan konfigurasi harus memakai fitur yang sama berulang, jadi fitur dihitung
SEKALI per klip dan disimpan per track.

Cache hanya valid selama hal-hal yang memengaruhi fitur tidak berubah: ambang
keypoint, pengaturan tracker, ``velocity_window_s``, dan ``max_gap_s``. Semuanya
sengaja TIDAK ikut dituning. Ambang postur dan persistence tidak mempengaruhi
fitur, hanya cara memakainya, sehingga aman berubah tanpa membangun ulang cache.
Kesetaraan dengan pipeline nyata dijaga oleh test yang membandingkan simulasi
dengan ``FallMonitor`` pada klip yang sama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pfv.config import AppConfig
from pfv.fall.features import StaticFeatures, TemporalFeatures, compute_static, compute_temporal
from pfv.fall.replay import NATIVE_FPS, clip_activity, clip_subject, replay_clip

FALL_ACTIVITIES = ("fall backwards", "fall forward", "fall left", "fall right", "fall sitting")

#: Jatuh yang searah sumbu kamera. Sisanya menyamping atau dari duduk.
CAMERA_AXIS_ACTIVITIES = ("fall forward",)


@dataclass
class TrackSeq:
    """Urutan pengamatan satu ``track_id``."""

    track_id: int
    times: List[float] = field(default_factory=list)
    static: List[StaticFeatures] = field(default_factory=list)
    temporal: List[Optional[TemporalFeatures]] = field(default_factory=list)


@dataclass
class ClipData:
    name: str
    subject: int
    activity: str
    duration_s: float
    onset_s: Optional[float]
    """Awal periode tergeletak menurut label. ``None`` untuk klip ADL."""

    lying_s: float
    """Lama periode tergeletak menurut label, dalam detik."""

    tracks: List[TrackSeq] = field(default_factory=list)

    @property
    def is_fall(self) -> bool:
        return self.activity in FALL_ACTIVITIES

    @property
    def is_camera_axis(self) -> bool:
        return self.activity in CAMERA_AXIS_ACTIVITIES


def build_clip_data(npz_path: Path, cfg: AppConfig) -> ClipData:
    """Putar ulang satu klip melalui tracker dan simpan fitur per track."""
    t = cfg.tracking
    tracker_kwargs = dict(
        window_s=t.window_s,
        track_activation_threshold=t.track_activation_threshold,
        lost_track_buffer=t.lost_track_buffer,
        minimum_matching_threshold=t.minimum_matching_threshold,
        minimum_consecutive_frames=t.minimum_consecutive_frames,
        prune_after_s=t.prune_after_s,
    )

    flags = np.load(npz_path)["fall_flags"]
    onset_idx = int(np.argmax(flags)) if flags.any() else None

    tracks: Dict[int, TrackSeq] = {}
    for step in replay_clip(
        npz_path, keypoint_conf=cfg.pose.keypoint_conf, tracker_kwargs=tracker_kwargs
    ):
        for person in step.tracked:
            seq = tracks.setdefault(person.track_id, TrackSeq(person.track_id))
            state = step.tracker.state(person.track_id)
            seq.times.append(step.timestamp_s)
            seq.static.append(compute_static(person.pose))
            seq.temporal.append(
                compute_temporal(
                    state,
                    max_gap_s=t.max_gap_s,
                    window_s=cfg.fall.timing.velocity_window_s,
                )
            )

    return ClipData(
        name=npz_path.stem,
        subject=clip_subject(npz_path),
        activity=clip_activity(npz_path),
        duration_s=len(flags) / NATIVE_FPS,
        onset_s=(onset_idx / NATIVE_FPS) if onset_idx is not None else None,
        lying_s=float(flags.sum()) / NATIVE_FPS,
        tracks=list(tracks.values()),
    )
