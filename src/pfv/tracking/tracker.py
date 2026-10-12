"""Tracking orang dengan ByteTrack, plus riwayat per orang.

Dua hal yang dibangun di sini dan dipakai hampir semua task setelahnya.

**``track_id`` yang bertahan.** Terverifikasi secara empiris pada supervision
0.30.5: ByteTrack memulihkan ``track_id`` yang sama setelah orangnya tidak
terdeteksi selama beberapa frame. Itulah yang membuat lock pelanggaran di Task 9
tidak mereset hanya karena seseorang sesaat membelakangi kamera.

**Riwayat yang menyimpan timestamp, bukan hanya urutan.** Ini yang mudah
terlewat. Ketika sebuah track hilang lalu muncul lagi, riwayatnya punya LUBANG.
Kalau kecepatan dihitung dengan mengasumsikan jarak antar sampel seragam, maka
pada lubang itu kecepatannya akan tampak melonjak -- dan lonjakan palsu itu
persis pola yang dicari detektor jatuh. Karena itu setiap sampel membawa
``timestamp_s``, dan :meth:`TrackState.is_contiguous` menyediakan cara memeriksa
apakah jendelanya layak dipakai.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pfv.io.frames import Frame
from pfv.perception.base import Keypoint, PersonPose

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_S = 3.0
_WINDOW_EPS_S = 1e-6


@dataclass(frozen=True)
class TrackSample:
    """Satu pengamatan seorang yang dilacak, pada satu frame."""

    frame_index: int
    timestamp_s: float
    pose: PersonPose


@dataclass(frozen=True)
class TrackedPerson:
    """Hasil per frame: pose yang sudah punya identitas."""

    track_id: int
    pose: PersonPose

    @property
    def label(self) -> str:
        return f"ID:{self.track_id}"


@dataclass
class TrackState:
    """Riwayat satu orang dalam jendela waktu bergerak.

    Panjang jendela dibatasi oleh WAKTU, bukan jumlah frame, supaya perilakunya
    sama pada sumber 20 fps maupun 30 fps.
    """

    track_id: int
    window_s: float = DEFAULT_WINDOW_S
    samples: Deque[TrackSample] = field(default_factory=deque)
    first_seen_s: Optional[float] = None
    last_seen_s: Optional[float] = None
    n_observations: int = 0

    def append(self, sample: TrackSample) -> None:
        self.samples.append(sample)
        self.n_observations += 1
        if self.first_seen_s is None:
            self.first_seen_s = sample.timestamp_s
        self.last_seen_s = sample.timestamp_s
        self._trim()

    def _trim(self) -> None:
        """Buang sampel yang lebih tua dari ``window_s`` dari sampel terbaru."""
        if not self.samples:
            return
        # Toleransi kecil karena timestamp adalah hasil bagi float: 2.95 - 1.0
        # menghasilkan 1.9500000000000002, sehingga sampel di 1.95 (tepat di tepi
        # jendela) terbuang secara acak tergantung pembulatan.
        cutoff = self.samples[-1].timestamp_s - self.window_s - _WINDOW_EPS_S
        while len(self.samples) > 1 and self.samples[0].timestamp_s < cutoff:
            self.samples.popleft()

    # ----------------------------------------------------------------- akses

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def latest(self) -> Optional[TrackSample]:
        return self.samples[-1] if self.samples else None

    @property
    def oldest(self) -> Optional[TrackSample]:
        return self.samples[0] if self.samples else None

    @property
    def span_s(self) -> float:
        """Rentang waktu yang tercakup riwayat saat ini."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp_s - self.samples[0].timestamp_s

    @property
    def age_s(self) -> float:
        """Lama orang ini terlihat, sejak pertama kali muncul."""
        if self.first_seen_s is None or self.last_seen_s is None:
            return 0.0
        return self.last_seen_s - self.first_seen_s

    def is_contiguous(self, *, max_gap_s: float) -> bool:
        """``True`` bila tidak ada jeda lebih dari ``max_gap_s`` di riwayat.

        Wajib diperiksa sebelum menghitung kecepatan. Jeda berarti orangnya
        sempat tidak terdeteksi, dan kecepatan yang melintasi jeda itu tidak
        bermakna -- justru akan terlihat seperti gerakan mendadak.
        """
        if len(self.samples) < 2:
            return True
        for a, b in zip(self.samples, list(self.samples)[1:]):
            if b.timestamp_s - a.timestamp_s > max_gap_s:
                return False
        return True

    def keypoint_series(
        self, index: int
    ) -> List[Tuple[float, Optional[Tuple[float, float]]]]:
        """Riwayat satu keypoint sebagai ``(timestamp, titik atau None)``.

        ``None`` dipertahankan alih-alih dilewati, supaya pemanggil tahu bahwa
        keypoint-nya sempat tidak valid dan bisa memutuskan sendiri.
        """
        return [(s.timestamp_s, s.pose.keypoint(index)) for s in self.samples]

    def hip_centre_series(self) -> List[Tuple[float, Optional[Tuple[float, float]]]]:
        """Riwayat pusat pinggul. Sumber ``hip_vertical_velocity`` di Task 5."""
        return [(s.timestamp_s, s.pose.hip_centre()) for s in self.samples]

    def shoulder_centre_series(self) -> List[Tuple[float, Optional[Tuple[float, float]]]]:
        return [(s.timestamp_s, s.pose.shoulder_centre()) for s in self.samples]

    def bbox_series(self) -> List[Tuple[float, Tuple[float, float, float, float]]]:
        return [(s.timestamp_s, s.pose.bbox) for s in self.samples]

    def aspect_ratio_series(self) -> List[Tuple[float, Optional[float]]]:
        return [(s.timestamp_s, s.pose.bbox_aspect_ratio) for s in self.samples]

    def torso_valid_ratio(self) -> float:
        """Porsi sampel dalam jendela yang torso-nya lengkap.

        Dipakai Task 6 dan 8 untuk memutuskan apakah bukti cukup kuat, atau
        statusnya harus ``UNKNOWN``.
        """
        if not self.samples:
            return 0.0
        n_ok = sum(1 for s in self.samples if s.pose.has_full_torso())
        return n_ok / len(self.samples)


class PersonTracker:
    """Pembungkus ByteTrack yang memelihara :class:`TrackState` per orang.

    Contoh::

        tracker = PersonTracker(fps=20.0)
        for frame in source:
            poses = estimator.estimate(frame.image)
            tracked = tracker.update(poses, frame)
            for person in tracked:
                state = tracker.state(person.track_id)
    """

    def __init__(
        self,
        *,
        fps: float,
        window_s: float = DEFAULT_WINDOW_S,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.95,
        minimum_consecutive_frames: int = 1,
        prune_after_s: float = 5.0,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps harus positif, diberikan {fps}")
        if window_s <= 0:
            raise ValueError(f"window_s harus positif, diberikan {window_s}")

        self.fps = float(fps)
        self.window_s = float(window_s)
        self.prune_after_s = float(prune_after_s)
        self._states: Dict[int, TrackState] = {}
        self._last_timestamp_s: float = 0.0

        # supervision 0.30.5 membungkus ByteTrack dengan dekorator deprecation,
        # jadi impornya memunculkan FutureWarning yang tidak bisa kita perbaiki.
        # Parameter di bawah sudah diverifikasi diterima lewat scripts/probe_tracker.py.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import supervision as sv

            self._sv = sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=int(round(self.fps)),
                minimum_consecutive_frames=minimum_consecutive_frames,
            )

    # ---------------------------------------------------------------- update

    def update(self, poses: Sequence[PersonPose], frame: Frame) -> List[TrackedPerson]:
        """Beri identitas pada pose di satu frame, lalu perbarui riwayatnya.

        Mengembalikan daftar kosong bila tidak ada orang. Tracker tetap
        diperbarui pada frame kosong, karena itulah yang membuatnya tahu sebuah
        track sedang hilang dan bisa memulihkannya nanti.
        """
        self._last_timestamp_s = frame.timestamp_s
        detections = self._to_detections(poses)
        tracked = self._tracker.update_with_detections(detections)

        results: List[TrackedPerson] = []
        if tracked.tracker_id is None or len(tracked) == 0:
            self._prune()
            return results

        # Pemetaan balik memakai src_index yang dibawa lewat `data`, BUKAN
        # mengandalkan urutan keluaran. Probe menunjukkan urutannya terjaga pada
        # kasus sederhana, tetapi bergantung pada itu akan gagal secara diam-diam
        # begitu ada deteksi yang tidak terpasangkan.
        src = tracked.data.get("src_index") if tracked.data else None
        for row in range(len(tracked)):
            track_id = int(tracked.tracker_id[row])
            if src is not None:
                pose = poses[int(src[row])]
            else:  # pragma: no cover
                pose = poses[row]

            state = self._states.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id, window_s=self.window_s)
                self._states[track_id] = state
            state.append(
                TrackSample(
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    pose=pose,
                )
            )
            results.append(TrackedPerson(track_id=track_id, pose=pose))

        self._prune()
        return results

    def _to_detections(self, poses: Sequence[PersonPose]):
        """Ubah daftar :class:`PersonPose` menjadi ``sv.Detections``."""
        n = len(poses)
        if n == 0:
            return self._sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=int),
                data={"src_index": np.zeros(0, dtype=int)},
            )
        return self._sv.Detections(
            xyxy=np.array([p.bbox for p in poses], dtype=np.float32),
            confidence=np.array([p.confidence for p in poses], dtype=np.float32),
            class_id=np.zeros(n, dtype=int),
            data={"src_index": np.arange(n, dtype=int)},
        )

    def _prune(self) -> None:
        """Lupakan track yang sudah lama tidak terlihat, agar memori tidak tumbuh."""
        cutoff = self._last_timestamp_s - self.prune_after_s
        stale = [
            tid
            for tid, st in self._states.items()
            if st.last_seen_s is not None and st.last_seen_s < cutoff
        ]
        for tid in stale:
            del self._states[tid]
        if stale:
            logger.debug("melupakan track tak aktif: %s", stale)

    # ----------------------------------------------------------------- akses

    def state(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    @property
    def states(self) -> Dict[int, TrackState]:
        return dict(self._states)

    @property
    def active_ids(self) -> List[int]:
        return sorted(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def reset(self) -> None:
        """Mulai dari nol. Dipakai saat berpindah klip dalam satu proses."""
        self._states.clear()
        self._last_timestamp_s = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker.reset()
