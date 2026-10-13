"""Satu state machine per track, dihubungkan ke tracker.

Satu-satunya modul di paket ``fall`` yang tahu soal tracking. State machine dan
fiturnya tetap murni; di sini hanya perekatnya.
"""

from __future__ import annotations

from typing import Dict, Sequence

from pfv.config import FallConfig
from pfv.fall.features import compute_static, compute_temporal
from pfv.fall.state_machine import FallDecision, FallStateMachine
from pfv.tracking import PersonTracker, TrackedPerson


class FallMonitor:
    """Menjalankan :class:`FallStateMachine` untuk setiap orang yang dilacak.

    Contoh::

        monitor = FallMonitor(cfg.fall, max_gap_s=cfg.tracking.max_gap_s)
        for frame in source:
            tracked = tracker.update(estimator.estimate(frame.image), frame)
            decisions = monitor.update(tracker, tracked, frame.timestamp_s)
            for track_id, d in decisions.items():
                if d.alert:
                    ...

    Catatan untuk Task 9: bila ByteTrack memecah satu orang menjadi dua ID (7 dari
    100 klip CAUCAFall pada ambang 0,95), orang yang MASIH tergeletak akan
    mendapat mesin baru dan bisa memicu alert kedua. Cooldown di sini per track;
    cooldown berbasis waktu dan posisi ada di rule engine.
    """

    def __init__(self, cfg: FallConfig, *, max_gap_s: float = 0.2) -> None:
        self._cfg = cfg
        self._max_gap_s = max_gap_s
        self._machines: Dict[int, FallStateMachine] = {}

    def update(
        self,
        tracker: PersonTracker,
        tracked: Sequence[TrackedPerson],
        timestamp_s: float,
    ) -> Dict[int, FallDecision]:
        decisions: Dict[int, FallDecision] = {}
        for person in tracked:
            state = tracker.state(person.track_id)
            static = compute_static(person.pose)
            temporal = compute_temporal(
                state,
                max_gap_s=self._max_gap_s,
                window_s=self._cfg.timing.velocity_window_s,
            )
            machine = self._machines.setdefault(
                person.track_id, FallStateMachine(self._cfg)
            )
            decisions[person.track_id] = machine.update(timestamp_s, static, temporal)

        # Lupakan mesin milik track yang sudah dibuang tracker.
        alive = set(tracker.active_ids)
        for stale in [tid for tid in self._machines if tid not in alive]:
            del self._machines[stale]
        return decisions

    def machine(self, track_id: int) -> FallStateMachine:
        return self._machines[track_id]

    def __len__(self) -> int:
        return len(self._machines)

    def reset(self) -> None:
        self._machines.clear()
