"""Identitas orang antar frame.

Lapisan ini yang mengubah deteksi per frame menjadi riwayat per orang. Tanpa
riwayat, tidak ada fitur kecepatan, tidak ada ``persistence``, dan tidak ada cara
membedakan satu kejadian jatuh dari empat puluh frame terpisah.
"""

from pfv.tracking.tracker import (
    PersonTracker,
    TrackSample,
    TrackState,
    TrackedPerson,
)

__all__ = [
    "PersonTracker",
    "TrackSample",
    "TrackState",
    "TrackedPerson",
]
