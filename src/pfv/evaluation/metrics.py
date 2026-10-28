"""Metrik per kejadian dengan interval kepercayaan.

**Kenapa 'false alarm per jam' tidak dipakai sebagai angka utama.** Rencana awal
menargetkan 'tidak lebih dari 1 alarm palsu per jam'. Untuk CAUCAFall itu tidak
bermakna: seluruh ADL hanya 8,6 menit, jadi satu alarm palsu saja berekstrapolasi
menjadi ~7 per jam, dan nol alarm palsu hanya membuktikan 'kurang dari ~7 per jam'.
Yang dilaporkan sebagai gantinya adalah **proporsi klip ADL yang menghasilkan
alarm palsu**, dengan interval kepercayaan, yang jujur terhadap ukuran sampel.

**Recall dilaporkan dua kali.** Recall mentah, dan recall atas klip yang secara
struktural MUNGKIN tertangkap. Periode tergeletak di CAUCAFall minimum 2,15 detik,
jadi dengan persistence 3 detik beberapa klip tidak mungkin tertangkap oleh
algoritma apa pun. Mencampur keduanya menyalahkan algoritma atas keputusan
penyuntingan dataset.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Waktu minimum antara awal tergeletak pada label sampai alert mungkin terjadi,
#: di luar persistence itu sendiri: postur harus terbaca dulu, dan FALLING dimulai
#: dari frame pertama yang terbaca. Setengah detik adalah batas bawah yang longgar.
DETECTION_MARGIN_S = 0.5


@dataclass(frozen=True)
class ClipOutcome:
    name: str
    subject: int
    activity: str
    is_fall: bool
    is_camera_axis: bool
    lying_s: float
    n_alerts: int
    first_alert_s: Optional[float]
    latency_s: Optional[float]
    """Alert pertama dikurangi awal tergeletak menurut label. Negatif berarti alert
    mendahului label."""

    @property
    def alerted(self) -> bool:
        return self.n_alerts > 0

    def achievable(self, persistence_s: float) -> bool:
        """Apakah klip ini cukup panjang untuk SECARA TEORI menghasilkan alert."""
        return self.lying_s >= persistence_s + DETECTION_MARGIN_S


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Interval Wilson 95 persen untuk proporsi ``k/n``.

    Dipilih daripada interval normal karena tetap masuk akal untuk n kecil dan
    proporsi di dekat 0 atau 1 -- persis kondisi evaluasi ini. ``(0, 0)`` untuk n=0.
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Dijepit ke p: secara matematis estimasi titik selalu di dalam interval, tetapi
    # galat floating point bisa menghasilkan batas bawah 0.99999 untuk k=n=100,
    # sehingga interval tampak tidak mencakup estimasinya sendiri.
    return min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


@dataclass(frozen=True)
class Proportion:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def interval(self) -> Tuple[float, float]:
        return wilson_interval(self.k, self.n)

    def format(self) -> str:
        if self.n == 0:
            return "-"
        lo, hi = self.interval
        return f"{self.k}/{self.n} = {self.value:.0%} [{lo:.0%}-{hi:.0%}]"


@dataclass(frozen=True)
class Summary:
    recall: Proportion
    """Klip jatuh yang menghasilkan alert."""

    achievable_recall: Proportion
    """Hanya klip yang cukup panjang untuk SECARA TEORI tertangkap."""

    false_alarm_clips: Proportion
    """Klip ADL yang menghasilkan alert."""

    multi_alert_clips: Proportion
    """Klip jatuh dengan lebih dari satu alert (alarm ganda)."""

    camera_axis_recall: Proportion
    lateral_recall: Proportion

    latency_median_s: Optional[float]
    latency_p90_s: Optional[float]
    latency_max_s: Optional[float]
    n_latencies: int


def summarize(outcomes: Sequence[ClipOutcome], persistence_s: float) -> Summary:
    falls = [o for o in outcomes if o.is_fall]
    adl = [o for o in outcomes if not o.is_fall]
    achievable = [o for o in falls if o.achievable(persistence_s)]
    axis = [o for o in falls if o.is_camera_axis]
    lateral = [o for o in falls if not o.is_camera_axis]

    latencies: List[float] = sorted(
        o.latency_s for o in falls if o.alerted and o.latency_s is not None
    )

    def pct(q: float) -> Optional[float]:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(math.ceil(q * len(latencies))) - 1)
        return latencies[max(0, idx)]

    return Summary(
        recall=Proportion(sum(o.alerted for o in falls), len(falls)),
        achievable_recall=Proportion(sum(o.alerted for o in achievable), len(achievable)),
        false_alarm_clips=Proportion(sum(o.alerted for o in adl), len(adl)),
        multi_alert_clips=Proportion(sum(o.n_alerts > 1 for o in falls), len(falls)),
        camera_axis_recall=Proportion(sum(o.alerted for o in axis), len(axis)),
        lateral_recall=Proportion(sum(o.alerted for o in lateral), len(lateral)),
        latency_median_s=statistics.median(latencies) if latencies else None,
        latency_p90_s=pct(0.9),
        latency_max_s=max(latencies) if latencies else None,
        n_latencies=len(latencies),
    )
