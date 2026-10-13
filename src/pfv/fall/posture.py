"""Dari fitur satu frame menjadi keputusan postur: tergeletak, tegak, atau tak tahu.

Fungsi murni. Tiga status, bukan dua, dan itu disengaja: bukti yang tipis tidak
boleh menghasilkan tuduhan, tetapi juga tidak boleh menghasilkan 'aman'. Orang
yang pinggul dan kakinya tertutup meja bukan orang yang sedang berdiri.

**Dua kelompok indikator yang HARUS sama-sama setuju.**

* *Orientasi tubuh*: sudut trunk, sebaran vertikal keypoint, panjang trunk.
  Menjawab 'apakah tubuh ini mendatar atau ringkas?'
* *Hubungan tinggi anggota badan*: selisih tinggi pinggul terhadap pergelangan
  kaki dan terhadap lutut. Menjawab 'apakah pinggul berada di ketinggian kaki?'

Desain awal menghitung kelima indikator sebagai lima suara setara dan meminta dua.
Diputar ulang pada klip nyata (subjek latih saja), itu gagal dengan dua cara yang
berbeda, dan keduanya berakar pada hal yang sama -- suara yang tidak independen:

* Orang yang DUDUK di lantai memenuhi kedua indikator tinggi anggota badan
  sekaligus. Itu dua suara untuk SATU fakta (pinggul dekat tanah).
* Orang yang MEMBUNGKUK mengambil objek memenuhi ketiga indikator orientasi
  (trunk miring, ringkas, memendek) tanpa pernah tergeletak. Pinggulnya tetap
  tinggi di atas kaki.

Mensyaratkan KEDUA kelompok menolak keduanya dengan alasan fisik, bukan dengan
ambang yang lebih ketat: duduk gagal di orientasi (trunk tetap tegak), membungkuk
gagal di tinggi anggota badan (pinggul tetap jauh di atas kaki). Tergeletak, ke arah
mana pun, memenuhi keduanya.

**Kenapa tetap beberapa indikator di dalam kelompok.** Analisis per subjek
menunjukkan tidak ada satu fitur yang bekerja untuk semua arah jatuh (lihat
``data/DATASETS.md``): sudut trunk andal untuk jatuh menyamping tetapi hanya benar
pada 3 dari 10 subjek untuk jatuh searah kamera, sedangkan panjang trunk
kebalikannya. Di dalam kelompok orientasi, satu suara dari tiga sudah cukup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pfv.config import PostureConfig
from pfv.fall.features import StaticFeatures

ORIENTATION_INDICATORS = ("trunk_angle", "keypoint_spread", "trunk_length")
LIMB_INDICATORS = ("hip_ankle", "hip_knee")


class Posture(str, Enum):
    LYING = "lying"
    UPRIGHT = "upright"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PostureResult:
    posture: Posture
    votes: Dict[str, Optional[bool]]
    """Per indikator: ``True`` memilih tergeletak, ``False`` tidak, ``None`` tak tersedia."""

    n_available: int
    n_lying: int
    """Total suara tergeletak dari semua indikator (untuk diagnosis)."""

    n_orientation_votes: int = 0
    n_limb_votes: int = 0


def _vote(value: Optional[float], threshold: float, lying_if_above: bool) -> Optional[bool]:
    if value is None:
        return None
    return value >= threshold if lying_if_above else value <= threshold


def _count(votes: Dict[str, Optional[bool]], names) -> "tuple[int, int]":
    """Kembalikan ``(tersedia, memilih_tergeletak)`` untuk sebuah kelompok."""
    group = [votes[n] for n in names if votes[n] is not None]
    return len(group), sum(1 for v in group if v)


def classify_posture(features: StaticFeatures, cfg: PostureConfig) -> PostureResult:
    """Putuskan postur dari fitur satu frame.

    * ``LYING``   -- kelompok orientasi DAN kelompok anggota badan sama-sama
      mencapai jumlah suara minimumnya.
    * ``UNKNOWN`` -- salah satu kelompok tidak punya satu pun indikator tersedia
      (tidak bisa dinilai), atau total indikator tersedia di bawah
      ``min_available_indicators``. Dan bukan LYING.
    * ``UPRIGHT`` -- sisanya: ada cukup bukti dan bukti itu tidak menyatakan
      tergeletak.
    """
    votes: Dict[str, Optional[bool]] = {
        "trunk_angle": _vote(features.trunk_angle_deg, cfg.trunk_angle_min_deg, True),
        "keypoint_spread": _vote(
            features.keypoint_spread_y_norm, cfg.keypoint_spread_y_max, False
        ),
        "trunk_length": _vote(features.trunk_length_norm, cfg.trunk_length_max, False),
        "hip_ankle": _vote(features.hip_ankle_dy_norm, cfg.hip_ankle_dy_max, False),
        "hip_knee": _vote(features.hip_knee_dy_norm, cfg.hip_knee_dy_max, False),
    }

    o_avail, o_votes = _count(votes, ORIENTATION_INDICATORS)
    l_avail, l_votes = _count(votes, LIMB_INDICATORS)
    n_available = o_avail + l_avail
    n_lying = o_votes + l_votes

    def result(posture: Posture) -> PostureResult:
        return PostureResult(posture, votes, n_available, n_lying, o_votes, l_votes)

    if o_votes >= cfg.min_orientation_votes and l_votes >= cfg.min_limb_votes:
        return result(Posture.LYING)

    # Salah satu kelompok wajib tidak bisa dinilai sama sekali: 'tidak terbukti
    # tergeletak' berbeda dari 'terbukti tidak tergeletak'.
    if o_avail == 0 or l_avail == 0 or n_available < cfg.min_available_indicators:
        return result(Posture.UNKNOWN)

    return result(Posture.UPRIGHT)
