"""Baseline postur untuk eksperimen ablasi.

Semuanya memenuhi tanda tangan ``PostureFn`` dan dijalankan melalui state machine,
persistence, dan cooldown YANG SAMA dengan model penuh. Itu yang membuat
perbandingannya adil: yang berbeda hanya cara memutuskan 'tergeletak' dari satu frame.

Baseline dipilih karena mewakili cara yang umum di literatur deteksi jatuh berbasis
pose, bukan karena mudah dikalahkan:

* ``aspect``           rasio aspek kotak saja
* ``angle``            sudut trunk saja
* ``angle_or_aspect``  gabungan dua fitur paling umum; tergeletak bila salah satu setuju
* ``orientation``      hanya kelompok orientasi (tanpa kelompok anggota badan);
                       ablasi yang mengukur sumbangan kelompok anggota badan

Masing-masing dituning pada subjek latih dengan prosedur yang sama dengan model
penuh. Baseline yang disetel asal-asalan akan membuat klaim 'model kami lebih baik'
tidak bermakna.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from pfv.config import PostureConfig
from pfv.fall.features import StaticFeatures
from pfv.fall.posture import Posture, PostureResult, classify_posture
from pfv.fall.state_machine import PostureFn


def _result(posture: Posture, votes: Dict[str, Optional[bool]]) -> PostureResult:
    avail = [v for v in votes.values() if v is not None]
    return PostureResult(posture, votes, len(avail), sum(1 for v in avail if v))


def aspect_baseline(max_ratio: float) -> PostureFn:
    """Tergeletak bila rasio aspek kotak <= ``max_ratio``."""

    def fn(f: StaticFeatures, _cfg: PostureConfig) -> PostureResult:
        if f.bbox_aspect_ratio is None:
            return _result(Posture.UNKNOWN, {"aspect": None})
        lying = f.bbox_aspect_ratio <= max_ratio
        return _result(Posture.LYING if lying else Posture.UPRIGHT, {"aspect": lying})

    return fn


def angle_baseline(min_deg: float) -> PostureFn:
    """Tergeletak bila sudut trunk >= ``min_deg``."""

    def fn(f: StaticFeatures, _cfg: PostureConfig) -> PostureResult:
        if f.trunk_angle_deg is None:
            return _result(Posture.UNKNOWN, {"angle": None})
        lying = f.trunk_angle_deg >= min_deg
        return _result(Posture.LYING if lying else Posture.UPRIGHT, {"angle": lying})

    return fn


def angle_or_aspect_baseline(min_deg: float, max_ratio: float) -> PostureFn:
    """Tergeletak bila sudut trunk ATAU rasio aspek setuju."""

    def fn(f: StaticFeatures, _cfg: PostureConfig) -> PostureResult:
        a = None if f.trunk_angle_deg is None else f.trunk_angle_deg >= min_deg
        r = None if f.bbox_aspect_ratio is None else f.bbox_aspect_ratio <= max_ratio
        votes = {"angle": a, "aspect": r}
        if a is None and r is None:
            return _result(Posture.UNKNOWN, votes)
        lying = bool(a) or bool(r)
        return _result(Posture.LYING if lying else Posture.UPRIGHT, votes)

    return fn


def orientation_only(min_votes: int) -> PostureFn:
    """Hanya kelompok orientasi, tanpa kelompok anggota badan.

    Memakai ambang dari :class:`PostureConfig` yang diberikan. Ablasi ini mengukur
    apa yang hilang bila kelompok anggota badan dibuang -- khususnya apakah duduk
    dan membungkuk kembali menghasilkan alarm palsu.
    """

    def fn(f: StaticFeatures, cfg: PostureConfig) -> PostureResult:
        full = classify_posture(f, cfg)
        o = {k: full.votes[k] for k in ("trunk_angle", "keypoint_spread", "trunk_length")}
        avail = [v for v in o.values() if v is not None]
        if not avail:
            return _result(Posture.UNKNOWN, o)
        lying = sum(1 for v in avail if v) >= min_votes
        return _result(Posture.LYING if lying else Posture.UPRIGHT, o)

    return fn


#: Nama yang dipakai di laporan, dalam urutan tampil.
MODEL_ORDER = ("aspect", "angle", "angle_or_aspect", "orientation", "full")

MODEL_LABELS: Dict[str, str] = {
    "aspect": "rasio aspek saja",
    "angle": "sudut trunk saja",
    "angle_or_aspect": "sudut ATAU rasio aspek",
    "orientation": "orientasi saja (tanpa anggota badan)",
    "full": "model penuh (dua kelompok)",
}

BaselineFactory = Callable[..., PostureFn]
