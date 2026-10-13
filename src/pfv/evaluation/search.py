"""Pencarian ambang dan aturan seleksi, tiga bagian data.

::

    latih     subjek 1-6   mencari ambang (grid search)
    validasi  subjek 7-8   memilih di antara kandidat yang lolos
    uji       subjek 9-10  dilihat SEKALI, setelah semuanya dibekukan

**Aturan seleksi ditulis di sini sebelum ada angka uji, dan tidak boleh diubah
setelah melihat hasil uji.**

1. Dari kandidat, simpan yang punya klip ADL berisi alarm palsu paling banyak
   ``MAX_TRAIN_FALSE_ALARM_CLIPS`` pada subjek LATIH.
2. Dari yang tersisa, simpan yang recall latihnya paling banyak satu klip di bawah
   yang terbaik. Banyak konfigurasi hampir bersaing pada 30 klip; membedakannya
   dengan data latih hanya menyetel derau.
3. Urutkan sisanya dengan data VALIDASI: alarm palsu paling sedikit, lalu recall
   tertinggi, lalu recall latih tertinggi, lalu urutan grid (deterministik).

Persistence TIDAK dituning; semua model memakai nilai bawaan. Itu mengurangi derajat
kebebasan, dan membuat perbandingan antar model hanya soal cara memutuskan postur.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pfv.config import AppConfig, FallConfig
from pfv.evaluation import baselines
from pfv.evaluation.cache import ClipData, build_clip_data
from pfv.evaluation.metrics import ClipOutcome, Summary, summarize
from pfv.evaluation.simulate import simulate_clip
from pfv.fall.replay import clip_subject
from pfv.fall.state_machine import PostureFn

TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6)
VALIDATION_SUBJECTS = (7, 8)
TEST_SUBJECTS = (9, 10)

MAX_TRAIN_FALSE_ALARM_CLIPS = 1
RECALL_TOLERANCE_CLIPS = 1


class HeldOutAccessError(RuntimeError):
    """Kode tuning mencoba memuat subjek uji."""


def assert_no_test_subjects(subjects: Sequence[int]) -> None:
    """Dipanggil oleh semua kode tuning. Subjek uji tidak boleh pernah dimuat di sana."""
    leaked = sorted(set(subjects) & set(TEST_SUBJECTS))
    if leaked:
        raise HeldOutAccessError(
            f"subjek uji {leaked} tidak boleh dimuat saat tuning. Subjek itu hanya "
            "dilihat sekali oleh scripts/evaluate_fall.py setelah semua ambang dibekukan."
        )


def load_clips(
    pose_dir: Path, subjects: Sequence[int], cfg: AppConfig, *, allow_test: bool = False
) -> List[ClipData]:
    """Muat dan cache fitur klip untuk subjek tertentu. Hanya itu yang dibaca."""
    if not allow_test:
        assert_no_test_subjects(subjects)
    wanted = set(subjects)
    clips = []
    for path in sorted(Path(pose_dir).glob("*.npz")):
        if clip_subject(path) in wanted:
            clips.append(build_clip_data(path, cfg))
    return clips


# ------------------------------------------------------------------ kandidat


@dataclass(frozen=True)
class Spec:
    """Satu kandidat: nama model dan parameternya. Bisa diserialisasi ke JSON."""

    model: str
    params: Tuple[Tuple[str, float], ...]

    @property
    def as_dict(self) -> Dict[str, float]:
        return dict(self.params)

    def label(self) -> str:
        return f"{self.model}(" + ", ".join(f"{k}={v:g}" for k, v in self.params) + ")"


def _spec(model: str, **params: float) -> Spec:
    return Spec(model, tuple(sorted(params.items())))


ANGLE_GRID = (30.0, 37.0, 45.0, 55.0)
SPREAD_GRID = (0.45, 0.52, 0.59, 0.66)
HIP_ANKLE_GRID = (0.20, 0.27, 0.34)
HIP_KNEE_GRID = (0.08, 0.12, 0.16)
TRUNK_LEN_GRID = (0.20, 0.24, 0.28)
ASPECT_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)
ANGLE_ONLY_GRID = (25.0, 30.0, 37.0, 45.0, 55.0, 65.0)


def candidates(model: str) -> List[Spec]:
    """Grid kandidat untuk satu model. Urutannya deterministik."""
    if model == "full":
        return [
            _spec(
                "full",
                trunk_angle_min_deg=a, keypoint_spread_y_max=s,
                hip_ankle_dy_max=ha, hip_knee_dy_max=hk, trunk_length_max=tl,
            )
            for a, s, ha, hk, tl in itertools.product(
                ANGLE_GRID, SPREAD_GRID, HIP_ANKLE_GRID, HIP_KNEE_GRID, TRUNK_LEN_GRID
            )
        ]
    if model == "orientation":
        return [
            _spec(
                "orientation",
                trunk_angle_min_deg=a, keypoint_spread_y_max=s,
                trunk_length_max=tl, min_votes=float(v),
            )
            for a, s, tl, v in itertools.product(
                ANGLE_GRID, SPREAD_GRID, TRUNK_LEN_GRID, (1, 2)
            )
        ]
    if model == "aspect":
        return [_spec("aspect", max_ratio=r) for r in ASPECT_GRID]
    if model == "angle":
        return [_spec("angle", min_deg=a) for a in ANGLE_ONLY_GRID]
    if model == "angle_or_aspect":
        return [
            _spec("angle_or_aspect", min_deg=a, max_ratio=r)
            for a, r in itertools.product(ANGLE_ONLY_GRID, ASPECT_GRID)
        ]
    raise ValueError(f"model tidak dikenal: {model!r}")


def build(spec: Spec, base: FallConfig) -> Tuple[FallConfig, Optional[PostureFn]]:
    """Ubah spesifikasi menjadi konfigurasi dan fungsi postur siap pakai."""
    p = spec.as_dict
    if spec.model == "full":
        posture = base.posture.model_copy(update=p)
        return base.model_copy(update={"posture": posture}), None
    if spec.model == "orientation":
        thresholds = {k: v for k, v in p.items() if k != "min_votes"}
        posture = base.posture.model_copy(update=thresholds)
        return (
            base.model_copy(update={"posture": posture}),
            baselines.orientation_only(int(p["min_votes"])),
        )
    if spec.model == "aspect":
        return base, baselines.aspect_baseline(p["max_ratio"])
    if spec.model == "angle":
        return base, baselines.angle_baseline(p["min_deg"])
    if spec.model == "angle_or_aspect":
        return base, baselines.angle_or_aspect_baseline(p["min_deg"], p["max_ratio"])
    raise ValueError(f"model tidak dikenal: {spec.model!r}")


def evaluate(
    spec: Spec, clips: Sequence[ClipData], base: FallConfig
) -> Tuple[Summary, List[ClipOutcome]]:
    cfg, fn = build(spec, base)
    outcomes = [simulate_clip(c, cfg, fn) for c in clips]
    return summarize(outcomes, cfg.timing.persistence_s), outcomes


# ------------------------------------------------------------------- seleksi


@dataclass
class Scored:
    spec: Spec
    train: Summary
    validation: Optional[Summary] = None


def select(
    specs: Sequence[Spec],
    train: Sequence[ClipData],
    validation: Sequence[ClipData],
    base: FallConfig,
) -> Tuple[Scored, List[Scored], int]:
    """Terapkan aturan seleksi. Kembalikan (terpilih, shortlist, jumlah_lolos_langkah_1)."""
    scored = [Scored(s, evaluate(s, train, base)[0]) for s in specs]

    eligible = [x for x in scored if x.train.false_alarm_clips.k <= MAX_TRAIN_FALSE_ALARM_CLIPS]
    if not eligible:
        # Tidak ada yang memenuhi batas: ambil yang alarm palsunya paling sedikit
        # alih-alih gagal, dan biarkan laporan menunjukkan batas itu tidak tercapai.
        floor = min(x.train.false_alarm_clips.k for x in scored)
        eligible = [x for x in scored if x.train.false_alarm_clips.k == floor]

    best = max(x.train.recall.k for x in eligible)
    shortlist = [x for x in eligible if x.train.recall.k >= best - RECALL_TOLERANCE_CLIPS]

    for x in shortlist:
        x.validation = evaluate(x.spec, validation, base)[0]

    order = {id(x): i for i, x in enumerate(shortlist)}
    ranked = sorted(
        shortlist,
        key=lambda x: (
            x.validation.false_alarm_clips.k,
            -x.validation.recall.k,
            -x.train.recall.k,
            order[id(x)],
        ),
    )
    return ranked[0], ranked, len(eligible)


# ------------------------------------------------------------------ pembekuan


def fingerprint(payload: Dict[str, object]) -> str:
    """Hash isi yang dibekukan, agar perubahan diam-diam terdeteksi."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def spec_to_json(spec: Spec) -> Dict[str, object]:
    return {"model": spec.model, "params": spec.as_dict}


def spec_from_json(data: Dict[str, object]) -> Spec:
    params = data["params"]
    assert isinstance(params, dict)
    return Spec(str(data["model"]), tuple(sorted((k, float(v)) for k, v in params.items())))
