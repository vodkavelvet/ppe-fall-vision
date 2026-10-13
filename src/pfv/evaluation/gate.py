"""Gerbang keputusan dan kunci satu-kali untuk evaluasi subjek uji."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from pfv.evaluation.metrics import Summary


class TestRunLockedError(RuntimeError):
    """Evaluasi uji sudah pernah dijalankan untuk konfigurasi beku yang sama."""

    # Namanya berawalan 'Test', jadi pytest mencoba mengumpulkannya sebagai kelas test.
    __test__ = False


@dataclass(frozen=True)
class Gate:
    achievable_recall_min: float
    false_alarm_clip_rate_max: float
    latency_median_max_s: float


@dataclass(frozen=True)
class CriterionResult:
    name: str
    required: str
    observed: str
    passed: bool


@dataclass(frozen=True)
class Verdict:
    passed: bool
    criteria: List[CriterionResult]

    @property
    def label(self) -> str:
        return "LOLOS" if self.passed else "TIDAK LOLOS"


def load_gate(path: Path) -> Gate:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return Gate(
            achievable_recall_min=float(data["achievable_recall_min"]),
            false_alarm_clip_rate_max=float(data["false_alarm_clip_rate_max"]),
            latency_median_max_s=float(data["latency_median_max_s"]),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: kriteria gerbang hilang: {exc}") from exc


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def judge(summary: Summary, gate: Gate) -> Verdict:
    """Terapkan gerbang pada estimasi titik. Interval kepercayaan dilaporkan terpisah."""
    ar = summary.achievable_recall
    fa = summary.false_alarm_clips
    lat = summary.latency_median_s

    results = [
        CriterionResult(
            "recall (klip yang mungkin tertangkap)",
            f">= {gate.achievable_recall_min:.0%}",
            ar.format(),
            ar.n > 0 and ar.value >= gate.achievable_recall_min,
        ),
        CriterionResult(
            "klip ADL dengan alarm palsu",
            f"<= {gate.false_alarm_clip_rate_max:.0%}",
            fa.format(),
            fa.n > 0 and fa.value <= gate.false_alarm_clip_rate_max,
        ),
        CriterionResult(
            "median latensi dari awal tergeletak",
            f"<= {gate.latency_median_max_s:g} s",
            "-" if lat is None else f"{lat:.1f} s",
            lat is not None and lat <= gate.latency_median_max_s,
        ),
    ]
    return Verdict(all(r.passed for r in results), results)


# ----------------------------------------------------------------- kunci uji


def read_runs(log_path: Path) -> List[Dict[str, object]]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_lock(log_path: Path, fingerprint: str, *, allow_rerun: bool) -> Optional[Dict[str, object]]:
    """Tolak bila konfigurasi beku ini sudah pernah dievaluasi pada subjek uji.

    Mengembalikan catatan sebelumnya bila ``allow_rerun`` mengizinkannya.
    """
    previous = [r for r in read_runs(log_path) if r.get("frozen_fingerprint") == fingerprint]
    if not previous:
        return None
    last = previous[-1]
    if not allow_rerun:
        raise TestRunLockedError(
            f"Subjek uji sudah dievaluasi untuk konfigurasi beku {fingerprint} pada "
            f"{last.get('timestamp')}. Melihatnya lagi lalu menyetel ulang ambang "
            "membuat angka uji tidak lagi tak terlihat. Bila memang perlu "
            "(misalnya skrip evaluasi rusak), pakai --allow-rerun; kejadian itu "
            "akan tercatat sebagai ulangan di eval/test_runs.jsonl."
        )
    return last


def record_run(
    log_path: Path, *, fingerprint: str, gate_hash: str, rerun: bool, verdict: Optional[str]
) -> Dict[str, object]:
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_fingerprint": fingerprint,
        "gate_hash": gate_hash,
        "rerun": rerun,
        "verdict": verdict,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
