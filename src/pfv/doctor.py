"""Pemeriksaan kelayakan sebuah sumber video untuk deteksi jatuh (``pfv doctor``).

Tujuannya: tahu apakah sebuah kamera layak SEBELUM menghabiskan waktu menyetel
ambang. Sistem ini dirancang di sekitar pose, dan pose hanya sebaik gambar yang
masuk.

**Tidak semua ambang di sini sama kuatnya, dan laporan menandainya.**

* ``fps``  -- BERDASAR PENGUKURAN. Fitur kecepatan butuh beberapa frame dalam
  setengah sampai satu detik jatuh; di bawah ~15 fps hanya ada 3-5 frame.
* sisanya -- HEURISTIK yang belum divalidasi pada kamera lain. Nilainya dipilih
  masuk akal dan dikalibrasi hanya terhadap CAUCAFall, satu-satunya sumber yang
  kita punya. Laporan menyebut ini terang-terangan, supaya peringatan tidak
  dibaca sebagai kepastian.

Bagian penilaian (:func:`assess`) murni dan menerima sampel yang sudah diukur,
sehingga bisa diuji tanpa model dan tanpa kamera.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class Severity(str, Enum):
    OK = "ok"
    WARN = "peringatan"
    FAIL = "tidak layak"


@dataclass(frozen=True)
class Sample:
    """Hasil pengukuran satu frame."""

    detected: bool
    bbox: Optional[Tuple[float, float, float, float]] = None
    full_torso: bool = False
    has_ankle: bool = False
    has_knee: bool = False


@dataclass(frozen=True)
class SourceMeta:
    width: int
    height: int
    declared_fps: Optional[float]
    measured_fps: Optional[float]
    """Laju frame terukur dari wall clock. Untuk berkas ini throughput baca, bukan
    properti rekamannya."""

    is_live: bool
    n_frames: int


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    value: str
    detail: str
    evidence: str
    """``'terukur'`` atau ``'heuristik'``."""


@dataclass(frozen=True)
class Thresholds:
    min_fps: float = 15.0
    min_person_height_frac: float = 0.15
    min_detection_rate: float = 0.80
    min_torso_rate: float = 0.80
    min_ankle_rate: float = 0.80
    min_median_aspect: float = 1.5
    min_samples: int = 30


@dataclass(frozen=True)
class Report:
    meta: SourceMeta
    checks: List[Check]
    n_samples: int
    metrics: dict = field(default_factory=dict)

    @property
    def verdict(self) -> Severity:
        if any(c.severity is Severity.FAIL for c in self.checks):
            return Severity.FAIL
        if any(c.severity is Severity.WARN for c in self.checks):
            return Severity.WARN
        return Severity.OK


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1) + 0.5)))
    return ordered[idx]


def assess(
    samples: Sequence[Sample], meta: SourceMeta, th: Thresholds = Thresholds()
) -> Report:
    """Nilai kelayakan sumber dari sampel yang sudah diukur."""
    checks: List[Check] = []
    n = len(samples)
    metrics: dict = {"n_samples": n}

    # ------------------------------------------------------------------ fps
    fps = meta.measured_fps if meta.is_live and meta.measured_fps else meta.declared_fps
    if fps is None:
        checks.append(Check(
            "frame rate", Severity.WARN, "tidak diketahui",
            "sumber tidak melaporkan fps yang masuk akal; ukur dengan sumber live lebih lama",
            "terukur",
        ))
    else:
        metrics["fps"] = fps
        sev = Severity.OK if fps >= th.min_fps else Severity.FAIL
        shown = f"{fps:.1f} fps" + (" (terukur)" if meta.is_live else " (dari metadata)")
        detail = (
            "cukup untuk fitur kecepatan" if sev is Severity.OK else
            f"jatuh berlangsung 0,5-1 detik: hanya ~{max(1, int(fps * 0.75))} frame sepanjang kejadian"
        )
        checks.append(Check("frame rate", sev, shown, detail, "terukur"))

    # ----------------------------------------------------------- resolusi
    checks.append(Check(
        "resolusi", Severity.OK, f"{meta.width}x{meta.height}",
        "informasi; pose dijalankan pada ukuran yang diskalakan", "terukur",
    ))

    if n < th.min_samples:
        checks.append(Check(
            "jumlah sampel", Severity.FAIL, f"{n} frame",
            f"kurang dari {th.min_samples}; hasil lainnya tidak dapat dipercaya", "terukur",
        ))
        return Report(meta, checks, n, metrics)

    # ---------------------------------------------------------- deteksi orang
    detected = [s for s in samples if s.detected and s.bbox is not None]
    rate = len(detected) / n
    metrics["detection_rate"] = rate
    checks.append(Check(
        "orang terdeteksi", Severity.OK if rate >= th.min_detection_rate else Severity.WARN,
        f"{rate:.0%} frame",
        "orang sering tidak terlihat; periksa sudut, cahaya, atau apakah ada orang di adegan"
        if rate < th.min_detection_rate else "memadai",
        "heuristik",
    ))
    if not detected:
        return Report(meta, checks, n, metrics)

    # ------------------------------------------------------------- ukuran orang
    heights = [(s.bbox[3] - s.bbox[1]) / meta.height for s in detected]  # type: ignore[index]
    med_h, p10_h = statistics.median(heights), _pct(heights, 0.10)
    metrics.update(person_height_frac_median=med_h, person_height_frac_p10=p10_h)
    ok_h = med_h >= th.min_person_height_frac
    checks.append(Check(
        "tinggi orang di frame", Severity.OK if ok_h else Severity.WARN,
        f"median {med_h:.0%}, p10 {p10_h:.0%} dari tinggi frame",
        "orang terlalu kecil; keypoint kaki dan lutut menjadi tidak andal"
        if not ok_h else "memadai",
        "heuristik",
    ))

    # ---------------------------------------------------------------- torso
    torso = sum(s.full_torso for s in detected) / len(detected)
    metrics["torso_rate"] = torso
    checks.append(Check(
        "torso lengkap", Severity.OK if torso >= th.min_torso_rate else Severity.WARN,
        f"{torso:.0%} frame berorang",
        "sudut dan panjang trunk sering tidak tersedia" if torso < th.min_torso_rate else "memadai",
        "heuristik",
    ))

    # -------------------------------------------------- pergelangan kaki, lutut
    ankle = sum(s.has_ankle for s in detected) / len(detected)
    knee = sum(s.has_knee for s in detected) / len(detected)
    metrics.update(ankle_rate=ankle, knee_rate=knee)
    ok_a = ankle >= th.min_ankle_rate
    checks.append(Check(
        "pergelangan kaki terdeteksi", Severity.OK if ok_a else Severity.WARN,
        f"{ankle:.0%} frame berorang (lutut {knee:.0%})",
        "fitur pinggul-kaki (terbaik secara keseluruhan) sering tak tersedia; kaki mungkin "
        "terpotong frame atau tertutup" if not ok_a else "memadai",
        "heuristik",
    ))

    # ------------------------------------------------- pemendekan / sudut kamera
    aspects = [
        (s.bbox[3] - s.bbox[1]) / (s.bbox[2] - s.bbox[0])  # type: ignore[index]
        for s in detected if (s.bbox[2] - s.bbox[0]) > 1  # type: ignore[index]
    ]
    if aspects:
        med_a = statistics.median(aspects)
        metrics["aspect_median"] = med_a
        ok_asp = med_a >= th.min_median_aspect
        checks.append(Check(
            "rasio tinggi/lebar kotak", Severity.OK if ok_asp else Severity.WARN,
            f"median {med_a:.2f}",
            "kotak orang tampak sangat memendek; kemungkinan kamera terlalu menunduk, dan "
            "perbedaan berdiri-tergeletak mengecil" if not ok_asp else
            "kotak masih cukup tinggi untuk dibedakan dari tergeletak",
            "heuristik",
        ))
    return Report(meta, checks, n, metrics)


def format_report(report: Report) -> str:
    """Teks laporan untuk terminal."""
    sym = {Severity.OK: "ok     ", Severity.WARN: "PERINGAT", Severity.FAIL: "GAGAL   "}
    lines = ["=" * 78, "PFV DOCTOR: kelayakan sumber untuk deteksi jatuh", "=" * 78]
    for c in report.checks:
        lines.append(f"[{sym[c.severity]}] {c.name:28s} {c.value}")
        if c.severity is not Severity.OK or c.name == "resolusi":
            lines.append(f"           {c.detail}  [{c.evidence}]")
    lines.append("-" * 78)
    lines.append(f"Vonis: {report.verdict.value.upper()} ({report.n_samples} frame diperiksa)")
    if any(c.evidence == "heuristik" and c.severity is not Severity.OK for c in report.checks):
        lines.append(
            "Catatan: ambang berlabel [heuristik] belum divalidasi pada kamera lain selain "
            "CAUCAFall.\nPeringatan itu petunjuk untuk diperiksa, bukan kepastian bahwa sistem gagal."
        )
    return "\n".join(lines)
