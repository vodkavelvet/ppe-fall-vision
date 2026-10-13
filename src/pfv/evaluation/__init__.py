"""Harness evaluasi deteksi jatuh (Task 7b).

Tiga prinsip yang membentuk paket ini, semuanya agar angka yang dilaporkan bisa
dipertanggungjawabkan di paper:

* **Per kejadian, bukan per frame.** Satu jatuh yang terdeteksi 40 frame berturut-
  turut adalah satu keberhasilan, bukan empat puluh.
* **Split per subjek, tiga bagian.** Latih (1-6) untuk mencari ambang, validasi
  (7-8) untuk memilih di antara kandidat, uji (9-10) dilihat SEKALI.
* **Interval kepercayaan selalu menyertai.** Dengan 20 klip uji, satu klip
  menggeser recall 5 poin. Angka tanpa interval menyiratkan presisi yang tidak ada.
"""

from pfv.evaluation.cache import ClipData, TrackSeq, build_clip_data
from pfv.evaluation.metrics import ClipOutcome, Summary, summarize, wilson_interval
from pfv.evaluation.simulate import simulate_clip

__all__ = [
    "ClipData",
    "ClipOutcome",
    "Summary",
    "TrackSeq",
    "build_clip_data",
    "simulate_clip",
    "summarize",
    "wilson_interval",
]
