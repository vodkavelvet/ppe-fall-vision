"""ppe-fall-vision: kepatuhan PPE dan deteksi jatuh dari video.

Lapisan paket:
    io/          -- masukan/keluaran video, tanpa pengetahuan tentang model
    perception/  -- pose & PPE detector di belakang Protocol (Task 3, 8)
    tracking/    -- identitas orang antar frame (Task 4)
    fusion/      -- asosiasi part PPE ke track_id (Task 8)
    fall/        -- fitur & state machine, FUNGSI MURNI tanpa OpenCV (Task 5, 6)
    rules/       -- debounce, cooldown, lock pelanggaran (Task 9)
    alert/       -- sink notifikasi (Task 10)
    storage/     -- event store & klip bukti (Task 9)

Aturan dependensi: paket ini TIDAK BOLEH mengimpor `train/`, dan `train/` tidak
boleh mengimpor GUI atau alert. Yang dibagi hanya definisi kelas, preprocessing,
dan fungsi fitur.
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"

#: Batas bawah versi Python. Bukan 3.14 karena image Kaggle memakai Python
#: sendiri (saat ini 3.11) dan kode yang sama harus jalan di sana.
MIN_PYTHON = (3, 10)


def _check_python_version() -> None:
    """Gagal cepat dengan pesan yang jelas bila interpreternya terlalu tua.

    macOS memasang Python 3.9 di ``/usr/bin/python3`` sebagai bagian dari OS, dan
    itu yang muncul kalau ``python3`` dipanggil tanpa mengaktifkan ``.venv``.
    Tanpa penjaga ini, gejalanya berupa ``ImportError`` dari dependensi yang
    menyesatkan; dengan penjaga ini penyebabnya langsung terbaca.
    """
    if sys.version_info < MIN_PYTHON:
        current = ".".join(str(n) for n in sys.version_info[:3])
        required = ".".join(str(n) for n in MIN_PYTHON)
        raise RuntimeError(
            f"ppe-fall-vision butuh Python >= {required}, tetapi dijalankan dengan "
            f"{current} dari {sys.executable}.\n"
            "Kemungkinan besar ini Python bawaan macOS. Pakai interpreter proyek:\n"
            "    .venv/bin/python -m pytest\n"
            "    .venv/bin/pfv play --source 0\n"
            "atau aktifkan dulu venv-nya: source .venv/bin/activate"
        )


_check_python_version()
