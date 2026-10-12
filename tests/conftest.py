"""Fixture bersama.

Test Task 1 dan 2 memakai video serta struktur dataset SINTETIS yang dibuat di
``tmp_path``, bukan data nyata. Konsekuensinya suite bisa jalan di mesin mana pun
termasuk kernel Kaggle, dan tetap jalan sebelum bundel CAUCAFall 7,9 GB diunduh.

Test yang memang perlu data nyata memakai fixture ``in_repo_root`` lalu
``pytest.skip`` bila bundelnya tidak ada.

Pembantu yang dipakai lintas modul ada di ``pfv_test_helpers``, bukan di sini.
Alasannya: ``ultralytics`` memasang direktori ``tests/`` miliknya sendiri di
``site-packages``, sehingga ``from tests.conftest import ...`` bisa teresolusi ke
paket yang salah.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pytest

from pfv_test_helpers import (
    CAUCAFALL_FPS,
    CAUCAFALL_SIZE,
    write_label,
    write_png,
)

import cv2
import numpy as np


def _write_video(path: Path, width: int, height: int, n_frames: int, fps: float) -> Path:
    """Tulis video dengan pola per frame yang bisa diverifikasi.

    Setiap frame diberi struktur unik berdasar indeksnya, sehingga test bisa
    memastikan frame MANA yang di-yield -- itu yang membuat verifikasi
    ``frame_skip`` bermakna, bukan cuma menghitung jumlah frame.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV tidak bisa membuka VideoWriter untuk mp4v")
    try:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x0 = int((width - 20) * i / max(1, n_frames - 1))
            frame[:, x0 : x0 + 20] = 255
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"gagal menulis video sintetis: {path}")
    return path


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Factory video sintetis. Default 30 frame, 640x480, 30 fps."""

    counter = {"n": 0}

    def _factory(
        width: int = 640,
        height: int = 480,
        n_frames: int = 30,
        fps: float = 30.0,
        name: str = "",
    ) -> Path:
        counter["n"] += 1
        filename = name or f"synthetic_{counter['n']}.mp4"
        return _write_video(tmp_path / filename, width, height, n_frames, fps)

    return _factory


@pytest.fixture
def video_30(make_video: Callable[..., Path]) -> Path:
    """Video baseline: 30 frame, 640x480, 30 fps."""
    return make_video()


@pytest.fixture
def caucafall_like(make_video: Callable[..., Path]) -> Tuple[Path, Tuple[int, int]]:
    """Video dengan geometri CAUCAFall sebenarnya: 720x480 (rasio 3:2), 20 fps."""
    path = make_video(
        width=CAUCAFALL_SIZE[0],
        height=CAUCAFALL_SIZE[1],
        n_frames=12,
        fps=CAUCAFALL_FPS,
        name="caucafall_like.mp4",
    )
    return path, CAUCAFALL_SIZE


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Akar repo, ditemukan dari lokasi file ini.

    Dipakai agar test yang menyentuh ``config/default.yaml`` atau bundel dataset
    tidak bergantung pada direktori kerja saat pytest dijalankan.
    """
    root = Path(__file__).resolve().parent.parent
    assert (root / "pyproject.toml").exists(), f"akar repo tidak dikenali: {root}"
    return root


@pytest.fixture
def in_repo_root(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Jalankan test dari akar repo, untuk menguji resolusi path relatif."""
    monkeypatch.chdir(repo_root)
    return repo_root


@pytest.fixture
def make_caucafall(tmp_path: Path):
    """Factory struktur CAUCAFall sintetis.

    Dipakai agar test tidak bergantung pada bundel 7,9 GB. Semua pola anomali
    yang ditemukan di audit bisa direproduksi lewat parameter ``anomaly``.
    """

    def _factory(
        subject: int,
        activity: str,
        fall_frames: Optional[Tuple[int, int]] = None,
        n_frames: int = 10,
        prefix: str = "cas",
        anomaly: str = "",
        root: Optional[Path] = None,
    ) -> Path:
        base = (root or tmp_path / "CAUCAFall") / f"Subject.{subject}" / activity
        base.mkdir(parents=True, exist_ok=True)

        for i in range(n_frames):
            # Penomoran mengikuti pola nyata: prefiks + digit subjek + 5 digit.
            stem = f"{prefix}{subject}{i + 1:05d}"
            is_fall = fall_frames is not None and fall_frames[0] <= i <= fall_frames[1]
            # Kotak tegak saat berdiri, mendatar saat tergeletak.
            w, h = (0.15, 0.55) if not is_fall else (0.5, 0.18)
            png_stem, txt_stem = stem, stem

            if anomaly == "copia" and i == 3:
                png_stem = f"{stem} - copia"
            elif anomaly == "letter_label" and i == 3:
                txt_stem = f"{stem}a"
            elif anomaly == "extra_png" and i == n_frames - 1:
                write_png(base / f"{stem}.png")
                continue  # gambar tanpa label

            write_png(base / f"{png_stem}.png")
            write_label(base / f"{txt_stem}.txt", 1 if is_fall else 0, w=w, h=h)

        names = "nofall\r\nfall\r\n" if fall_frames is not None else "nofall\r\n"
        (base / "classes.txt").write_text(names, encoding="utf-8")
        return base

    return _factory
