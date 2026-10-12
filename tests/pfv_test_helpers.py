"""Pembantu bersama untuk test.

**Kenapa bukan diimpor dari ``conftest``.** Paket ``ultralytics`` memasang
direktori ``tests/`` miliknya sendiri di level teratas ``site-packages``, lengkap
dengan ``__init__.py``. Akibatnya ``from tests.conftest import ...`` bisa
teresolusi ke paket mereka, bukan ke direktori test proyek ini, dan gagal dengan
``ImportError`` yang sangat membingungkan.

Modul ini diimpor sebagai ``pfv_test_helpers`` -- nama yang cukup spesifik untuk
tidak bertabrakan dengan apa pun. Karena ``tests/`` tidak punya ``__init__.py``,
pytest menambahkan direktori itu ke ``sys.path`` sehingga impornya langsung
bekerja.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pfv.perception.base import N_KEYPOINTS, Keypoint

# --------------------------------------------------------------------- CAUCAFall

#: Geometri CAUCAFall yang SEBENARNYA, hasil audit 100 folder bundel terunduh.
#: Deskripsi Mendeley menyebut 1080x960 @ 23 fps, tetapi itu kemampuan kamera;
#: file yang dikirim seragam 720x480 @ 20 fps. Lihat data/DATASETS.md.
CAUCAFALL_SIZE = (720, 480)
CAUCAFALL_FPS = 20.0


def write_png(path: Path, width: int = 720, height: int = 480) -> None:
    """Tulis PNG dengan blok terang di tengah, agar penskalaan teruji pada gambar
    yang tidak seragam."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = 200
    assert cv2.imwrite(str(path), frame), f"gagal menulis {path}"


def write_label(
    path: Path,
    cls_id: int,
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.2,
    h: float = 0.5,
) -> None:
    """Tulis satu label YOLO ternormalisasi."""
    path.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


# -------------------------------------------------------------------------- pose


def make_keypoints(conf: float = 1.0, **overrides: float) -> np.ndarray:
    """17 keypoint pada posisi yang bisa diprediksi, dengan confidence seragam.

    ``overrides`` menyetel confidence satu keypoint, mis. ``left_hip=0.1``.
    """
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    for i in range(N_KEYPOINTS):
        kp[i] = (10.0 + i, 20.0 + i, conf)
    for name, value in overrides.items():
        kp[int(Keypoint[name.upper()]), 2] = value
    return kp


def standing_keypoints(
    *, x: float = 100.0, top: float = 50.0, height: float = 160.0, conf: float = 0.9
) -> np.ndarray:
    """Keypoint orang BERDIRI: bahu di atas pinggul, trunk hampir vertikal."""
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    shoulder_y = top + height * 0.25
    hip_y = top + height * 0.55
    kp[Keypoint.LEFT_SHOULDER] = (x - 12, shoulder_y, conf)
    kp[Keypoint.RIGHT_SHOULDER] = (x + 12, shoulder_y, conf)
    kp[Keypoint.LEFT_HIP] = (x - 9, hip_y, conf)
    kp[Keypoint.RIGHT_HIP] = (x + 9, hip_y, conf)
    kp[Keypoint.NOSE] = (x, top + height * 0.08, conf)
    kp[Keypoint.LEFT_KNEE] = (x - 9, top + height * 0.78, conf)
    kp[Keypoint.RIGHT_KNEE] = (x + 9, top + height * 0.78, conf)
    return kp


def lying_keypoints(
    *, y: float = 300.0, left: float = 100.0, width: float = 160.0, conf: float = 0.9
) -> np.ndarray:
    """Keypoint orang TERGELETAK: bahu dan pinggul sejajar horizontal."""
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    shoulder_x = left + width * 0.25
    hip_x = left + width * 0.60
    kp[Keypoint.LEFT_SHOULDER] = (shoulder_x, y - 10, conf)
    kp[Keypoint.RIGHT_SHOULDER] = (shoulder_x, y + 10, conf)
    kp[Keypoint.LEFT_HIP] = (hip_x, y - 8, conf)
    kp[Keypoint.RIGHT_HIP] = (hip_x, y + 8, conf)
    kp[Keypoint.NOSE] = (left + width * 0.08, y, conf)
    kp[Keypoint.LEFT_KNEE] = (left + width * 0.85, y - 8, conf)
    kp[Keypoint.RIGHT_KNEE] = (left + width * 0.85, y + 8, conf)
    return kp
