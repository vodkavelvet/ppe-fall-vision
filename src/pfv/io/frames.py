"""Tipe frame dan penskalaan, bersama untuk semua jenis sumber.

Dipisahkan dari :mod:`pfv.io.video_source` supaya sumber sekuens gambar
(:mod:`pfv.io.image_sequence`) memakai tipe dan aturan penskalaan yang sama
persis. Kalau keduanya punya definisi sendiri, cepat atau lambat keduanya akan
menyimpang dan bug koordinat akan muncul di tempat yang sulit ditelusuri.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from pfv.config import ResizeConfig, ResizeMode


class VideoSourceError(RuntimeError):
    """Sumber frame tidak bisa dibuka atau tidak valid."""


@dataclass(frozen=True)
class FrameTransform:
    """Pemetaan dari koordinat frame yang sudah diskalakan ke frame asli.

    ``scale`` selalu satu skalar, bukan sepasang, karena penskalaan tidak pernah
    anisotropik. Itulah jaminan rasio aspek terjaga.
    """

    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.pad_x == 0 and self.pad_y == 0

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        """Titik pada frame terproses -> titik pada frame asli."""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale

    def to_source_box(
        self, box: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Kotak ``(x1, y1, x2, y2)`` pada frame terproses -> frame asli."""
        x1, y1, x2, y2 = box
        sx1, sy1 = self.to_source(x1, y1)
        sx2, sy2 = self.to_source(x2, y2)
        return sx1, sy1, sx2, sy2


@dataclass(frozen=True)
class Frame:
    """Satu frame beserta konteks yang dibutuhkan lapisan di atasnya."""

    index: int
    """Indeks pada stream ASLI. Dengan ``frame_skip=1`` urutannya 0, 2, 4, ..."""

    image: np.ndarray
    timestamp_s: float
    transform: FrameTransform
    source_size: Tuple[int, int]
    """Ukuran ``(width, height)`` frame asli sebelum penskalaan."""

    @property
    def size(self) -> Tuple[int, int]:
        """Ukuran ``(width, height)`` gambar setelah penskalaan."""
        h, w = self.image.shape[:2]
        return w, h


def compute_resize(
    width: int, height: int, cfg: ResizeConfig
) -> Tuple[int, int, FrameTransform]:
    """Hitung ukuran target dan transformasinya, tanpa menyentuh gambar.

    Dipisahkan dari eksekusinya supaya bisa diuji tanpa membuat frame, dan supaya
    ``pfv doctor`` (Task 7a) bisa melaporkan rencana penskalaan tanpa memproses
    video.
    """
    if width <= 0 or height <= 0:
        raise VideoSourceError(f"ukuran frame tidak valid: {width}x{height}")

    if cfg.mode is ResizeMode.NONE:
        return width, height, FrameTransform()

    if cfg.mode is ResizeMode.LONGEST_SIDE:
        assert cfg.longest_side is not None  # dijamin validator ResizeConfig
        scale = cfg.longest_side / max(width, height)
        if scale > 1.0 and not cfg.allow_upscale:
            scale = 1.0
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        if (new_w, new_h) == (width, height):
            return width, height, FrameTransform()
        return new_w, new_h, FrameTransform(scale=scale)

    # LETTERBOX: skala seragam lalu padding terpusat. Rasio aspek tetap terjaga;
    # yang berubah hanya kanvasnya.
    assert cfg.letterbox_size is not None  # dijamin validator ResizeConfig
    target_w, target_h = cfg.letterbox_size
    scale = min(target_w / width, target_h / height)
    if scale > 1.0 and not cfg.allow_upscale:
        scale = 1.0
    new_w = max(1, min(target_w, int(round(width * scale))))
    new_h = max(1, min(target_h, int(round(height * scale))))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return target_w, target_h, FrameTransform(scale=scale, pad_x=pad_x, pad_y=pad_y)


def apply_resize(image: np.ndarray, cfg: ResizeConfig) -> Tuple[np.ndarray, FrameTransform]:
    """Terapkan penskalaan sesuai ``cfg``, kembalikan gambar dan transformasinya."""
    height, width = image.shape[:2]
    out_w, out_h, transform = compute_resize(width, height, cfg)

    if transform.is_identity and (out_w, out_h) == (width, height):
        return image, transform

    scaled_w = max(1, int(round(width * transform.scale)))
    scaled_h = max(1, int(round(height * transform.scale)))
    if cfg.mode is ResizeMode.LETTERBOX:
        scaled_w = min(scaled_w, out_w)
        scaled_h = min(scaled_h, out_h)

    # INTER_AREA lebih baik untuk mengecilkan; INTER_LINEAR untuk memperbesar.
    interpolation = cv2.INTER_AREA if transform.scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=interpolation)

    if cfg.mode is not ResizeMode.LETTERBOX:
        return scaled, transform

    canvas_shape: Tuple[int, ...]
    if scaled.ndim == 3:
        canvas_shape = (out_h, out_w, scaled.shape[2])
    else:
        canvas_shape = (out_h, out_w)
    canvas = np.zeros(canvas_shape, dtype=scaled.dtype)
    canvas[
        transform.pad_y : transform.pad_y + scaled_h,
        transform.pad_x : transform.pad_x + scaled_w,
    ] = scaled
    return canvas, transform
