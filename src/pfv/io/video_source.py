"""Abstraksi sumber video.

Satu antarmuka untuk file, webcam, dan RTSP, supaya sisa pipeline tidak pernah
perlu tahu dari mana frame datang. Ini yang memungkinkan Task 19 menambah
multi-stream RTSP tanpa menyentuh kode deteksi.

Dua hal di modul ini yang terlihat sepele tapi bukan kosmetik:

1. **Penskalaan selalu menjaga rasio aspek.** ``bbox_aspect_ratio`` adalah fitur
   deteksi jatuh (Task 5). Kalau frame 720x480 milik CAUCAFall ditarik menjadi
   640x640, orang berdiri jadi terlihat gepeng dan fiturnya rusak secara diam-diam.
   Audit dataset menunjukkan pemisahan berdiri versus berbaring hanya 4,2 kali,
   jadi distorsi sebesar 1,33 kali dari penarikan 3:2 ke 1:1 bukan hal sepele.
2. **Setiap frame membawa :class:`FrameTransform`.** Deteksi berjalan di gambar
   yang sudah diskalakan, tetapi event, crop bukti, dan klip harus merujuk ke
   koordinat frame ASLI. Menyimpan transformasinya bersama frame mencegah
   kekeliruan koordinat yang sulit dilacak nanti.

Tipe frame dan fungsi penskalaannya sendiri ada di :mod:`pfv.io.frames`, supaya
:mod:`pfv.io.image_sequence` memakai definisi yang sama persis.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2

from pfv.config import ResizeConfig
from pfv.io.frames import (
    Frame,
    FrameTransform,
    VideoSourceError,
    apply_resize,
    compute_resize,
)

logger = logging.getLogger(__name__)

_NETWORK_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")

# Direkspor ulang agar `from pfv.io.video_source import Frame, compute_resize, ...`
# yang sudah dipakai di test dan modul lain tetap bekerja setelah pemindahan.
__all__ = [
    "Frame",
    "FrameTransform",
    "SourceKind",
    "VideoSource",
    "VideoSourceError",
    "apply_resize",
    "compute_resize",
    "parse_source",
]


class SourceKind(str, Enum):
    FILE = "file"
    WEBCAM = "webcam"
    NETWORK = "network"

    @property
    def is_live(self) -> bool:
        """Sumber live tidak punya jumlah frame tetap dan tidak bisa di-seek."""
        return self is not SourceKind.FILE


def parse_source(spec: Union[str, int, Path]) -> Tuple[SourceKind, Union[int, str]]:
    """Tentukan jenis sumber dan nilai yang bisa diterima OpenCV.

    Aturannya sengaja eksplisit: ``"0"`` adalah webcam, bukan nama file bernama
    nol, dan path file diverifikasi keberadaannya di sini supaya pesan errornya
    jelas alih-alih ``VideoCapture`` gagal tanpa keterangan.
    """
    if isinstance(spec, int):
        return SourceKind.WEBCAM, spec

    if isinstance(spec, Path):
        text = str(spec)
    else:
        text = str(spec).strip()

    if not text:
        raise VideoSourceError("source kosong")

    lowered = text.lower()
    if lowered.startswith(_NETWORK_PREFIXES):
        return SourceKind.NETWORK, text

    if text.isdigit():
        return SourceKind.WEBCAM, int(text)

    path = Path(text).expanduser()
    if not path.exists():
        raise VideoSourceError(f"file video tidak ditemukan: {path}")
    if path.is_dir():
        raise VideoSourceError(f"source menunjuk ke direktori, bukan file video: {path}")
    return SourceKind.FILE, str(path)


class VideoSource:
    """Iterator frame atas file, webcam, atau stream jaringan.

    Contoh::

        with VideoSource("clip.mp4", resize=cfg) as src:
            for frame in src:
                ...
            print(src.effective_fps)
    """

    def __init__(
        self,
        spec: Union[str, int, Path],
        *,
        frame_skip: int = 0,
        resize: Optional[ResizeConfig] = None,
        loop: bool = False,
        fall_mode: bool = False,
        min_fall_fps: float = 15.0,
    ) -> None:
        if frame_skip < 0:
            raise VideoSourceError("frame_skip tidak boleh negatif")

        self.kind, self._target = parse_source(spec)
        self.spec = spec
        self.frame_skip = frame_skip
        self.resize_config = resize if resize is not None else ResizeConfig()
        self.loop = loop and self.kind is SourceKind.FILE
        self.fall_mode = fall_mode
        self.min_fall_fps = min_fall_fps

        self._capture: Optional["cv2.VideoCapture"] = None
        self._frames_yielded = 0
        self._elapsed_s = 0.0
        self._warnings: list = []

        self._open()
        self._check_temporal_resolution()

    # ---------------------------------------------------------------- lifecycle

    def _open(self) -> None:
        capture = cv2.VideoCapture(self._target)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"gagal membuka sumber video: {self._target!r} (jenis: {self.kind.value})"
            )
        self._capture = capture

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        # Webcam dan beberapa stream RTSP melaporkan 0 atau angka tak masuk akal.
        self.declared_fps: Optional[float] = raw_fps if 0.0 < raw_fps < 1000.0 else None

        raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count: Optional[int] = raw_count if raw_count > 0 else None

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.source_size: Optional[Tuple[int, int]] = (
            (width, height) if width > 0 and height > 0 else None
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # ------------------------------------------------------------- diagnostics

    @property
    def step(self) -> int:
        """Jarak antar frame yang diproses. ``frame_skip=0`` berarti step 1."""
        return self.frame_skip + 1

    @property
    def sampled_fps(self) -> Optional[float]:
        """Resolusi temporal SETELAH ``frame_skip``.

        Ini angka yang menentukan apakah fitur kecepatan pada Task 5 bermakna,
        bukan fps yang dilaporkan sumber.
        """
        if self.declared_fps is None:
            return None
        return self.declared_fps / self.step

    @property
    def effective_fps(self) -> Optional[float]:
        """Laju frame terukur dari wall clock, dihitung dari frame yang di-yield.

        Untuk sumber live ini angka yang jujur: webcam dan RTSP sering jauh di
        bawah fps yang diklaim. Untuk file, angka ini mengukur throughput baca,
        bukan properti rekamannya -- untuk itu pakai :attr:`declared_fps`.
        """
        if self._frames_yielded == 0 or self._elapsed_s <= 0.0:
            return None
        return self._frames_yielded / self._elapsed_s

    @property
    def frames_yielded(self) -> int:
        return self._frames_yielded

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    @property
    def warnings(self) -> Tuple[str, ...]:
        """Peringatan yang terkumpul, juga dikirim ke logger."""
        return tuple(self._warnings)

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        logger.warning(message)

    def _check_temporal_resolution(self) -> None:
        """Peringatkan bila setelan sekarang merusak fitur deteksi jatuh.

        Sengaja dijalankan di konstruktor, bukan saat iterasi, supaya pengguna
        tahu sebelum menunggu video selesai diproses.
        """
        if not self.fall_mode:
            return

        if self.frame_skip > 0:
            self._warn(
                f"frame_skip={self.frame_skip} aktif bersamaan dengan mode fall. "
                "Melewati frame merusak hip_vertical_velocity karena kejadian jatuh "
                "hanya berlangsung 0,5-1 detik. Untuk menghemat komputasi, "
                "turunkan resolusi lewat video.resize, bukan frame_skip."
            )

        sampled = self.sampled_fps
        if sampled is None:
            self._warn(
                "sumber tidak melaporkan fps yang masuk akal, jadi resolusi temporal "
                "tidak bisa diperiksa di muka. Jalankan `pfv doctor` untuk mengukur "
                "fps efektif yang sebenarnya."
            )
        elif sampled < self.min_fall_fps:
            self._warn(
                f"fps efektif {sampled:.1f} di bawah minimum {self.min_fall_fps:.1f} "
                "untuk deteksi jatuh. Kejadian jatuh berlangsung 0,5-1 detik, jadi pada "
                f"{sampled:.1f} fps hanya ada sekitar "
                f"{max(1, int(sampled * 0.75))} frame sepanjang kejadian dan fitur "
                "kecepatan menjadi tidak bermakna."
            )

    # ---------------------------------------------------------------- iteration

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is None:
            raise VideoSourceError("sumber video sudah ditutup")

        capture = self._capture
        step = self.step
        started = time.perf_counter()
        raw_index = -1
        live_start: Optional[float] = None

        while True:
            ok, image = capture.read()
            if not ok:
                if self.loop and raw_index >= 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    raw_index = -1
                    continue
                break

            raw_index += 1
            if raw_index % step != 0:
                continue

            if self.source_size is None:
                h, w = image.shape[:2]
                self.source_size = (w, h)

            processed, transform = apply_resize(image, self.resize_config)

            now = time.perf_counter()
            if self.kind is SourceKind.FILE and self.declared_fps:
                timestamp_s = raw_index / self.declared_fps
            else:
                if live_start is None:
                    live_start = now
                timestamp_s = now - live_start

            self._frames_yielded += 1
            self._elapsed_s = now - started

            yield Frame(
                index=raw_index,
                image=processed,
                timestamp_s=timestamp_s,
                transform=transform,
                source_size=self.source_size,
            )

        self._elapsed_s = time.perf_counter() - started

    def describe(self) -> str:
        """Ringkasan satu baris untuk log dan CLI."""
        size = f"{self.source_size[0]}x{self.source_size[1]}" if self.source_size else "?"
        fps = f"{self.declared_fps:.2f}" if self.declared_fps else "tidak dilaporkan"
        count = str(self.frame_count) if self.frame_count else "tidak diketahui"
        return (
            f"{self.kind.value} {self._target!r} | {size} | fps sumber: {fps} | "
            f"jumlah frame: {count} | step: {self.step} | resize: {self.resize_config.mode.value}"
        )
