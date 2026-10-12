"""Sumber frame dari sekuens gambar.

Antarmukanya sama dengan :class:`pfv.io.video_source.VideoSource`, sehingga
pipeline tidak perlu tahu bedanya.

**Kenapa ini ada, bukan mentranskode ke video.** Audit CAUCAFall menunjukkan
``.avi`` bawaannya hanya 1,7 persen dari ukuran PNG -- 6,6 KB versus 400 KB per
frame -- dan tiga di antaranya terpotong. Estimasi pose adalah satu-satunya
masukan deteksi jatuh, jadi menurunkan kualitas gambar di pintu masuk akan
membatasi semua yang di atasnya. Membaca PNG lossless langsung menghindari
transkode, menghindari duplikasi 7,6 GB, dan menangani anomali penamaan secara
alami karena daftar frame-nya eksplisit.

UR Fall juga didistribusikan sebagai sekuens PNG, jadi modul ini melayani
keduanya.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2

from pfv.config import ResizeConfig
from pfv.io.frames import Frame, VideoSourceError, apply_resize

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ImageSequenceSource:
    """Iterator frame atas daftar gambar yang urutannya sudah ditentukan.

    Urutan diambil apa adanya dari ``paths``; modul ini tidak menebak urutan dari
    pola nama file. Itu disengaja, karena penamaan CAUCAFall tidak seragam dan
    penentuan urutan adalah tanggung jawab :mod:`pfv.data.caucafall`.

    Contoh::

        with ImageSequenceSource(clip.image_paths, fps=20.0) as src:
            for frame in src:
                ...
    """

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        fps: float,
        frame_skip: int = 0,
        resize: Optional[ResizeConfig] = None,
        loop: bool = False,
        fall_mode: bool = False,
        min_fall_fps: float = 15.0,
    ) -> None:
        if frame_skip < 0:
            raise VideoSourceError("frame_skip tidak boleh negatif")
        if fps <= 0:
            raise VideoSourceError(f"fps harus positif, diberikan {fps}")

        self.paths: List[Path] = [Path(p) for p in paths]
        if not self.paths:
            raise VideoSourceError("sekuens gambar kosong")

        missing = [p for p in self.paths[:5] if not p.exists()]
        if missing:
            raise VideoSourceError(f"gambar tidak ditemukan: {missing[0]}")

        self.declared_fps: Optional[float] = float(fps)
        self.frame_skip = frame_skip
        self.resize_config = resize if resize is not None else ResizeConfig()
        self.loop = loop
        self.fall_mode = fall_mode
        self.min_fall_fps = min_fall_fps

        self.frame_count: Optional[int] = len(self.paths)
        self.source_size: Optional[Tuple[int, int]] = None
        self._closed = False
        self._frames_yielded = 0
        self._elapsed_s = 0.0
        self._warnings: List[str] = []

        self._check_temporal_resolution()

    # ---------------------------------------------------------------- lifecycle

    def release(self) -> None:
        """Tidak ada handle OS yang dipegang; disediakan agar antarmukanya sama."""
        self._closed = True

    def __enter__(self) -> "ImageSequenceSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # ------------------------------------------------------------- diagnostics

    @property
    def is_live(self) -> bool:
        return False

    @property
    def step(self) -> int:
        return self.frame_skip + 1

    @property
    def sampled_fps(self) -> Optional[float]:
        """Resolusi temporal SETELAH ``frame_skip``."""
        if self.declared_fps is None:
            return None
        return self.declared_fps / self.step

    @property
    def effective_fps(self) -> Optional[float]:
        """Laju terukur dari wall clock. Untuk sekuens gambar ini throughput baca."""
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
        return tuple(self._warnings)

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        logger.warning(message)

    def _check_temporal_resolution(self) -> None:
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
        if sampled is not None and sampled < self.min_fall_fps:
            self._warn(
                f"fps efektif {sampled:.1f} di bawah minimum {self.min_fall_fps:.1f} "
                "untuk deteksi jatuh. Kejadian jatuh berlangsung 0,5-1 detik, jadi pada "
                f"{sampled:.1f} fps hanya ada sekitar "
                f"{max(1, int(sampled * 0.75))} frame sepanjang kejadian dan fitur "
                "kecepatan menjadi tidak bermakna."
            )

    # ---------------------------------------------------------------- iteration

    def __iter__(self) -> Iterator[Frame]:
        if self._closed:
            raise VideoSourceError("sumber video sudah ditutup")

        started = time.perf_counter()
        step = self.step

        while True:
            for raw_index, path in enumerate(self.paths):
                if raw_index % step != 0:
                    continue

                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise VideoSourceError(f"gagal membaca gambar: {path}")

                if self.source_size is None:
                    h, w = image.shape[:2]
                    self.source_size = (w, h)

                processed, transform = apply_resize(image, self.resize_config)
                now = time.perf_counter()

                self._frames_yielded += 1
                self._elapsed_s = now - started

                yield Frame(
                    index=raw_index,
                    image=processed,
                    # Timestamp diturunkan dari indeks dan fps, bukan wall clock,
                    # supaya deterministik dan bisa dibandingkan dengan label.
                    timestamp_s=raw_index / float(self.declared_fps or 1.0),
                    transform=transform,
                    source_size=self.source_size,
                )

            if not self.loop:
                break

        self._elapsed_s = time.perf_counter() - started

    def describe(self) -> str:
        size = f"{self.source_size[0]}x{self.source_size[1]}" if self.source_size else "?"
        return (
            f"image_sequence {len(self.paths)} gambar | {size} | "
            f"fps: {self.declared_fps:.2f} | step: {self.step} | "
            f"resize: {self.resize_config.mode.value}"
        )


def sequence_from_directory(directory: Path, *, fps: float, **kwargs: object) -> ImageSequenceSource:
    """Bantu untuk kasus sederhana: semua gambar di satu direktori, urut nama.

    Untuk CAUCAFall pakai :func:`pfv.data.caucafall.load_clip` lalu berikan
    ``clip.image_paths``. Fungsi ini hanya untuk direktori yang penamaannya rapi,
    misalnya hasil ekstraksi sendiri.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise VideoSourceError(f"bukan direktori: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise VideoSourceError(f"tidak ada gambar di {directory}")
    return ImageSequenceSource(paths, fps=fps, **kwargs)  # type: ignore[arg-type]
