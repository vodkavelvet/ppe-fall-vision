"""Test ImageSequenceSource.

Sumber ini ada supaya PNG lossless CAUCAFall bisa masuk pipeline tanpa transkode.
Antarmukanya harus sepadan dengan :class:`VideoSource`, karena pipeline di atasnya
tidak boleh peduli dari mana frame datang.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import pytest

from pfv.config import ResizeConfig, ResizeMode
from pfv.io import ImageSequenceSource, VideoSource, VideoSourceError
from pfv.io.image_sequence import sequence_from_directory
from pfv_test_helpers import write_png


@pytest.fixture
def images(tmp_path: Path) -> List[Path]:
    """Sepuluh PNG 720x480, geometri asli CAUCAFall."""
    paths = []
    for i in range(10):
        p = tmp_path / f"frame{i:05d}.png"
        write_png(p, 720, 480)
        paths.append(p)
    return paths


class TestIteration:
    def test_yields_every_image_in_order(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0) as src:
            frames = list(src)
        assert [f.index for f in frames] == list(range(10))
        assert src.frames_yielded == 10

    def test_frame_skip_takes_every_nth(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0, frame_skip=2) as src:
            assert [f.index for f in src] == [0, 3, 6, 9]
            assert src.step == 3

    def test_order_follows_the_given_list_not_the_filenames(self, images: List[Path]) -> None:
        """Urutan adalah tanggung jawab pemanggil, karena penamaan CAUCAFall tidak seragam."""
        reversed_paths = list(reversed(images))
        with ImageSequenceSource(reversed_paths, fps=20.0) as src:
            frames = list(src)
        assert frames[0].image is not None
        # index merujuk posisi dalam daftar, bukan nomor pada nama file.
        assert [f.index for f in frames] == list(range(10))

    def test_loop_repeats_the_sequence(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0, loop=True) as src:
            it = iter(src)
            frames = [next(it) for _ in range(15)]
        assert [f.index for f in frames[:10]] == list(range(10))
        assert frames[10].index == 0

    def test_iteration_after_release_raises(self, images: List[Path]) -> None:
        src = ImageSequenceSource(images, fps=20.0)
        src.release()
        with pytest.raises(VideoSourceError, match="ditutup"):
            list(src)


class TestGeometry:
    def test_source_size_detected_from_first_image(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0) as src:
            frame = next(iter(src))
        assert frame.source_size == (720, 480)
        assert src.source_size == (720, 480)

    def test_resize_preserves_aspect_ratio(self, images: List[Path]) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=360)
        with ImageSequenceSource(images, fps=20.0, resize=cfg) as src:
            frame = next(iter(src))
        w, h = frame.size
        assert max(w, h) == 360
        assert abs(w / h - 720 / 480) < 0.01
        assert frame.source_size == (720, 480)

    def test_native_geometry_passes_through_default_longest_side(
        self, images: List[Path]
    ) -> None:
        """720x480 di bawah 960, jadi tidak diperbesar tanpa allow_upscale."""
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=960)
        with ImageSequenceSource(images, fps=20.0, resize=cfg) as src:
            frame = next(iter(src))
        assert frame.size == (720, 480)
        assert frame.transform.is_identity


class TestTimestamps:
    def test_derived_from_index_and_fps_not_wall_clock(self, images: List[Path]) -> None:
        """Harus deterministik supaya bisa dibandingkan dengan label per frame."""
        with ImageSequenceSource(images, fps=20.0) as src:
            frames = list(src)
        assert frames[0].timestamp_s == pytest.approx(0.0)
        assert frames[4].timestamp_s == pytest.approx(0.2)
        assert frames[9].timestamp_s == pytest.approx(0.45)

    def test_timestamps_respect_frame_skip(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0, frame_skip=1) as src:
            frames = list(src)
        assert frames[1].timestamp_s == pytest.approx(2 / 20.0)

    def test_repeated_runs_give_identical_timestamps(self, images: List[Path]) -> None:
        a = [f.timestamp_s for f in ImageSequenceSource(images, fps=20.0)]
        b = [f.timestamp_s for f in ImageSequenceSource(images, fps=20.0)]
        assert a == b


class TestValidation:
    def test_empty_sequence_rejected(self) -> None:
        with pytest.raises(VideoSourceError, match="kosong"):
            ImageSequenceSource([], fps=20.0)

    def test_missing_image_rejected_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="tidak ditemukan"):
            ImageSequenceSource([tmp_path / "hilang.png"], fps=20.0)

    def test_non_positive_fps_rejected(self, images: List[Path]) -> None:
        with pytest.raises(VideoSourceError, match="fps harus positif"):
            ImageSequenceSource(images, fps=0.0)

    def test_negative_frame_skip_rejected(self, images: List[Path]) -> None:
        with pytest.raises(VideoSourceError, match="negatif"):
            ImageSequenceSource(images, fps=20.0, frame_skip=-1)

    def test_unreadable_image_fails_loudly_during_iteration(
        self, images: List[Path], tmp_path: Path
    ) -> None:
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"bukan png")
        with pytest.raises(VideoSourceError, match="gagal membaca"):
            list(ImageSequenceSource(images + [broken], fps=20.0))


class TestFallModeWarnings:
    def test_caucafall_fps_clears_the_minimum(self, images: List[Path]) -> None:
        src = ImageSequenceSource(images, fps=20.0, fall_mode=True, min_fall_fps=15.0)
        src.release()
        assert src.warnings == ()

    def test_frame_skip_warns_in_fall_mode(self, images: List[Path], caplog) -> None:
        with caplog.at_level(logging.WARNING):
            src = ImageSequenceSource(
                images, fps=20.0, frame_skip=1, fall_mode=True, min_fall_fps=15.0
            )
        src.release()
        joined = " ".join(src.warnings)
        assert "frame_skip" in joined
        # Pada 20 fps, frame_skip=1 menjatuhkan ke 10 fps: dua peringatan sekaligus.
        assert "di bawah minimum" in joined
        assert src.sampled_fps == pytest.approx(10.0)

    def test_no_warning_when_fall_mode_off(self, images: List[Path]) -> None:
        src = ImageSequenceSource(images, fps=5.0, frame_skip=4, fall_mode=False)
        src.release()
        assert src.warnings == ()


class TestInterfaceParityWithVideoSource:
    """Pipeline harus bisa menerima keduanya tanpa cabang khusus."""

    ATTRS = (
        "step",
        "declared_fps",
        "sampled_fps",
        "effective_fps",
        "frames_yielded",
        "elapsed_s",
        "warnings",
        "frame_count",
        "source_size",
        "describe",
        "release",
    )

    def test_same_public_surface(self, images: List[Path], video_30: Path) -> None:
        seq = ImageSequenceSource(images, fps=20.0)
        vid = VideoSource(video_30)
        try:
            for name in self.ATTRS:
                assert hasattr(seq, name), f"ImageSequenceSource kehilangan {name}"
                assert hasattr(vid, name), f"VideoSource kehilangan {name}"
        finally:
            seq.release()
            vid.release()

    def test_both_are_context_managers_yielding_frames(
        self, images: List[Path], video_30: Path
    ) -> None:
        with ImageSequenceSource(images, fps=20.0) as seq, VideoSource(video_30) as vid:
            a = next(iter(seq))
            b = next(iter(vid))
        for frame in (a, b):
            assert frame.index == 0
            assert frame.transform is not None
            assert len(frame.source_size) == 2

    def test_describe_is_single_line(self, images: List[Path]) -> None:
        with ImageSequenceSource(images, fps=20.0) as src:
            next(iter(src))
            text = src.describe()
        assert "\n" not in text
        assert "720x480" in text


class TestSequenceFromDirectory:
    def test_reads_sorted_images(self, tmp_path: Path, images: List[Path]) -> None:
        src = sequence_from_directory(tmp_path, fps=20.0)
        try:
            assert len(src.paths) == 10
            assert src.paths == sorted(src.paths)
        finally:
            src.release()

    def test_empty_directory_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "kosong"
        empty.mkdir()
        with pytest.raises(VideoSourceError, match="tidak ada gambar"):
            sequence_from_directory(empty, fps=20.0)

    def test_non_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="bukan direktori"):
            sequence_from_directory(tmp_path / "hilang", fps=20.0)
