"""Test VideoSource: pembacaan frame, frame_skip, penskalaan, peringatan."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

from pfv.config import ResizeConfig, ResizeMode
from pfv.io import SourceKind, VideoSource, VideoSourceError
from pfv.io.video_source import apply_resize, compute_resize, parse_source
from pfv_test_helpers import CAUCAFALL_FPS, CAUCAFALL_SIZE


# --------------------------------------------------------------- parse_source


class TestParseSource:
    def test_digit_string_is_webcam(self) -> None:
        assert parse_source("0") == (SourceKind.WEBCAM, 0)
        assert parse_source("2") == (SourceKind.WEBCAM, 2)

    def test_int_is_webcam(self) -> None:
        assert parse_source(1) == (SourceKind.WEBCAM, 1)

    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://cam.local/stream1",
            "RTSP://cam.local/stream1",
            "http://cam.local/video",
            "https://cam.local/video",
        ],
    )
    def test_network_urls(self, url: str) -> None:
        kind, target = parse_source(url)
        assert kind is SourceKind.NETWORK
        assert target == url

    def test_existing_file(self, video_30: Path) -> None:
        kind, target = parse_source(str(video_30))
        assert kind is SourceKind.FILE
        assert target == str(video_30)

    def test_missing_file_raises_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="tidak ditemukan"):
            parse_source(tmp_path / "tidak_ada.mp4")

    def test_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="direktori"):
            parse_source(tmp_path)

    def test_empty_rejected(self) -> None:
        with pytest.raises(VideoSourceError, match="kosong"):
            parse_source("   ")

    def test_live_flag(self) -> None:
        assert SourceKind.WEBCAM.is_live
        assert SourceKind.NETWORK.is_live
        assert not SourceKind.FILE.is_live


# ------------------------------------------------------------- compute_resize


class TestComputeResize:
    def test_mode_none_is_identity(self) -> None:
        w, h, t = compute_resize(1080, 960, ResizeConfig(mode=ResizeMode.NONE))
        assert (w, h) == (1080, 960)
        assert t.is_identity

    def test_longest_side_preserves_aspect_ratio(self) -> None:
        """Rasio aspek non-persegi harus tetap setelah diskalakan."""
        src_w, src_h = 1080, 960
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=640)
        w, h, t = compute_resize(src_w, src_h, cfg)

        assert max(w, h) == 640
        # Toleransi 1 piksel untuk pembulatan, bukan lebih.
        assert abs(w / h - src_w / src_h) < 0.01
        assert t.scale == pytest.approx(640 / 1080)
        assert (t.pad_x, t.pad_y) == (0, 0)

    def test_longest_side_uses_the_longer_dimension(self) -> None:
        # Potret: tinggi yang lebih panjang, jadi tinggi yang dipatok.
        w, h, _ = compute_resize(480, 960, ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=480))
        assert h == 480
        assert w == 240

    def test_no_upscale_by_default(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=2000)
        w, h, t = compute_resize(640, 480, cfg)
        assert (w, h) == (640, 480)
        assert t.is_identity

    def test_upscale_when_allowed(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=1280, allow_upscale=True)
        w, h, t = compute_resize(640, 480, cfg)
        assert (w, h) == (1280, 960)
        assert t.scale == pytest.approx(2.0)

    def test_letterbox_pads_instead_of_stretching(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LETTERBOX, letterbox_size=(640, 640))
        w, h, t = compute_resize(1080, 960, cfg)

        assert (w, h) == (640, 640)
        assert t.scale == pytest.approx(640 / 1080)
        # Sumber lebih lebar dari tinggi, jadi padding ada di atas dan bawah.
        assert t.pad_x == 0
        assert t.pad_y > 0

    def test_invalid_frame_size_rejected(self) -> None:
        with pytest.raises(VideoSourceError, match="tidak valid"):
            compute_resize(0, 480, ResizeConfig())


class TestApplyResize:
    def test_longest_side_output_shape_and_aspect(self) -> None:
        image = np.zeros((960, 1080, 3), dtype=np.uint8)
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=540)
        out, t = apply_resize(image, cfg)

        h, w = out.shape[:2]
        assert max(w, h) == 540
        assert abs(w / h - 1080 / 960) < 0.01
        assert t.scale == pytest.approx(0.5)

    def test_letterbox_output_is_exact_canvas_with_padding(self) -> None:
        image = np.full((960, 1080, 3), 255, dtype=np.uint8)
        cfg = ResizeConfig(mode=ResizeMode.LETTERBOX, letterbox_size=(640, 640))
        out, t = apply_resize(image, cfg)

        assert out.shape[:2] == (640, 640)
        # Baris paling atas berada di area padding, jadi tetap hitam.
        assert out[0, :, :].max() == 0
        # Tengah kanvas berisi gambar.
        assert out[320, 320, :].max() == 255
        assert t.pad_y > 0

    def test_identity_returns_same_array(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        out, t = apply_resize(image, ResizeConfig(mode=ResizeMode.NONE))
        assert out is image
        assert t.is_identity


# ----------------------------------------------------------- FrameTransform


class TestFrameTransform:
    def test_round_trip_longest_side(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=540)
        _, _, t = compute_resize(1080, 960, cfg)
        # Titik tengah gambar terskala harus memetakan ke tengah gambar asli.
        x, y = t.to_source(270, 240)
        assert x == pytest.approx(540, abs=1)
        assert y == pytest.approx(480, abs=1)

    def test_round_trip_letterbox_accounts_for_padding(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LETTERBOX, letterbox_size=(640, 640))
        _, _, t = compute_resize(1080, 960, cfg)
        # Sudut kiri-atas gambar sesungguhnya berada setelah padding.
        x, y = t.to_source(t.pad_x, t.pad_y)
        assert x == pytest.approx(0, abs=1)
        assert y == pytest.approx(0, abs=1)

    def test_box_mapping(self) -> None:
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=540)
        _, _, t = compute_resize(1080, 960, cfg)
        box = t.to_source_box((10, 20, 110, 220))
        assert box == pytest.approx((20, 40, 220, 440), abs=1)


# ---------------------------------------------------------------- iteration


class TestVideoSourceIteration:
    def test_yields_every_frame_by_default(self, video_30: Path) -> None:
        with VideoSource(video_30) as src:
            frames = list(src)

        assert len(frames) == 30
        assert [f.index for f in frames] == list(range(30))
        assert src.frames_yielded == 30

    def test_frame_skip_takes_every_nth_frame(self, video_30: Path) -> None:
        with VideoSource(video_30, frame_skip=2) as src:
            indices = [f.index for f in src]

        # frame_skip=2 berarti step 3: lewati 2 frame di antara yang diproses.
        assert indices == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        assert src.step == 3

    def test_frame_skip_one_halves_the_stream(self, video_30: Path) -> None:
        with VideoSource(video_30, frame_skip=1) as src:
            indices = [f.index for f in src]
        assert indices == list(range(0, 30, 2))

    def test_negative_frame_skip_rejected(self, video_30: Path) -> None:
        with pytest.raises(VideoSourceError, match="negatif"):
            VideoSource(video_30, frame_skip=-1)

    def test_metadata_read_from_file(self, caucafall_like: Tuple[Path, Tuple[int, int]]) -> None:
        path, size = caucafall_like
        with VideoSource(path) as src:
            assert src.kind is SourceKind.FILE
            assert src.source_size == size
            assert src.declared_fps == pytest.approx(CAUCAFALL_FPS, abs=0.5)
            assert src.frame_count == 12

    def test_caucafall_geometry_passes_through_default_resize(
        self, caucafall_like: Tuple[Path, Tuple[int, int]]
    ) -> None:
        """720x480 lebih kecil dari longest_side default 960, jadi tidak diskalakan.

        Tanpa ``allow_upscale``, memperbesar akan menambah piksel tanpa menambah
        informasi dan hanya memperlambat inference.
        """
        path, size = caucafall_like
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=960)
        with VideoSource(path, resize=cfg) as src:
            frame = next(iter(src))
        assert frame.size == size
        assert frame.transform.is_identity

    def test_caucafall_fps_clears_fall_minimum(
        self, caucafall_like: Tuple[Path, Tuple[int, int]]
    ) -> None:
        """20 fps di atas ambang 15, jadi tidak ada peringatan -- tapi marginnya tipis."""
        path, _ = caucafall_like
        src = VideoSource(path, fall_mode=True, min_fall_fps=15.0)
        src.release()
        assert src.warnings == ()
        assert src.sampled_fps == pytest.approx(CAUCAFALL_FPS, abs=0.5)

    def test_any_frame_skip_breaks_caucafall_fall_mode(
        self, caucafall_like: Tuple[Path, Tuple[int, int]]
    ) -> None:
        """Pada 20 fps, frame_skip=1 menjatuhkan ke 10 fps dan langsung di bawah ambang.

        Ini konsekuensi konkret dari geometri CAUCAFall yang sebenarnya: marginnya
        hanya 5 fps, jadi frame_skip sekecil apa pun sudah merusak.
        """
        path, _ = caucafall_like
        src = VideoSource(path, frame_skip=1, fall_mode=True, min_fall_fps=15.0)
        src.release()
        assert src.sampled_fps == pytest.approx(10.0, abs=0.3)
        assert any("di bawah minimum" in w for w in src.warnings)

    def test_source_size_preserved_on_frames_even_after_resize(
        self, caucafall_like: Tuple[Path, Tuple[int, int]]
    ) -> None:
        path, size = caucafall_like
        cfg = ResizeConfig(mode=ResizeMode.LONGEST_SIDE, longest_side=540)
        with VideoSource(path, resize=cfg) as src:
            frame = next(iter(src))

        assert frame.source_size == size
        assert frame.size != size
        # Rasio aspek tetap: ini yang melindungi bbox_aspect_ratio.
        w, h = frame.size
        assert abs(w / h - size[0] / size[1]) < 0.01

    def test_timestamps_follow_declared_fps_for_files(self, video_30: Path) -> None:
        with VideoSource(video_30) as src:
            frames = list(src)

        assert frames[0].timestamp_s == pytest.approx(0.0)
        # 30 fps, jadi frame ke-15 berada di detik 0,5.
        assert frames[15].timestamp_s == pytest.approx(0.5, abs=0.02)

    def test_timestamps_respect_frame_skip(self, video_30: Path) -> None:
        with VideoSource(video_30, frame_skip=1) as src:
            frames = list(src)
        # Indeks tetap merujuk stream asli, jadi timestamp melompat 2 frame.
        assert frames[1].timestamp_s == pytest.approx(2 / 30, abs=0.02)

    def test_loop_restarts_file(self, video_30: Path) -> None:
        with VideoSource(video_30, loop=True) as src:
            iterator = iter(src)
            frames = [next(iterator) for _ in range(45)]

        assert len(frames) == 45
        # Setelah 30 frame, indeks kembali ke 0.
        assert frames[30].index == 0

    def test_effective_fps_measured_only_after_iteration(self, video_30: Path) -> None:
        src = VideoSource(video_30)
        assert src.effective_fps is None
        list(src)
        assert src.effective_fps is not None and src.effective_fps > 0
        src.release()

    def test_sampled_fps_accounts_for_frame_skip(self, video_30: Path) -> None:
        with VideoSource(video_30, frame_skip=2) as src:
            assert src.declared_fps == pytest.approx(30.0, abs=0.5)
            assert src.sampled_fps == pytest.approx(10.0, abs=0.2)

    def test_iteration_after_release_raises(self, video_30: Path) -> None:
        src = VideoSource(video_30)
        src.release()
        with pytest.raises(VideoSourceError, match="ditutup"):
            list(src)

    def test_describe_is_single_line(self, video_30: Path) -> None:
        with VideoSource(video_30) as src:
            text = src.describe()
        assert "\n" not in text
        assert "640x480" in text


# ----------------------------------------------------------------- warnings


class TestFallModeWarnings:
    def test_frame_skip_with_fall_mode_warns(self, video_30: Path, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            src = VideoSource(video_30, frame_skip=2, fall_mode=True, min_fall_fps=15.0)
        src.release()

        joined = " ".join(src.warnings)
        assert "frame_skip" in joined
        assert "hip_vertical_velocity" in joined
        assert any("frame_skip" in rec.message for rec in caplog.records)

    def test_low_sampled_fps_warns(self, make_video: Callable[..., Path], caplog) -> None:
        slow = make_video(n_frames=10, fps=8.0, name="slow.mp4")
        with caplog.at_level(logging.WARNING):
            src = VideoSource(slow, fall_mode=True, min_fall_fps=15.0)
        src.release()

        joined = " ".join(src.warnings)
        assert "di bawah minimum" in joined

    def test_no_warning_when_fps_is_adequate(self, video_30: Path) -> None:
        src = VideoSource(video_30, fall_mode=True, min_fall_fps=15.0)
        src.release()
        assert src.warnings == ()

    def test_no_warning_when_fall_mode_off(self, video_30: Path) -> None:
        src = VideoSource(video_30, frame_skip=5, fall_mode=False)
        src.release()
        assert src.warnings == ()

    def test_unknown_fps_warns_to_run_doctor(self, video_30: Path, monkeypatch) -> None:
        # Simulasikan sumber yang tidak melaporkan fps, seperti banyak webcam
        # dan sebagian stream RTSP.
        import cv2 as _cv2

        real_get = _cv2.VideoCapture.get

        def fake_get(self, prop):  # type: ignore[no-untyped-def]
            if prop == _cv2.CAP_PROP_FPS:
                return 0.0
            return real_get(self, prop)

        monkeypatch.setattr(_cv2.VideoCapture, "get", fake_get, raising=True)

        src = VideoSource(video_30, fall_mode=True)
        src.release()

        assert src.declared_fps is None
        assert src.sampled_fps is None
        assert any("pfv doctor" in w for w in src.warnings)
