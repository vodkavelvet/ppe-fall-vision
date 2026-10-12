"""Test CLI. Semuanya mode headless (``--no-window``) supaya bisa jalan otomatis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from typer.testing import CliRunner

from pfv import __version__
from pfv.cli import FpsMeter, app

runner = CliRunner()


class TestVersionCommand:
    def test_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestPlayCommand:
    def test_plays_file_headless(self, video_30: Path) -> None:
        result = runner.invoke(app, ["play", "--source", str(video_30), "--no-window"])
        assert result.exit_code == 0
        assert "30 frame" in result.stdout
        assert "640x480" in result.stdout

    def test_max_frames_stops_early(self, video_30: Path) -> None:
        result = runner.invoke(
            app, ["play", "--source", str(video_30), "--no-window", "--max-frames", "7"]
        )
        assert result.exit_code == 0
        assert "7 frame" in result.stdout

    def test_frame_skip_flag_overrides_config(self, video_30: Path) -> None:
        result = runner.invoke(
            app, ["play", "--source", str(video_30), "--no-window", "--frame-skip", "2"]
        )
        assert result.exit_code == 0
        assert "step: 3" in result.stdout
        assert "10 frame" in result.stdout

    def test_missing_file_exits_nonzero_with_readable_message(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["play", "--source", str(tmp_path / "hilang.mp4"), "--no-window"]
        )
        assert result.exit_code == 1
        assert "tidak ditemukan" in result.output

    def test_explicit_config_is_used(
        self, video_30: Path, make_video: Callable[..., Path], tmp_path: Path
    ) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text(
            "video:\n"
            "  resize:\n"
            "    mode: letterbox\n"
            "    letterbox_size: [320, 320]\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app, ["play", "--source", str(video_30), "--config", str(cfg), "--no-window"]
        )
        assert result.exit_code == 0
        assert "resize: letterbox" in result.stdout


class TestFpsMeter:
    def test_needs_two_samples(self) -> None:
        meter = FpsMeter()
        assert meter.tick(0.0) is None
        assert meter.tick(0.1) is not None

    def test_computes_rate_over_window(self) -> None:
        meter = FpsMeter(window=5)
        for i in range(5):
            fps = meter.tick(i * 0.1)  # 10 fps
        assert fps is not None
        assert abs(fps - 10.0) < 0.01

    def test_handles_zero_span(self) -> None:
        meter = FpsMeter()
        meter.tick(1.0)
        assert meter.tick(1.0) is None


class TestClipCommand:
    """Demo Task 2: memutar klip CAUCAFall dari PNG dengan overlay ground truth."""

    def test_plays_fall_clip_and_reports_interval(
        self, make_caucafall, tmp_path: Path
    ) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(
            1, "Fall forward", fall_frames=(4, 9), n_frames=10, prefix="cfs", root=root
        )
        result = runner.invoke(
            app,
            ["clip", "Subject.1/Fall forward", "--dataset-root", str(root), "--no-window"],
        )
        assert result.exit_code == 0
        assert "s01_fall_forward" in result.stdout
        assert "onset: 4" in result.stdout
        assert "(4, 9)" in result.stdout
        assert "10 frame" in result.stdout

    def test_adl_clip_reports_no_onset(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(5, "Kneel", n_frames=8, prefix="ars", root=root)
        result = runner.invoke(
            app, ["clip", "Subject.5/Kneel", "--dataset-root", str(root), "--no-window"]
        )
        assert result.exit_code == 0
        assert "adl" in result.stdout
        assert "onset: -" in result.stdout

    def test_max_frames_stops_early(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Walk", n_frames=20, prefix="cams", root=root)
        result = runner.invoke(
            app,
            [
                "clip", "Subject.1/Walk",
                "--dataset-root", str(root),
                "--no-window",
                "--max-frames", "6",
            ],
        )
        assert result.exit_code == 0
        assert "6 frame dalam" in result.stdout

    def test_missing_clip_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["clip", "Subject.99/Tidak ada", "--dataset-root", str(tmp_path), "--no-window"],
        )
        assert result.exit_code == 1
        assert "bukan direktori" in result.output

    def test_naming_anomaly_clip_plays_without_loss(
        self, make_caucafall, tmp_path: Path
    ) -> None:
        """Pola Subject.2/Fall backwards: satu PNG bernama '... - copia'."""
        root = tmp_path / "CAUCAFall"
        make_caucafall(
            2, "Fall backwards", fall_frames=(5, 9), n_frames=10,
            prefix="cas", anomaly="copia", root=root,
        )
        result = runner.invoke(
            app,
            ["clip", "Subject.2/Fall backwards", "--dataset-root", str(root), "--no-window"],
        )
        assert result.exit_code == 0
        assert "10 frame" in result.stdout


class TestPoseAndTrackCommands:
    """`pfv pose` dan `pfv track` menjalankan model nyata, jadi di-skip tanpa bobot."""

    @pytest.fixture(autouse=True)
    def _weights(self, in_repo_root: Path) -> None:
        if not (in_repo_root / "models" / "yolo11n-pose.pt").exists():
            pytest.skip("bobot pose belum diunduh")

    def _clip(self, make_caucafall, tmp_path: Path) -> Path:
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Walk", n_frames=6, prefix="cams", root=root)
        return root

    def test_pose_runs_headless_and_reports_rates(self, make_caucafall, tmp_path: Path) -> None:
        root = self._clip(make_caucafall, tmp_path)
        result = runner.invoke(
            app,
            ["pose", "--clip", "Subject.1/Walk", "--dataset-root", str(root), "--no-window"],
        )
        assert result.exit_code == 0, result.output
        assert "6 frame" in result.stdout
        assert "ada orang" in result.stdout
        assert "torso lengkap" in result.stdout

    def test_track_runs_headless_and_reports_distinct_ids(
        self, make_caucafall, tmp_path: Path
    ) -> None:
        root = self._clip(make_caucafall, tmp_path)
        result = runner.invoke(
            app,
            ["track", "--clip", "Subject.1/Walk", "--dataset-root", str(root), "--no-window"],
        )
        assert result.exit_code == 0, result.output
        assert "6 frame" in result.stdout
        assert "jumlah ID berbeda" in result.stdout

    def test_track_missing_clip_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["track", "--clip", "Subject.9/Hilang", "--dataset-root", str(tmp_path), "--no-window"],
        )
        assert result.exit_code == 1

    def test_pose_missing_source_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["pose", "--source", str(tmp_path / "hilang.mp4"), "--no-window"]
        )
        assert result.exit_code == 1


class TestDatasetRootEnvVar:
    """PFV_DATASET_ROOT memungkinkan anggota tim menaruh dataset di mana saja."""

    def test_clip_reads_dataset_root_from_environment(
        self, make_caucafall, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "di_tempat_lain" / "CAUCAFall"
        make_caucafall(
            1, "Fall forward", fall_frames=(4, 9), n_frames=10, prefix="cfs", root=root
        )
        monkeypatch.setenv("PFV_DATASET_ROOT", str(root))

        result = runner.invoke(app, ["clip", "Subject.1/Fall forward", "--no-window"])
        assert result.exit_code == 0, result.output
        assert "s01_fall_forward" in result.stdout

    def test_explicit_flag_wins_over_environment(
        self, make_caucafall, tmp_path: Path, monkeypatch
    ) -> None:
        real = tmp_path / "nyata" / "CAUCAFall"
        make_caucafall(1, "Walk", n_frames=4, prefix="cams", root=real)
        monkeypatch.setenv("PFV_DATASET_ROOT", str(tmp_path / "salah"))

        result = runner.invoke(
            app,
            ["clip", "Subject.1/Walk", "--dataset-root", str(real), "--no-window"],
        )
        assert result.exit_code == 0, result.output

    def test_wrong_environment_value_fails_clearly(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PFV_DATASET_ROOT", str(tmp_path / "tidak_ada"))
        result = runner.invoke(app, ["clip", "Subject.1/Walk", "--no-window"])
        assert result.exit_code == 1
        assert "bukan direktori" in result.output
