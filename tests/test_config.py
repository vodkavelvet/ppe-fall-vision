"""Test pemuatan config dan validasinya."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pfv.config import AppConfig, ResizeConfig, ResizeMode, Secrets, load_config


class TestAppConfigFromYaml:
    def test_defaults_when_yaml_absent(self) -> None:
        cfg = AppConfig.from_yaml(None)
        assert cfg.video.source == "0"
        assert cfg.video.frame_skip == 0
        assert cfg.video.resize.mode is ResizeMode.NONE
        assert cfg.fall.enabled is True
        assert cfg.fall.min_fps == 15.0

    def test_reads_shipped_default_config(self, repo_root: Path) -> None:
        path = repo_root / "config" / "default.yaml"
        assert path.exists(), "config/default.yaml harus ada di repo"
        cfg = AppConfig.from_yaml(path)
        assert cfg.video.resize.mode is ResizeMode.LONGEST_SIDE
        assert cfg.video.resize.longest_side == 960
        # Default harus memproses setiap frame, karena fall aktif secara default.
        assert cfg.video.frame_skip == 0
        assert cfg.fall.enabled is True

    def test_dotted_overrides_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("video:\n  source: '0'\n  frame_skip: 0\n", encoding="utf-8")
        cfg = AppConfig.from_yaml(path, **{"video.source": "clip.mp4", "video.frame_skip": 3})
        assert cfg.video.source == "clip.mp4"
        assert cfg.video.frame_skip == 3

    def test_none_overrides_ignored(self, tmp_path: Path) -> None:
        """Flag CLI yang tidak diisi tidak boleh menghapus nilai dari YAML."""
        path = tmp_path / "c.yaml"
        path.write_text("video:\n  source: 'clip.mp4'\n", encoding="utf-8")
        cfg = AppConfig.from_yaml(path, **{"video.source": None})
        assert cfg.video.source == "clip.mp4"

    def test_override_creates_missing_nesting(self) -> None:
        cfg = AppConfig.from_yaml(None, **{"fall.min_fps": 25.0})
        assert cfg.fall.min_fps == 25.0

    def test_empty_yaml_is_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert AppConfig.from_yaml(path).video.source == "0"

    def test_non_mapping_yaml_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            AppConfig.from_yaml(path)

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        """Salah tulis nama setelan harus gagal keras, bukan diabaikan diam-diam."""
        path = tmp_path / "typo.yaml"
        path.write_text("video:\n  frameskip: 2\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            AppConfig.from_yaml(path)

    def test_negative_frame_skip_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.from_yaml(None, **{"video.frame_skip": -1})


class TestResizeConfigValidation:
    def test_longest_side_requires_value(self) -> None:
        with pytest.raises(ValidationError, match="longest_side"):
            ResizeConfig(mode=ResizeMode.LONGEST_SIDE)

    def test_letterbox_requires_size(self) -> None:
        with pytest.raises(ValidationError, match="letterbox_size"):
            ResizeConfig(mode=ResizeMode.LETTERBOX)

    def test_letterbox_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="positif"):
            ResizeConfig(mode=ResizeMode.LETTERBOX, letterbox_size=(0, 640))

    def test_no_stretch_mode_exists(self) -> None:
        """Mode yang mendistorsi rasio aspek sengaja tidak disediakan."""
        assert {m.value for m in ResizeMode} == {"none", "longest_side", "letterbox"}

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResizeConfig(mode="stretch")  # type: ignore[arg-type]


class TestLoadConfig:
    def test_falls_back_to_shipped_default(self, in_repo_root: Path) -> None:
        cfg = load_config()
        assert cfg.video.resize.longest_side == 960

    def test_uses_builtin_defaults_when_no_config_file_present(self, tmp_path: Path, monkeypatch) -> None:
        """Di direktori tanpa config/default.yaml, jatuh ke default bawaan kode."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.video.resize.mode is ResizeMode.NONE
        assert cfg.video.source == "0"

    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("fall:\n  min_fps: 20\n", encoding="utf-8")
        assert load_config(path).fall.min_fps == 20.0


class TestSecrets:
    def test_reads_prefixed_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PFV_TELEGRAM_TOKEN", "abc123")
        assert Secrets().telegram_token == "abc123"

    def test_missing_secrets_are_none_not_errors(self, monkeypatch) -> None:
        for key in ("PFV_TELEGRAM_TOKEN", "PFV_TELEGRAM_CHAT_ID", "PFV_ROBOFLOW_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        secrets = Secrets(_env_file=None)
        assert secrets.telegram_token is None
        assert secrets.roboflow_api_key is None
