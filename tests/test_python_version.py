"""Penjaga versi interpreter.

macOS memasang Python 3.9 di ``/usr/bin/python3`` sebagai bagian dari OS, dan
versi itu tidak bisa (dan tidak boleh) dihapus. Yang bisa dilakukan adalah
memastikan salah interpreter gagal dengan pesan yang jelas, bukan dengan
``ImportError`` yang menyesatkan dari dependensi.
"""

from __future__ import annotations

import sys

import pytest

import pfv
from pfv import MIN_PYTHON, _check_python_version


class TestPythonVersionGuard:
    def test_current_interpreter_satisfies_minimum(self) -> None:
        assert sys.version_info >= MIN_PYTHON, (
            f"test dijalankan dengan Python {sys.version.split()[0]}; "
            "pakai .venv/bin/python"
        )

    def test_minimum_matches_requires_python_in_pyproject(self, repo_root) -> None:
        """Satu sumber kebenaran: batas di kode harus sama dengan di pyproject."""
        text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        expected = ">=" + ".".join(str(n) for n in MIN_PYTHON)
        assert f'requires-python = "{expected}"' in text

    def test_guard_passes_on_supported_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
        _check_python_version()  # tidak boleh raise

    def test_guard_rejects_old_version_with_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 9, 6, "final", 0))
        monkeypatch.setattr(sys, "executable", "/usr/bin/python3")

        with pytest.raises(RuntimeError) as excinfo:
            _check_python_version()

        message = str(excinfo.value)
        assert "3.9.6" in message
        assert "/usr/bin/python3" in message
        # Pesan harus memberi tahu apa yang harus dilakukan, bukan cuma menolak.
        assert ".venv/bin/python" in message

    def test_guard_runs_at_import_time(self) -> None:
        """Penjaga dipanggil saat impor, bukan hanya tersedia sebagai fungsi."""
        source = (pfv.__file__ and open(pfv.__file__, encoding="utf-8").read()) or ""
        assert source.rstrip().endswith("_check_python_version()")
