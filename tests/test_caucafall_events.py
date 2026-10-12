"""Test pembentukan ground truth evaluasi dan verifikasi invarian bundel."""

from __future__ import annotations

from pathlib import Path

import pytest

from pfv.data.caucafall import (
    EVENT_CSV_FIELDS,
    EXPECTED_N_LABELLED_FRAMES,
    NATIVE_FPS,
    clip_to_row,
    discover_clips,
    load_clip,
    verify_bundle,
)


class TestClipToRow:
    def test_fall_clip_row(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        d = make_caucafall(
            3, "Fall forward", fall_frames=(4, 9), n_frames=10, prefix="cfs", root=root
        )
        row = clip_to_row(load_clip(d), root)

        assert row["clip_id"] == "s03_fall_forward"
        assert row["subject_id"] == 3
        assert row["kind"] == "fall"
        assert row["n_frames"] == 10
        assert row["fps"] == NATIVE_FPS
        assert row["fall_start_frame"] == 4
        assert row["fall_end_frame"] == 9
        assert row["n_fall_frames"] == 6
        assert row["directory"] == "Subject.3/Fall forward"

    def test_seconds_derived_from_native_fps(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        d = make_caucafall(
            1, "Fall left", fall_frames=(10, 19), n_frames=20, prefix="cis", root=root
        )
        row = clip_to_row(load_clip(d), root)
        # 20 fps, jadi frame 10 berada di detik 0,5.
        assert row["fall_start_s"] == pytest.approx(0.5)
        assert row["duration_s"] == pytest.approx(1.0)

    def test_adl_clip_has_empty_fall_columns(self, make_caucafall, tmp_path: Path) -> None:
        """String kosong, bukan 0. Nilai 0 akan terbaca sebagai 'jatuh di frame 0'."""
        root = tmp_path / "CAUCAFall"
        d = make_caucafall(2, "Kneel", n_frames=8, prefix="ars", root=root)
        row = clip_to_row(load_clip(d), root)

        assert row["kind"] == "adl"
        assert row["fall_start_frame"] == ""
        assert row["fall_end_frame"] == ""
        assert row["fall_start_s"] == ""
        assert row["n_fall_frames"] == 0

    def test_row_keys_match_csv_contract(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        d = make_caucafall(1, "Walk", n_frames=4, prefix="cams", root=root)
        row = clip_to_row(load_clip(d), root)
        assert set(row) == set(EVENT_CSV_FIELDS)


class TestVerifyBundle:
    def _bundle(self, make_caucafall, root: Path, n_fall: int, n_adl: int) -> list:
        """Bangun bundel kecil; invarian jumlah pasti dilanggar, itu disengaja."""
        for i in range(n_fall):
            make_caucafall(
                1, f"Fall forward" if i == 0 else "Fall left",
                fall_frames=(2, 4), n_frames=6, prefix="cfs", root=root,
            )
        for i in range(n_adl):
            make_caucafall(
                1, "Walk" if i == 0 else "Kneel", n_frames=6, prefix="cams", root=root
            )
        return discover_clips(root)

    def test_reports_wrong_clip_count(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        clips = self._bundle(make_caucafall, root, n_fall=1, n_adl=1)
        problems = verify_bundle(clips)
        assert any("jumlah klip" in p for p in problems)

    def test_reports_fall_clip_without_fall_label(self, make_caucafall, tmp_path: Path) -> None:
        """Ini gejala paling berbahaya: ground truth hilang tanpa error lain."""
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Fall forward", fall_frames=None, n_frames=6, prefix="cfs", root=root)
        problems = verify_bundle(discover_clips(root))
        assert any("klip jatuh tanpa label fall" in p for p in problems)

    def test_reports_adl_clip_with_fall_label(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Walk", fall_frames=(2, 3), n_frames=6, prefix="cams", root=root)
        problems = verify_bundle(discover_clips(root))
        assert any("klip ADL punya label fall" in p for p in problems)

    def test_reports_frame_with_multiple_boxes(self, make_caucafall, tmp_path: Path) -> None:
        """Seluruh 19.999 label bundel asli punya tepat satu kotak."""
        root = tmp_path / "CAUCAFall"
        d = make_caucafall(1, "Walk", n_frames=6, prefix="cams", root=root)
        clip = load_clip(d)
        clip.frames[2].label_path.write_text(
            "0 0.3 0.5 0.1 0.5\n0 0.7 0.5 0.1 0.5\n", encoding="utf-8"
        )
        problems = verify_bundle(discover_clips(root))
        assert any("jumlah kotak != 1" in p for p in problems)

    def test_reports_unknown_activity_name(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Berlari", n_frames=4, prefix="xxx", root=root)
        problems = verify_bundle(discover_clips(root))
        assert any("aktivitas tak dikenal" in p for p in problems)

    def test_frame_total_invariant_is_the_audited_number(self) -> None:
        assert EXPECTED_N_LABELLED_FRAMES == 19999


class TestRealBundleIfPresent:
    """Dijalankan hanya bila bundel asli ada di mesin ini.

    Test lain semuanya sintetis supaya suite tetap jalan di mana pun, termasuk di
    kernel Kaggle. Yang ini mengunci hasil audit terhadap data sebenarnya.
    """

    ROOT = Path("Dataset CAUCAFall/CAUCAFall")

    @pytest.fixture(autouse=True)
    def _skip_without_bundle(self, in_repo_root: Path) -> None:
        if not (in_repo_root / self.ROOT).is_dir():
            pytest.skip("bundel CAUCAFall tidak ada di mesin ini")

    def test_bundle_satisfies_every_audited_invariant(self) -> None:
        clips = discover_clips(self.ROOT)
        assert verify_bundle(clips) == []

    def test_the_six_anomalous_folders_lose_no_labels(self) -> None:
        expected_unlabelled = {
            "Subject.2/Fall backwards": 0,
            "Subject.8/Hop": 0,
            "Subject.8/Pick up object": 0,
            "Subject.9/Walk": 0,
            "Subject.5/Kneel": 1,
            "Subject.6/Walk": 1,
        }
        for rel, n_unlabelled in expected_unlabelled.items():
            clip = load_clip(self.ROOT / rel)
            assert len(clip.unlabelled_images) == n_unlabelled, rel

    def test_every_fall_clip_has_exactly_one_contiguous_interval(self) -> None:
        clips = [c for c in discover_clips(self.ROOT) if c.kind == "fall"]
        assert len(clips) == 50
        assert all(len(c.fall_intervals()) == 1 for c in clips)
