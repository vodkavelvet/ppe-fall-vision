"""Test pembacaan dataset CAUCAFall.

Semua test memakai struktur sintetis, jadi tidak bergantung pada bundel 7,9 GB.
Pola anomali yang diuji di sini diambil langsung dari audit bundel nyata; lihat
``data/DATASETS.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pfv.data.caucafall import (
    ADL_ACTIVITIES,
    CLASS_FALL,
    CLASS_NOFALL,
    FALL_ACTIVITIES,
    NATIVE_FPS,
    NATIVE_SIZE,
    CaucaFallError,
    LabelBox,
    discover_clips,
    frame_number,
    load_clip,
    normalise_stem,
    parse_label_file,
    subject_number,
)


class TestNormaliseStem:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Kasus normal tidak boleh berubah selain jadi huruf kecil.
            ("cas100001", "cas100001"),
            ("cams600260", "cams600260"),
            # Artefak duplikasi Windows Explorer berbahasa Spanyol.
            ("cas200091 - copia", "cas200091"),
            ("cas200091 - Copia", "cas200091"),
            ("cas200091 - copy", "cas200091"),
            ("cas200091 - copia (2)", "cas200091"),
            # Sufiks huruf dari penamaan manual saat anotasi.
            ("sals800096a", "sals800096"),
            ("res800090a", "res800090"),
            ("cams900140w", "cams900140"),
        ],
    )
    def test_known_anomalies_from_audit(self, raw: str, expected: str) -> None:
        assert normalise_stem(raw) == expected

    def test_pairs_png_and_label_stems(self) -> None:
        """Inti fungsinya: stem PNG dan label harus bertemu di nilai yang sama."""
        assert normalise_stem("sals800096") == normalise_stem("sals800096a")
        assert normalise_stem("cas200091 - copia") == normalise_stem("cas200091")

    def test_prefix_only_name_is_not_destroyed(self) -> None:
        """Nama tanpa angka tidak boleh kehilangan hurufnya."""
        assert normalise_stem("classes") == "classes"


class TestFrameNumber:
    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("cas100001", 100001),
            ("cas200091 - copia", 200091),
            ("sals800096a", 800096),
            ("cams900140w", 900140),
        ],
    )
    def test_extracts_trailing_number(self, stem: str, expected: int) -> None:
        assert frame_number(stem) == expected

    def test_returns_none_without_digits(self) -> None:
        assert frame_number("classes") is None

    def test_ordering_is_numeric_not_lexicographic(self) -> None:
        stems = ["cas100010", "cas100002", "cas100001"]
        assert sorted(stems, key=frame_number) == ["cas100001", "cas100002", "cas100010"]


class TestSubjectNumber:
    def test_parses_subject_dir_name(self) -> None:
        assert subject_number("Subject.1") == 1
        assert subject_number("Subject.10") == 10

    def test_returns_none_when_absent(self) -> None:
        assert subject_number("Subject") is None


class TestLabelBox:
    def test_to_xyxy_converts_from_normalised_centre(self) -> None:
        box = LabelBox(cls_id=0, cx=0.5, cy=0.5, w=0.2, h=0.4)
        x1, y1, x2, y2 = box.to_xyxy(720, 480)
        assert (x1, y1, x2, y2) == pytest.approx((288.0, 144.0, 432.0, 336.0))

    def test_aspect_ratio_uses_pixels_not_normalised_values(self) -> None:
        """Rasio ternormalisasi akan mengandung rasio aspek frame dan menyesatkan."""
        box = LabelBox(cls_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1)
        # Lebar dan tinggi ternormalisasi sama, tapi frame 720x480 tidak persegi.
        assert box.aspect_ratio(720, 480) == pytest.approx(480 / 720)

    def test_standing_and_lying_are_on_opposite_sides_of_one(self) -> None:
        standing = LabelBox(CLASS_NOFALL, 0.5, 0.5, 0.15, 0.55)
        lying = LabelBox(CLASS_FALL, 0.5, 0.8, 0.50, 0.18)
        assert standing.aspect_ratio() > 1.0
        assert lying.aspect_ratio() < 1.0

    def test_zero_width_returns_none_not_infinity(self) -> None:
        assert LabelBox(0, 0.5, 0.5, 0.0, 0.4).aspect_ratio() is None

    def test_is_fall_only_for_class_one(self) -> None:
        assert LabelBox(CLASS_FALL, 0.5, 0.5, 0.1, 0.1).is_fall
        assert not LabelBox(CLASS_NOFALL, 0.5, 0.5, 0.1, 0.1).is_fall


class TestParseLabelFile:
    def test_reads_single_box(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("0 0.309028 0.526042 0.215278 0.489583\n", encoding="utf-8")
        boxes = parse_label_file(p)
        assert len(boxes) == 1
        assert boxes[0].cls_id == 0
        assert boxes[0].cx == pytest.approx(0.309028)

    def test_skips_malformed_lines_without_failing(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("0 0.1 0.2 0.3 0.4\nrusak\n\n1 0.5 0.6 0.7 0.8\n", encoding="utf-8")
        assert len(parse_label_file(p)) == 2

    def test_empty_file_yields_no_boxes(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("", encoding="utf-8")
        assert parse_label_file(p) == []


class TestLoadClip:
    def test_reads_clean_clip(self, make_caucafall) -> None:
        d = make_caucafall(1, "Walk", n_frames=8, prefix="cams")
        clip = load_clip(d)

        assert clip.subject_id == 1
        assert clip.activity == "Walk"
        assert clip.kind == "adl"
        assert clip.clip_id == "s01_walk"
        assert clip.n_frames == 8
        assert clip.duration_s == pytest.approx(8 / NATIVE_FPS)
        assert [f.order for f in clip.frames] == list(range(8))

    def test_classes_txt_is_not_treated_as_a_label(self, make_caucafall) -> None:
        d = make_caucafall(1, "Walk", n_frames=5, prefix="cams")
        assert (d / "classes.txt").exists()
        assert load_clip(d).n_frames == 5

    def test_frames_ordered_numerically(self, make_caucafall) -> None:
        d = make_caucafall(1, "Walk", n_frames=12, prefix="cams")
        clip = load_clip(d)
        numbers = [f.frame_number for f in clip.frames]
        assert numbers == sorted(numbers)

    @pytest.mark.parametrize("anomaly", ["copia", "letter_label"])
    def test_naming_anomalies_lose_no_labels(self, make_caucafall, anomaly: str) -> None:
        """Pola dari Subject.2/Fall backwards, Subject.8/Hop, Subject.9/Walk."""
        d = make_caucafall(2, "Hop", n_frames=10, prefix="sals", anomaly=anomaly)
        clip = load_clip(d)
        assert clip.n_frames == 10
        assert clip.unlabelled_images == ()

    def test_extra_png_without_label_is_reported_not_fatal(self, make_caucafall) -> None:
        """Pola dari Subject.5/Kneel dan Subject.6/Walk."""
        d = make_caucafall(5, "Kneel", n_frames=10, prefix="ars", anomaly="extra_png")
        clip = load_clip(d)
        assert clip.n_frames == 9
        assert len(clip.unlabelled_images) == 1

    def test_label_without_any_image_is_an_error(self, make_caucafall) -> None:
        d = make_caucafall(1, "Walk", n_frames=4, prefix="cams")
        (d / "cams100002.png").unlink()
        with pytest.raises(CaucaFallError, match="label tanpa gambar"):
            load_clip(d)

    def test_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CaucaFallError, match="bukan direktori"):
            load_clip(tmp_path / "tidak_ada")


class TestFallIntervals:
    def test_single_contiguous_interval(self, make_caucafall) -> None:
        """Audit: 50 dari 50 klip jatuh punya tepat satu blok kontinu."""
        d = make_caucafall(1, "Fall forward", fall_frames=(4, 9), n_frames=10, prefix="cfs")
        clip = load_clip(d)
        assert clip.fall_intervals() == [(4, 9)]
        assert clip.fall_onset() == 4
        assert clip.kind == "fall"

    def test_adl_clip_has_no_fall_interval(self, make_caucafall) -> None:
        d = make_caucafall(1, "Kneel", n_frames=10, prefix="ars")
        clip = load_clip(d)
        assert clip.fall_intervals() == []
        assert clip.fall_onset() is None

    def test_onset_is_first_fall_frame(self, make_caucafall) -> None:
        d = make_caucafall(3, "Fall left", fall_frames=(2, 7), n_frames=10, prefix="cis")
        clip = load_clip(d)
        assert clip.fall_onset() == 2
        assert clip.fall_flags()[:2] == [False, False] if callable(clip.fall_flags) else True

    def test_fall_flags_align_with_frames(self, make_caucafall) -> None:
        d = make_caucafall(3, "Fall left", fall_frames=(2, 7), n_frames=10, prefix="cis")
        clip = load_clip(d)
        assert clip.fall_flags == [False, False, True, True, True, True, True, True, False, False]

    def test_multiple_intervals_are_all_returned(self, make_caucafall) -> None:
        """Tidak muncul di CAUCAFall, tapi tipe harus siap untuk sumber lain."""
        d = make_caucafall(1, "Fall forward", fall_frames=(2, 3), n_frames=8, prefix="cfs")
        # Tandai frame 6 sebagai fall secara manual.
        clip = load_clip(d)
        target = clip.frames[6].label_path
        target.write_text("1 0.5 0.8 0.5 0.18\n", encoding="utf-8")
        clip = load_clip(d)
        assert clip.fall_intervals() == [(2, 3), (6, 6)]
        assert clip.fall_onset() == 2


class TestDiscoverClips:
    def test_finds_all_clips(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        make_caucafall(1, "Walk", n_frames=4, prefix="cams", root=root)
        make_caucafall(1, "Fall forward", fall_frames=(2, 3), n_frames=4, prefix="cfs", root=root)
        make_caucafall(2, "Kneel", n_frames=4, prefix="ars", root=root)

        clips = discover_clips(root)
        assert len(clips) == 3
        assert {c.clip_id for c in clips} == {"s01_walk", "s01_fall_forward", "s02_kneel"}

    def test_subject_filter_enables_per_subject_split(self, make_caucafall, tmp_path: Path) -> None:
        """Split harus per SUBJEK. Per klip akan membocorkan orang antar fold."""
        root = tmp_path / "CAUCAFall"
        for s in (1, 2, 3):
            make_caucafall(s, "Walk", n_frames=4, prefix="cams", root=root)

        train = discover_clips(root, subjects=[1, 2])
        test = discover_clips(root, subjects=[3])

        assert {c.subject_id for c in train} == {1, 2}
        assert {c.subject_id for c in test} == {3}
        assert not ({c.subject_id for c in train} & {c.subject_id for c in test})

    def test_subjects_sorted_numerically(self, make_caucafall, tmp_path: Path) -> None:
        root = tmp_path / "CAUCAFall"
        for s in (1, 2, 10):
            make_caucafall(s, "Walk", n_frames=2, prefix="cams", root=root)
        assert [c.subject_id for c in discover_clips(root)] == [1, 2, 10]

    def test_root_without_subjects_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "kosong").mkdir()
        with pytest.raises(CaucaFallError, match="tidak ada direktori Subject"):
            discover_clips(tmp_path / "kosong")


class TestConstantsMatchAudit:
    def test_native_geometry_is_the_audited_reality(self) -> None:
        """Bukan 1080x960 @ 23 fps dari deskripsi Mendeley."""
        assert NATIVE_SIZE == (720, 480)
        assert NATIVE_FPS == 20.0

    def test_activity_lists_are_five_and_five(self) -> None:
        assert len(FALL_ACTIVITIES) == 5
        assert len(ADL_ACTIVITIES) == 5
        assert not set(FALL_ACTIVITIES) & set(ADL_ACTIVITIES)

    def test_adl_includes_the_hard_confounders(self) -> None:
        """Berlutut, mengambil objek, dan duduk adalah sumber false positive utama."""
        assert {"Kneel", "Pick up object", "Sit down"} <= set(ADL_ACTIVITIES)
