"""Test pfv doctor: penilaian kelayakan sumber."""

from __future__ import annotations

from typing import List, Optional

import pytest

from pfv.doctor import (
    Sample,
    Severity,
    SourceMeta,
    Thresholds,
    assess,
    format_report,
)

H = 480


def person(height_frac: float = 0.45, aspect: float = 2.4, **kw) -> Sample:
    h = height_frac * H
    return Sample(
        detected=True, bbox=(100.0, 50.0, 100.0 + h / aspect, 50.0 + h),
        full_torso=kw.get("full_torso", True), has_ankle=kw.get("has_ankle", True),
        has_knee=kw.get("has_knee", True),
    )


def meta(fps: Optional[float] = 20.0, live: bool = False, measured: Optional[float] = None) -> SourceMeta:
    return SourceMeta(720, H, fps, measured, live, 100)


def good(n: int = 100) -> List[Sample]:
    return [person() for _ in range(n)]


def check(report, name: str):
    return next(c for c in report.checks if c.name == name)


class TestHealthySource:
    def test_good_source_is_ok(self) -> None:
        r = assess(good(), meta())
        assert r.verdict is Severity.OK
        assert all(c.severity is Severity.OK for c in r.checks)

    def test_metrics_are_recorded(self) -> None:
        m = assess(good(), meta()).metrics
        assert m["fps"] == 20.0
        assert m["detection_rate"] == 1.0
        assert m["person_height_frac_median"] == pytest.approx(0.45, abs=0.01)


class TestFrameRate:
    def test_below_minimum_fails(self) -> None:
        r = assess(good(), meta(fps=8.0))
        assert check(r, "frame rate").severity is Severity.FAIL
        assert r.verdict is Severity.FAIL

    def test_boundary_is_inclusive(self) -> None:
        assert check(assess(good(), meta(fps=15.0)), "frame rate").severity is Severity.OK

    def test_just_below_boundary_fails(self) -> None:
        assert check(assess(good(), meta(fps=14.9)), "frame rate").severity is Severity.FAIL

    def test_fps_check_is_labelled_as_measured(self) -> None:
        assert check(assess(good(), meta()), "frame rate").evidence == "terukur"

    def test_live_source_uses_measured_not_declared_fps(self) -> None:
        """Webcam dan RTSP sering jauh di bawah fps yang diklaim."""
        r = assess(good(), meta(fps=30.0, live=True, measured=9.0))
        c = check(r, "frame rate")
        assert c.severity is Severity.FAIL
        assert "terukur" in c.value

    def test_file_source_uses_declared_fps(self) -> None:
        r = assess(good(), meta(fps=25.0, live=False, measured=999.0))
        assert r.metrics["fps"] == 25.0

    def test_unknown_fps_is_a_warning_not_a_pass(self) -> None:
        r = assess(good(), meta(fps=None))
        assert check(r, "frame rate").severity is Severity.WARN

    def test_threshold_is_configurable(self) -> None:
        r = assess(good(), meta(fps=20.0), Thresholds(min_fps=25.0))
        assert check(r, "frame rate").severity is Severity.FAIL


class TestPersonSize:
    def test_small_person_warns(self) -> None:
        r = assess([person(0.08) for _ in range(100)], meta())
        assert check(r, "tinggi orang di frame").severity is Severity.WARN

    def test_boundary(self) -> None:
        r = assess([person(0.15) for _ in range(100)], meta())
        assert check(r, "tinggi orang di frame").severity is Severity.OK

    def test_heuristic_checks_are_labelled(self) -> None:
        r = assess([person(0.08) for _ in range(100)], meta())
        assert check(r, "tinggi orang di frame").evidence == "heuristik"


class TestDetectionAndKeypoints:
    def test_low_detection_rate_warns(self) -> None:
        samples = good(30) + [Sample(detected=False)] * 70
        r = assess(samples, meta())
        assert check(r, "orang terdeteksi").severity is Severity.WARN

    def test_nobody_detected_stops_early_without_crashing(self) -> None:
        r = assess([Sample(detected=False)] * 100, meta())
        assert check(r, "orang terdeteksi").severity is Severity.WARN
        assert all(c.name != "tinggi orang di frame" for c in r.checks)

    def test_missing_ankles_warn(self) -> None:
        r = assess([person(has_ankle=False) for _ in range(100)], meta())
        assert check(r, "pergelangan kaki terdeteksi").severity is Severity.WARN

    def test_incomplete_torso_warns(self) -> None:
        r = assess([person(full_torso=False) for _ in range(100)], meta())
        assert check(r, "torso lengkap").severity is Severity.WARN

    def test_foreshortened_boxes_warn(self) -> None:
        """Kamera terlalu menunduk: kotak orang berdiri nyaris persegi."""
        r = assess([person(aspect=1.1) for _ in range(100)], meta())
        assert check(r, "rasio tinggi/lebar kotak").severity is Severity.WARN


class TestSampleSize:
    def test_too_few_samples_fail_and_stop(self) -> None:
        r = assess(good(5), meta())
        assert check(r, "jumlah sampel").severity is Severity.FAIL
        assert r.verdict is Severity.FAIL
        assert all(c.name != "torso lengkap" for c in r.checks)

    def test_empty(self) -> None:
        assert assess([], meta()).verdict is Severity.FAIL


class TestVerdict:
    def test_fail_outranks_warning(self) -> None:
        r = assess([person(0.05) for _ in range(100)], meta(fps=5.0))
        assert r.verdict is Severity.FAIL

    def test_warning_only(self) -> None:
        r = assess([person(0.05) for _ in range(100)], meta())
        assert r.verdict is Severity.WARN


class TestFormat:
    def test_report_text_contains_verdict(self) -> None:
        text = format_report(assess(good(), meta()))
        assert "PFV DOCTOR" in text and "Vonis: OK" in text

    def test_heuristic_warning_carries_the_caveat(self) -> None:
        text = format_report(assess([person(0.05) for _ in range(100)], meta()))
        assert "belum divalidasi" in text

    def test_measured_failure_does_not_carry_the_heuristic_caveat(self) -> None:
        text = format_report(assess(good(), meta(fps=5.0)))
        assert "belum divalidasi" not in text

    def test_no_warning_text_for_a_healthy_source(self) -> None:
        assert "belum divalidasi" not in format_report(assess(good(), meta()))
