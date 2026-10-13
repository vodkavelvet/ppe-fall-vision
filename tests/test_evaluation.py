"""Test harness evaluasi Task 7b: metrik, simulasi, baseline, seleksi, gerbang, kunci.

Semua data sintetis. Ada satu uji gladi bersih yang menjalankan ``evaluate_fall.py``
sampai selesai pada subjek uji PALSU, karena skrip itu hanya boleh dijalankan sekali
pada data sungguhan dan bug di dalamnya tidak boleh baru ketahuan setelah angka uji
terlihat.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from pfv.config import FallConfig
from pfv.evaluation import baselines
from pfv.evaluation.cache import ClipData, TrackSeq
from pfv.evaluation.gate import (
    Gate,
    TestRunLockedError,
    check_lock,
    file_hash,
    judge,
    load_gate,
    read_runs,
    record_run,
)
from pfv.evaluation.metrics import (
    DETECTION_MARGIN_S,
    ClipOutcome,
    Proportion,
    summarize,
    wilson_interval,
)
from pfv.evaluation.search import (
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VALIDATION_SUBJECTS,
    HeldOutAccessError,
    Spec,
    assert_no_test_subjects,
    build,
    candidates,
    fingerprint,
    load_clips,
    select,
    spec_from_json,
    spec_to_json,
)
from pfv.evaluation.simulate import simulate_clip
from pfv.fall import compute_static
from pfv.fall.features import StaticFeatures, TemporalFeatures
from pfv.fall.posture import Posture
from pfv.perception.base import PersonPose
from pfv_test_helpers import (
    lying_bbox,
    lying_keypoints,
    standing_bbox,
    standing_keypoints,
)

FPS = 20.0
BASE = FallConfig()


def outcome(
    *, fall: bool = True, alerted: bool = True, axis: bool = False,
    lying_s: float = 7.0, latency: Optional[float] = 3.0, n_alerts: int = 1, name: str = "x",
) -> ClipOutcome:
    n = n_alerts if alerted else 0
    return ClipOutcome(
        name=name, subject=1, activity="fall forward" if axis else "fall left",
        is_fall=fall, is_camera_axis=axis, lying_s=lying_s, n_alerts=n,
        first_alert_s=5.0 if n else None, latency_s=latency if n and fall else None,
    )


# ----------------------------------------------------------------------- Wilson


class TestWilsonInterval:
    def test_zero_of_ten(self) -> None:
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0
        assert hi == pytest.approx(0.2775, abs=0.001)

    def test_ten_of_ten(self) -> None:
        lo, hi = wilson_interval(10, 10)
        assert hi == 1.0
        assert lo == pytest.approx(0.7225, abs=0.001)

    def test_half(self) -> None:
        lo, hi = wilson_interval(5, 10)
        assert lo == pytest.approx(0.2366, abs=0.001)
        assert hi == pytest.approx(0.7634, abs=0.001)

    def test_empty_sample(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_stays_inside_unit_interval(self) -> None:
        for n in (1, 2, 5, 20, 100):
            for k in range(n + 1):
                lo, hi = wilson_interval(k, n)
                assert 0.0 <= lo <= k / n <= hi <= 1.0

    def test_narrower_with_more_data(self) -> None:
        small = wilson_interval(10, 20)
        large = wilson_interval(100, 200)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_twenty_clips_is_genuinely_uncertain(self) -> None:
        """Alasan interval selalu dilaporkan: 18/20 masih mencakup di bawah 70 persen."""
        lo, _ = wilson_interval(18, 20)
        assert lo < 0.70


class TestProportion:
    def test_format(self) -> None:
        assert Proportion(18, 20).format().startswith("18/20 = 90% [")

    def test_empty(self) -> None:
        assert Proportion(0, 0).format() == "-"
        assert np.isnan(Proportion(0, 0).value)


# --------------------------------------------------------------------- summarize


class TestSummarize:
    def test_recall_and_false_alarms(self) -> None:
        outs = [outcome(), outcome(), outcome(alerted=False),
                outcome(fall=False, alerted=False), outcome(fall=False, alerted=True)]
        s = summarize(outs, 3.0)
        assert (s.recall.k, s.recall.n) == (2, 3)
        assert (s.false_alarm_clips.k, s.false_alarm_clips.n) == (1, 2)

    def test_achievable_excludes_structurally_impossible_clips(self) -> None:
        """Periode tergeletak 2,15 detik tidak mungkin menghasilkan alert pada 3 detik."""
        outs = [outcome(lying_s=2.15, alerted=False), outcome(lying_s=7.0),
                outcome(lying_s=7.0, alerted=False)]
        s = summarize(outs, 3.0)
        assert (s.recall.k, s.recall.n) == (1, 3)
        assert (s.achievable_recall.k, s.achievable_recall.n) == (1, 2)

    def test_achievable_boundary(self) -> None:
        o = outcome(lying_s=3.0 + DETECTION_MARGIN_S)
        assert o.achievable(3.0)
        assert not outcome(lying_s=3.0 + DETECTION_MARGIN_S - 0.01).achievable(3.0)

    def test_camera_axis_and_lateral_are_split(self) -> None:
        outs = [outcome(axis=True), outcome(axis=True, alerted=False),
                outcome(axis=False), outcome(axis=False), outcome(axis=False)]
        s = summarize(outs, 3.0)
        assert (s.camera_axis_recall.k, s.camera_axis_recall.n) == (1, 2)
        assert (s.lateral_recall.k, s.lateral_recall.n) == (3, 3)

    def test_multi_alert_clips_counted_only_among_falls(self) -> None:
        outs = [outcome(n_alerts=2), outcome(), outcome(fall=False, n_alerts=3)]
        assert summarize(outs, 3.0).multi_alert_clips.k == 1

    def test_latency_statistics(self) -> None:
        outs = [outcome(latency=v) for v in (2.0, 3.0, 4.0, 5.0, 10.0)]
        s = summarize(outs, 3.0)
        assert s.latency_median_s == 4.0
        assert s.latency_max_s == 10.0
        assert s.latency_p90_s == 10.0
        assert s.n_latencies == 5

    def test_missed_clips_do_not_contribute_latency(self) -> None:
        outs = [outcome(latency=2.0), outcome(alerted=False)]
        assert summarize(outs, 3.0).n_latencies == 1

    def test_no_alerts_means_no_latency(self) -> None:
        s = summarize([outcome(alerted=False)], 3.0)
        assert s.latency_median_s is None

    def test_false_alarms_never_contribute_latency(self) -> None:
        s = summarize([outcome(fall=False, alerted=True)], 3.0)
        assert s.n_latencies == 0

    def test_empty(self) -> None:
        s = summarize([], 3.0)
        assert s.recall.n == 0 and s.false_alarm_clips.n == 0


# ------------------------------------------------------------------- simulasi


def track_from(poses: List[PersonPose], t0: float = 0.0, track_id: int = 1) -> TrackSeq:
    seq = TrackSeq(track_id)
    for i, pose in enumerate(poses):
        seq.times.append(t0 + i / FPS)
        seq.static.append(compute_static(pose))
        seq.temporal.append(TemporalFeatures(hip_velocity_norm=0.0, contiguous=True, n_samples=10))
    return seq


def standing_poses(n: int) -> List[PersonPose]:
    return [PersonPose(standing_bbox(), standing_keypoints(), 0.9)] * n


def lying_poses(n: int) -> List[PersonPose]:
    return [PersonPose(lying_bbox(), lying_keypoints(), 0.9)] * n


def clip(name: str, activity: str, tracks: List[TrackSeq], *, onset: Optional[float] = None,
         lying: float = 0.0, subject: int = 1, duration: float = 14.0) -> ClipData:
    return ClipData(name=name, subject=subject, activity=activity, duration_s=duration,
                    onset_s=onset, lying_s=lying, tracks=tracks)


def fall_clip(name: str = "s01_fall_left", activity: str = "fall left", subject: int = 1) -> ClipData:
    return clip(name, activity,
                [track_from(standing_poses(40) + lying_poses(160))],
                onset=2.0, lying=8.0, subject=subject)


def walk_clip(name: str = "s01_walk", subject: int = 1) -> ClipData:
    return clip(name, "walk", [track_from(standing_poses(200))], subject=subject)


class TestSimulate:
    def test_fall_alerts_with_latency_near_persistence(self) -> None:
        out = simulate_clip(fall_clip(), BASE)
        assert out.alerted and out.n_alerts == 1
        assert out.latency_s == pytest.approx(BASE.timing.persistence_s, abs=0.2)

    def test_walk_never_alerts(self) -> None:
        out = simulate_clip(walk_clip(), BASE)
        assert not out.alerted and out.latency_s is None

    def test_outcome_carries_labels(self) -> None:
        out = simulate_clip(fall_clip(activity="fall forward"), BASE)
        assert out.is_fall and out.is_camera_axis
        assert out.lying_s == 8.0

    def test_each_track_gets_its_own_machine(self) -> None:
        """Dua track yang sama-sama tergeletak menghasilkan dua alert, bukan satu."""
        c = clip("s01_fall_left", "fall left",
                 [track_from(lying_poses(160), track_id=1),
                  track_from(lying_poses(160), track_id=2)], onset=0.0, lying=8.0)
        assert simulate_clip(c, BASE).n_alerts == 2

    def test_posture_fn_is_injected(self) -> None:
        never = baselines.aspect_baseline(0.0)
        assert not simulate_clip(fall_clip(), BASE, never).alerted

    def test_empty_clip(self) -> None:
        out = simulate_clip(clip("s01_walk", "walk", []), BASE)
        assert not out.alerted


# -------------------------------------------------------------------- baseline


def feats(**kw: Optional[float]) -> StaticFeatures:
    return StaticFeatures(**kw)


class TestBaselines:
    cfg = BASE.posture

    def test_aspect(self) -> None:
        fn = baselines.aspect_baseline(1.0)
        assert fn(feats(bbox_aspect_ratio=0.5), self.cfg).posture is Posture.LYING
        assert fn(feats(bbox_aspect_ratio=2.0), self.cfg).posture is Posture.UPRIGHT
        assert fn(feats(), self.cfg).posture is Posture.UNKNOWN

    def test_aspect_threshold_is_inclusive(self) -> None:
        assert baselines.aspect_baseline(1.0)(feats(bbox_aspect_ratio=1.0), self.cfg).posture is Posture.LYING

    def test_angle(self) -> None:
        fn = baselines.angle_baseline(45.0)
        assert fn(feats(trunk_angle_deg=80.0), self.cfg).posture is Posture.LYING
        assert fn(feats(trunk_angle_deg=10.0), self.cfg).posture is Posture.UPRIGHT
        assert fn(feats(), self.cfg).posture is Posture.UNKNOWN

    def test_angle_is_blind_to_camera_axis_fall(self) -> None:
        """Mode kegagalan yang menjadi inti klaim: sudut 5 derajat untuk jatuh searah kamera."""
        fn = baselines.angle_baseline(45.0)
        assert fn(feats(trunk_angle_deg=5.0), self.cfg).posture is Posture.UPRIGHT

    def test_angle_or_aspect(self) -> None:
        fn = baselines.angle_or_aspect_baseline(45.0, 1.0)
        assert fn(feats(trunk_angle_deg=80.0, bbox_aspect_ratio=3.0), self.cfg).posture is Posture.LYING
        assert fn(feats(trunk_angle_deg=5.0, bbox_aspect_ratio=0.5), self.cfg).posture is Posture.LYING
        assert fn(feats(trunk_angle_deg=5.0, bbox_aspect_ratio=3.0), self.cfg).posture is Posture.UPRIGHT
        assert fn(feats(trunk_angle_deg=5.0), self.cfg).posture is Posture.UPRIGHT
        assert fn(feats(), self.cfg).posture is Posture.UNKNOWN

    def test_orientation_only_accepts_deep_bending_which_the_full_model_rejects(self) -> None:
        """Inilah sumbangan kelompok anggota badan, diukur sebagai ablasi."""
        from pfv.fall.posture import classify_posture

        bending = feats(trunk_angle_deg=60.0, trunk_length_norm=0.15, keypoint_spread_y_norm=0.45,
                        hip_ankle_dy_norm=0.40, hip_knee_dy_norm=0.22, bbox_aspect_ratio=0.9)
        assert baselines.orientation_only(1)(bending, self.cfg).posture is Posture.LYING
        assert classify_posture(bending, self.cfg).posture is Posture.UPRIGHT

    def test_orientation_only_min_votes(self) -> None:
        f = feats(trunk_angle_deg=60.0, trunk_length_norm=0.5, keypoint_spread_y_norm=0.9)
        assert baselines.orientation_only(1)(f, self.cfg).posture is Posture.LYING
        assert baselines.orientation_only(2)(f, self.cfg).posture is Posture.UPRIGHT

    def test_every_model_has_a_label(self) -> None:
        assert set(baselines.MODEL_LABELS) == set(baselines.MODEL_ORDER)
        assert baselines.MODEL_ORDER[-1] == "full"


# ------------------------------------------------------------------- kandidat


class TestCandidates:
    @pytest.mark.parametrize("model, size", [
        ("full", 432), ("orientation", 96), ("aspect", 7), ("angle", 6), ("angle_or_aspect", 42),
    ])
    def test_grid_sizes(self, model: str, size: int) -> None:
        assert len(candidates(model)) == size

    @pytest.mark.parametrize("model", baselines.MODEL_ORDER)
    def test_candidates_are_unique_and_deterministic(self, model: str) -> None:
        a, b = candidates(model), candidates(model)
        assert a == b
        assert len(set(a)) == len(a)

    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            candidates("tidak_ada")

    def test_every_model_in_order_has_candidates(self) -> None:
        for m in baselines.MODEL_ORDER:
            assert candidates(m)

    def test_full_params_are_applied_to_config(self) -> None:
        spec = candidates("full")[0]
        cfg, fn = build(spec, BASE)
        assert fn is None
        for k, v in spec.as_dict.items():
            assert getattr(cfg.posture, k) == v

    def test_build_does_not_mutate_the_base_config(self) -> None:
        before = BASE.posture.trunk_angle_min_deg
        build(candidates("full")[-1], BASE)
        assert BASE.posture.trunk_angle_min_deg == before

    @pytest.mark.parametrize("model", ["aspect", "angle", "angle_or_aspect", "orientation"])
    def test_baselines_get_a_posture_function(self, model: str) -> None:
        _, fn = build(candidates(model)[0], BASE)
        assert callable(fn)

    def test_persistence_is_never_part_of_a_candidate(self) -> None:
        for m in baselines.MODEL_ORDER:
            for spec in candidates(m):
                assert "persistence_s" not in spec.as_dict

    def test_spec_json_round_trip(self) -> None:
        for m in baselines.MODEL_ORDER:
            spec = candidates(m)[3 % len(candidates(m))]
            assert spec_from_json(json.loads(json.dumps(spec_to_json(spec)))) == spec


# --------------------------------------------------------------------- seleksi


class TestSelect:
    def _train(self) -> List[ClipData]:
        return [fall_clip("s01_fall_left"), fall_clip("s02_fall_left", subject=2),
                walk_clip("s01_walk"), walk_clip("s02_walk", 2), walk_clip("s03_walk", 3)]

    def test_candidate_with_too_many_false_alarms_is_dropped(self) -> None:
        # max_ratio=100 menandai SEMUA frame tergeletak -> alarm palsu di semua ADL.
        always = Spec("aspect", (("max_ratio", 100.0),))
        sane = Spec("aspect", (("max_ratio", 1.0),))
        chosen, _, n_eligible = select([always, sane], self._train(), [walk_clip("s07_walk", 7)], BASE)
        assert chosen.spec == sane
        assert n_eligible == 1

    def test_recall_decides_among_eligible(self) -> None:
        never = Spec("aspect", (("max_ratio", 0.0),))
        sane = Spec("aspect", (("max_ratio", 1.0),))
        chosen, _, _ = select([never, sane], self._train(), [walk_clip("s07_walk", 7)], BASE)
        assert chosen.spec == sane

    def test_validation_breaks_ties_by_false_alarms_first(self) -> None:
        # Pada data latih dua spesifikasi ini sama persis; validasi yang membedakan.
        a = Spec("aspect", (("max_ratio", 1.0),))
        b = Spec("aspect", (("max_ratio", 1.1),))
        val = [fall_clip("s07_fall_left", subject=7), walk_clip("s07_walk", 7)]
        chosen, ranked, _ = select([a, b], self._train(), val, BASE)
        assert len(ranked) == 2
        assert chosen.validation.false_alarm_clips.k <= ranked[-1].validation.false_alarm_clips.k

    def test_falls_back_to_fewest_false_alarms_when_none_eligible(self) -> None:
        always = Spec("aspect", (("max_ratio", 100.0),))
        chosen, _, n_eligible = select([always], self._train(), [walk_clip("s07_walk", 7)], BASE)
        assert chosen.spec == always
        assert n_eligible == 1

    def test_selection_is_deterministic(self) -> None:
        specs = candidates("aspect")
        val = [fall_clip("s07_fall_left", subject=7), walk_clip("s07_walk", 7)]
        a = select(specs, self._train(), val, BASE)[0].spec
        b = select(specs, self._train(), val, BASE)[0].spec
        assert a == b

    def test_selection_never_looks_at_test_subjects(self) -> None:
        """select() recebe apenas train/validation; sujeitos de teste nao fazem parte da assinatura."""
        import inspect

        assert list(inspect.signature(select).parameters) == ["specs", "train", "validation", "base"]


# ---------------------------------------------------------------- penjaga uji


class TestHeldOutGuard:
    def test_split_is_disjoint_and_complete(self) -> None:
        sets = [set(TRAIN_SUBJECTS), set(VALIDATION_SUBJECTS), set(TEST_SUBJECTS)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert set().union(*sets) == set(range(1, 11))

    def test_assert_rejects_test_subjects(self) -> None:
        with pytest.raises(HeldOutAccessError, match="subjek uji"):
            assert_no_test_subjects([1, 9])

    def test_assert_accepts_train_and_validation(self) -> None:
        assert_no_test_subjects(TRAIN_SUBJECTS + VALIDATION_SUBJECTS)

    def test_load_clips_refuses_test_subjects_without_reading_anything(self, tmp_path: Path) -> None:
        with pytest.raises(HeldOutAccessError):
            load_clips(tmp_path, [10], None)  # type: ignore[arg-type]

    def test_load_clips_does_not_open_test_files(self, tmp_path: Path, monkeypatch) -> None:
        write_npz(tmp_path / "s09_walk.npz", fall=False)
        write_npz(tmp_path / "s01_walk.npz", fall=False)
        from pfv.config import load_config

        opened: List[str] = []
        import pfv.evaluation.search as search

        real = search.build_clip_data
        monkeypatch.setattr(search, "build_clip_data", lambda p, c: (opened.append(p.stem), real(p, c))[1])
        clips = load_clips(tmp_path, [1], load_config())
        assert opened == ["s01_walk"]
        assert [c.subject for c in clips] == [1]


# ---------------------------------------------------------------------- gerbang


class TestGate:
    def test_shipped_gate_has_the_documented_values(self, repo_root: Path) -> None:
        g = load_gate(repo_root / "eval" / "gate.yaml")
        assert g == Gate(0.85, 0.10, 5.0)

    def test_missing_criterion_is_an_error(self, tmp_path: Path) -> None:
        p = tmp_path / "g.yaml"
        p.write_text("achievable_recall_min: 0.9\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hilang"):
            load_gate(p)

    def _summary(self, hits: int, n: int, fa: int, n_adl: int, lat: float = 3.0):
        outs = ([outcome(latency=lat) for _ in range(hits)]
                + [outcome(alerted=False) for _ in range(n - hits)]
                + [outcome(fall=False, alerted=True) for _ in range(fa)]
                + [outcome(fall=False, alerted=False) for _ in range(n_adl - fa)])
        return summarize(outs, 3.0)

    def test_passes_when_every_criterion_is_met(self) -> None:
        v = judge(self._summary(18, 20, 1, 20), Gate(0.85, 0.10, 5.0))
        assert v.passed and v.label == "LOLOS"

    @pytest.mark.parametrize("hits, fa, lat", [(16, 1, 3.0), (18, 3, 3.0), (18, 1, 6.0)])
    def test_fails_on_any_single_criterion(self, hits: int, fa: int, lat: float) -> None:
        v = judge(self._summary(hits, 20, fa, 20, lat), Gate(0.85, 0.10, 5.0))
        assert not v.passed and v.label == "TIDAK LOLOS"
        assert sum(not c.passed for c in v.criteria) == 1

    def test_boundaries_are_inclusive(self) -> None:
        v = judge(self._summary(17, 20, 2, 20, 5.0), Gate(0.85, 0.10, 5.0))
        assert v.passed

    def test_no_data_fails_rather_than_passing_vacuously(self) -> None:
        v = judge(summarize([], 3.0), Gate(0.85, 0.10, 5.0))
        assert not v.passed

    def test_no_latency_fails(self) -> None:
        v = judge(self._summary(0, 20, 0, 20), Gate(0.0, 1.0, 5.0))
        assert not v.criteria[2].passed


# ------------------------------------------------------------------------ kunci


class TestTestRunLock:
    def test_first_run_is_allowed(self, tmp_path: Path) -> None:
        assert check_lock(tmp_path / "runs.jsonl", "abc", allow_rerun=False) is None

    def test_second_run_is_refused(self, tmp_path: Path) -> None:
        log = tmp_path / "runs.jsonl"
        record_run(log, fingerprint="abc", gate_hash="g", rerun=False, verdict="LOLOS")
        with pytest.raises(TestRunLockedError, match="sudah dievaluasi"):
            check_lock(log, "abc", allow_rerun=False)

    def test_rerun_requires_explicit_permission_and_is_marked(self, tmp_path: Path) -> None:
        log = tmp_path / "runs.jsonl"
        record_run(log, fingerprint="abc", gate_hash="g", rerun=False, verdict="LOLOS")
        assert check_lock(log, "abc", allow_rerun=True) is not None
        record_run(log, fingerprint="abc", gate_hash="g", rerun=True, verdict="LOLOS")
        assert [r["rerun"] for r in read_runs(log)] == [False, True]

    def test_a_different_frozen_config_is_not_locked(self, tmp_path: Path) -> None:
        log = tmp_path / "runs.jsonl"
        record_run(log, fingerprint="abc", gate_hash="g", rerun=False, verdict="LOLOS")
        assert check_lock(log, "xyz", allow_rerun=False) is None

    def test_a_crashed_run_still_locks(self, tmp_path: Path) -> None:
        """Dicatat sebelum hasil dicetak, jadi skrip yang macet tidak bisa diulang diam-diam."""
        log = tmp_path / "runs.jsonl"
        record_run(log, fingerprint="abc", gate_hash="g", rerun=False, verdict=None)
        with pytest.raises(TestRunLockedError):
            check_lock(log, "abc", allow_rerun=False)

    def test_log_is_append_only(self, tmp_path: Path) -> None:
        log = tmp_path / "runs.jsonl"
        for i in range(3):
            record_run(log, fingerprint=f"f{i}", gate_hash="g", rerun=False, verdict=None)
        assert len(read_runs(log)) == 3

    def test_file_hash_changes_with_content(self, tmp_path: Path) -> None:
        p = tmp_path / "g.yaml"
        p.write_text("a", encoding="utf-8")
        h1 = file_hash(p)
        p.write_text("b", encoding="utf-8")
        assert file_hash(p) != h1


class TestFingerprint:
    def test_changes_when_anything_changes(self) -> None:
        assert fingerprint({"a": 1}) != fingerprint({"a": 2})

    def test_independent_of_key_order(self) -> None:
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


# ----------------------------------------------------------- gladi bersih skrip


def write_npz(path: Path, *, fall: bool, n_standing: int = 40, n_lying: int = 160) -> None:
    """Tulis keypoint sintetis seperti hasil ``extract_pose_caucafall.py``."""
    frames = (n_standing + n_lying) if fall else 200
    keypoints = np.zeros((frames, 17, 3), dtype=np.float32)
    bboxes = np.zeros((frames, 4), dtype=np.float32)
    flags = np.zeros(frames, dtype=bool)
    for i in range(frames):
        lying = fall and i >= n_standing
        keypoints[i] = lying_keypoints() if lying else standing_keypoints()
        bboxes[i] = lying_bbox() if lying else standing_bbox()
        flags[i] = lying
    np.savez(
        path, keypoints=keypoints, bboxes=bboxes, confidence=np.full(frames, 0.9, np.float32),
        detected=np.ones(frames, bool), n_people=np.ones(frames, np.int16), fall_flags=flags,
    )


def import_script(root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def make_frozen(path: Path) -> str:
    """Berkas beku sintetis dengan spesifikasi pertama tiap model."""
    payload = {
        "persistence_s": BASE.timing.persistence_s,
        "models": {
            m: {"spec": spec_to_json(candidates(m)[0])} for m in baselines.MODEL_ORDER
        },
    }
    payload["fingerprint"] = fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["fingerprint"]


class TestEvaluateScriptDryRun:
    """Menjalankan evaluate_fall.py sampai selesai pada subjek uji PALSU."""

    @pytest.fixture
    def env(self, tmp_path: Path, repo_root: Path):
        pose = tmp_path / "pose"
        pose.mkdir()
        write_npz(pose / "s09_fall_left.npz", fall=True)
        write_npz(pose / "s09_walk.npz", fall=False)
        write_npz(pose / "s10_fall_forward.npz", fall=True)
        write_npz(pose / "s10_walk.npz", fall=False)
        write_npz(pose / "s01_fall_left.npz", fall=True)
        write_npz(pose / "s01_walk.npz", fall=False)
        frozen = tmp_path / "frozen.json"
        fp = make_frozen(frozen)
        argv = [
            "--pose-dir", str(pose), "--frozen", str(frozen),
            "--gate", str(repo_root / "eval" / "gate.yaml"),
            "--log", str(tmp_path / "runs.jsonl"), "--out", str(tmp_path / "res.json"),
        ]
        return importlib.import_module("types").SimpleNamespace(
            argv=argv, tmp=tmp_path, fp=fp, frozen=frozen,
            module=import_script(repo_root, "evaluate_fall"),
        )

    def test_completes_and_records_the_run(self, env, in_repo_root, capsys) -> None:
        assert env.module.main(env.argv) == 0
        out = capsys.readouterr().out
        assert "HASIL PADA SUBJEK UJI" in out
        assert "GERBANG" in out
        runs = read_runs(env.tmp / "runs.jsonl")
        assert runs[0]["verdict"] is None and runs[-1]["verdict"] in ("LOLOS", "TIDAK LOLOS")
        assert all(r["frozen_fingerprint"] == env.fp for r in runs)
        assert (env.tmp / "res.json").exists()

    def test_second_run_is_refused(self, env, in_repo_root, capsys) -> None:
        env.module.main(env.argv)
        capsys.readouterr()
        assert env.module.main(env.argv) == 2
        assert "DITOLAK" in capsys.readouterr().out

    def test_rerun_is_loudly_marked(self, env, in_repo_root, capsys) -> None:
        env.module.main(env.argv)
        capsys.readouterr()
        assert env.module.main(env.argv + ["--allow-rerun"]) == 0
        assert "ULANGAN" in capsys.readouterr().out
        assert read_runs(env.tmp / "runs.jsonl")[-1]["rerun"] is True

    def test_tampered_frozen_file_is_refused(self, env, in_repo_root) -> None:
        data = json.loads(env.frozen.read_text())
        data["persistence_s"] = 1.0
        env.frozen.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit, match="berubah"):
            env.module.main(env.argv)

    def test_a_run_that_crashes_after_loading_test_data_still_locks(
        self, env, in_repo_root, monkeypatch
    ) -> None:
        monkeypatch.setattr(env.module, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("macet")))
        with pytest.raises(RuntimeError):
            env.module.main(env.argv)
        assert env.module.main(env.argv) == 2

    def test_a_perfect_synthetic_model_passes_the_gate(self, env, in_repo_root, capsys) -> None:
        env.module.main(env.argv)
        out = capsys.readouterr().out
        assert "GERBANG (model penuh): LOLOS" in out


class TestShippedFrozenFile:
    """Mengunci berkas beku yang dihasilkan tune_fall.py, bila ada."""

    @pytest.fixture(autouse=True)
    def _needs(self, in_repo_root: Path) -> None:
        if not (in_repo_root / "eval" / "frozen_models.json").exists():
            pytest.skip("eval/frozen_models.json belum ada; jalankan scripts/tune_fall.py")

    def _load(self, repo_root: Path) -> dict:
        return json.loads((repo_root / "eval" / "frozen_models.json").read_text())

    def test_fingerprint_is_intact(self, repo_root: Path) -> None:
        data = self._load(repo_root)
        stored = data.pop("fingerprint")
        assert stored == fingerprint(data)

    def test_no_test_subject_clip_was_used_for_tuning(self, repo_root: Path) -> None:
        data = self._load(repo_root)
        used = data["train_clips"] + data["validation_clips"]
        assert used
        assert all(int(name[1:3]) not in TEST_SUBJECTS for name in used)

    def test_every_model_is_frozen_with_a_valid_spec(self, repo_root: Path) -> None:
        data = self._load(repo_root)
        assert set(data["models"]) == set(baselines.MODEL_ORDER)
        for model, entry in data["models"].items():
            spec = spec_from_json(entry["spec"])
            assert spec.model == model
            assert spec in candidates(model)

    def test_frozen_persistence_matches_config(self, repo_root: Path) -> None:
        assert self._load(repo_root)["persistence_s"] == BASE.timing.persistence_s
