"""Test klasifikasi postur dan state machine jatuh.

Fitur sintetis dengan nilai diketahui, tanpa model dan tanpa tracker. Skenario
diambil dari plan Task 6: jatuh, jongkok, berlutut, merosot pelan, dan jatuh lalu
bangun. Waktu selalu dalam detik, jadi semua skenario diuji pada beberapa fps.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pytest

from pfv.config import AppConfig, FallConfig, PostureConfig
from pfv.fall import (
    FallDecision,
    FallState,
    FallStateMachine,
    Posture,
    StaticFeatures,
    TemporalFeatures,
    classify_posture,
)

POSTURE = PostureConfig()


# ------------------------------------------------------------- fitur sintetis


def feats(**kw: Optional[float]) -> StaticFeatures:
    return StaticFeatures(**kw)


def upright() -> StaticFeatures:
    """Tegak: tidak ada indikator yang memilih tergeletak."""
    return feats(
        bbox_aspect_ratio=2.5, trunk_angle_deg=5.0, trunk_length_norm=0.29,
        hip_knee_dy_norm=0.23, hip_ankle_dy_norm=0.40, head_hip_dy_norm=0.45,
        keypoint_spread_y_norm=0.85,
    )


def lying_lateral() -> StaticFeatures:
    """Tergeletak menyamping: sudut, sebaran, dan kaki serempak (4 suara)."""
    return feats(
        bbox_aspect_ratio=0.4, trunk_angle_deg=80.0, trunk_length_norm=0.33,
        hip_knee_dy_norm=0.03, hip_ankle_dy_norm=0.05, head_hip_dy_norm=0.02,
        keypoint_spread_y_norm=0.20,
    )


def lying_axial() -> StaticFeatures:
    """Tergeletak searah kamera: sudut TIDAK bergerak, panjang trunk menyusut.

    Mereproduksi mode kegagalan terukur -- sudut trunk 5 derajat, persis seperti
    berdiri -- dan harus tetap terbaca tergeletak lewat indikator lain.
    """
    return feats(
        bbox_aspect_ratio=0.9, trunk_angle_deg=5.0, trunk_length_norm=0.22,
        hip_knee_dy_norm=0.15, hip_ankle_dy_norm=0.23, head_hip_dy_norm=0.34,
        keypoint_spread_y_norm=0.57,
    )


def bending() -> StaticFeatures:
    """Membungkuk mengambil objek: trunk memendek (SATU suara), pinggul tetap tinggi."""
    return feats(
        bbox_aspect_ratio=2.4, trunk_angle_deg=8.0, trunk_length_norm=0.15,
        hip_knee_dy_norm=0.23, hip_ankle_dy_norm=0.40, head_hip_dy_norm=0.04,
        keypoint_spread_y_norm=0.65,
    )


def unknown() -> StaticFeatures:
    """Hampir semua indikator tak tersedia, mis. tubuh tertutup meja."""
    return feats(bbox_aspect_ratio=1.5, trunk_angle_deg=10.0)


def sitting_on_floor() -> StaticFeatures:
    """Duduk di lantai: pinggul dekat kaki dan lutut, tetapi trunk tetap tegak.

    Memenuhi KEDUA indikator anggota badan (dua suara untuk satu fakta) tanpa satu
    pun indikator orientasi. Penyebab alarm palsu `sit_down` pada subjek latih.
    """
    return feats(
        bbox_aspect_ratio=1.4, trunk_angle_deg=6.0, trunk_length_norm=0.29,
        hip_knee_dy_norm=0.05, hip_ankle_dy_norm=0.10, head_hip_dy_norm=0.40,
        keypoint_spread_y_norm=0.70,
    )


def bending_deep() -> StaticFeatures:
    """Membungkuk dalam: SEMUA indikator orientasi memilih tergeletak.

    Pinggul tetap jauh di atas kaki, jadi kelompok anggota badan menolak. Penyebab
    alarm palsu `pick_up_object` pada subjek latih.
    """
    return feats(
        bbox_aspect_ratio=0.9, trunk_angle_deg=60.0, trunk_length_norm=0.15,
        hip_knee_dy_norm=0.22, hip_ankle_dy_norm=0.40, head_hip_dy_norm=0.10,
        keypoint_spread_y_norm=0.45,
    )


def fast() -> TemporalFeatures:
    return TemporalFeatures(hip_velocity_norm=1.0, contiguous=True, n_samples=10, window_s=0.5)


def slow() -> TemporalFeatures:
    return TemporalFeatures(hip_velocity_norm=0.0, contiguous=True, n_samples=10, window_s=0.5)


# --------------------------------------------------------------------- driver


class Driver:
    """Menyuapkan segmen ``(durasi_detik, fitur)`` ke mesin pada fps tertentu."""

    def __init__(self, cfg: Optional[FallConfig] = None, fps: float = 20.0) -> None:
        self.cfg = cfg or FallConfig()
        self.machine = FallStateMachine(self.cfg)
        self.fps = fps
        self.t = 0.0
        self.log: List[FallDecision] = []

    def run(
        self,
        duration_s: float,
        static: StaticFeatures,
        temporal: Optional[TemporalFeatures] = None,
    ) -> List[FallDecision]:
        out = []
        n = max(1, round(duration_s * self.fps))
        for _ in range(n):
            d = self.machine.update(self.t, static, temporal)
            out.append(d)
            self.log.append(d)
            self.t += 1.0 / self.fps
        return out

    @property
    def alerts(self) -> List[FallDecision]:
        return [d for d in self.log if d.alert]

    @property
    def states(self) -> set:
        return {d.state for d in self.log}


# ---------------------------------------------------------------------- postur


class TestClassifyPosture:
    def test_upright(self) -> None:
        r = classify_posture(upright(), POSTURE)
        assert r.posture is Posture.UPRIGHT
        assert r.n_lying == 0
        assert r.n_available == 5

    def test_lateral_lying(self) -> None:
        r = classify_posture(lying_lateral(), POSTURE)
        assert r.posture is Posture.LYING
        assert r.n_lying == 4

    def test_axial_lying_is_caught_despite_upright_trunk_angle(self) -> None:
        """Alasan keputusan memakai suara beberapa indikator, bukan satu sudut."""
        f = lying_axial()
        assert f.trunk_angle_deg < POSTURE.trunk_angle_min_deg  # sudut GAGAL
        r = classify_posture(f, POSTURE)
        assert r.posture is Posture.LYING
        assert r.votes["trunk_angle"] is False

    def test_bending_is_not_lying(self) -> None:
        r = classify_posture(bending(), POSTURE)
        assert r.n_orientation_votes == 1
        assert r.n_limb_votes == 0
        assert r.posture is Posture.UPRIGHT

    def test_sitting_on_floor_is_rejected_for_having_no_orientation_vote(self) -> None:
        """Dua suara anggota badan untuk satu fakta tidak boleh cukup."""
        r = classify_posture(sitting_on_floor(), POSTURE)
        assert r.n_limb_votes == 2
        assert r.n_orientation_votes == 0
        assert r.posture is Posture.UPRIGHT

    def test_deep_bending_is_rejected_for_having_no_limb_vote(self) -> None:
        """Ketiga indikator orientasi setuju, tetapi pinggul tetap tinggi di atas kaki."""
        r = classify_posture(bending_deep(), POSTURE)
        assert r.n_orientation_votes == 3
        assert r.n_limb_votes == 0
        assert r.posture is Posture.UPRIGHT

    def test_both_groups_are_required(self) -> None:
        r = classify_posture(lying_lateral(), POSTURE)
        assert r.n_orientation_votes >= 1 and r.n_limb_votes >= 1
        assert r.posture is Posture.LYING

    def test_orientation_only_is_unknown_when_limbs_are_hidden(self) -> None:
        """Tidak terbukti tergeletak bukan berarti terbukti tegak."""
        f = feats(trunk_angle_deg=85.0, keypoint_spread_y_norm=0.2, trunk_length_norm=0.3)
        r = classify_posture(f, POSTURE)
        assert r.n_orientation_votes == 2
        assert r.posture is Posture.UNKNOWN

    def test_limbs_only_is_unknown_when_orientation_is_hidden(self) -> None:
        f = feats(hip_ankle_dy_norm=0.05, hip_knee_dy_norm=0.03)
        assert classify_posture(f, POSTURE).posture is Posture.UNKNOWN

    def test_too_few_indicators_is_unknown_not_upright(self) -> None:
        r = classify_posture(unknown(), POSTURE)
        assert r.posture is Posture.UNKNOWN
        assert r.n_available < POSTURE.min_available_indicators

    def test_all_missing_is_unknown(self) -> None:
        assert classify_posture(feats(), POSTURE).posture is Posture.UNKNOWN

    def test_votes_report_none_for_missing(self) -> None:
        r = classify_posture(feats(trunk_angle_deg=5.0), POSTURE)
        assert r.votes["trunk_angle"] is False
        assert r.votes["hip_ankle"] is None

    def test_thresholds_are_inclusive(self) -> None:
        f = feats(
            trunk_angle_deg=POSTURE.trunk_angle_min_deg,
            keypoint_spread_y_norm=POSTURE.keypoint_spread_y_max,
            hip_ankle_dy_norm=0.9, hip_knee_dy_norm=0.9, trunk_length_norm=0.9,
        )
        assert classify_posture(f, POSTURE).n_lying == 2

    def test_limb_votes_are_configurable(self) -> None:
        strict = PostureConfig(min_limb_votes=2)
        # Searah kamera hanya memenuhi satu indikator anggota badan.
        assert classify_posture(lying_axial(), strict).posture is not Posture.LYING
        assert classify_posture(lying_lateral(), strict).posture is Posture.LYING

    def test_orientation_votes_are_configurable(self) -> None:
        strict = PostureConfig(min_orientation_votes=3)
        assert classify_posture(lying_lateral(), strict).posture is not Posture.LYING


# ------------------------------------------------------------------ jatuh

@pytest.mark.parametrize("fps", [10.0, 20.0, 30.0])
class TestFall:
    def test_upright_stays_upright_without_alert(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(20.0, upright(), slow())
        assert d.states == {FallState.UPRIGHT}
        assert d.alerts == []

    def test_fall_alerts_exactly_once_after_persistence(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(2.0, upright(), slow())
        d.run(8.0, lying_lateral(), slow())

        assert len(d.alerts) == 1
        alert = d.alerts[0]
        assert alert.state is FallState.FALLEN
        assert alert.previous_state is FallState.FALLING
        assert alert.reason == "persistent_lying"

        persistence = d.cfg.timing.persistence_s
        # onset_s = saat postur tergeletak pertama terbaca = acuan latensi.
        assert alert.onset_s == pytest.approx(2.0, abs=1.0 / fps + 1e-6)
        assert alert.timestamp_s - alert.onset_s == pytest.approx(persistence, abs=1.5 / fps)

    def test_camera_axis_fall_alerts_despite_upright_trunk_angle(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(1.0, upright(), slow())
        d.run(6.0, lying_axial(), slow())
        assert len(d.alerts) == 1

    def test_alert_only_on_the_entering_step(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(10.0, lying_lateral(), slow())
        assert len(d.alerts) == 1
        assert [x.state for x in d.log[-5:]] == [FallState.FALLEN] * 5

    def test_slow_collapse_without_fast_descent_still_alerts(self, fps: float) -> None:
        """Pingsan atau merosot pelan: kecepatan tidak boleh menjadi syarat."""
        d = Driver(fps=fps)
        d.run(2.0, upright(), slow())
        d.run(6.0, lying_lateral(), slow())
        assert len(d.alerts) == 1


@pytest.mark.parametrize("fps", [10.0, 20.0, 30.0])
class TestConfounders:
    def test_crouch_for_ten_seconds_never_alerts(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(10.0, bending(), slow())
        assert d.alerts == []
        assert d.states == {FallState.UPRIGHT}

    def test_brief_lying_then_upright_never_alerts(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(1.0, upright(), slow())
        d.run(1.0, lying_lateral(), slow())
        d.run(6.0, upright(), slow())
        assert d.alerts == []
        assert FallState.FALLING in d.states
        assert d.machine.state is FallState.UPRIGHT

    def test_fast_descent_alone_does_not_alert(self, fps: float) -> None:
        """Kecepatan hanya petunjuk: mempercepat FALLING, bukan syarat FALLEN."""
        d = Driver(fps=fps)
        d.run(1.0, upright(), slow())
        d.run(0.5, upright(), fast())
        d.run(6.0, upright(), slow())
        assert d.alerts == []
        assert FallState.FALLING in d.states
        assert d.machine.state is FallState.UPRIGHT

    def test_falling_times_out_without_new_trigger(self, fps: float) -> None:
        d = Driver(fps=fps)
        d.run(0.3, upright(), fast())
        out = d.run(3.0, upright(), slow())
        assert any(x.reason == "falling_timeout" for x in out)

    def test_mostly_occluded_person_is_not_declared_fallen(self, fps: float) -> None:
        """Satu detik terbaca tergeletak lalu tertutup lama: bukti terlalu tipis."""
        d = Driver(fps=fps)
        d.run(1.0, lying_lateral(), slow())
        out = d.run(8.0, unknown(), slow())
        assert d.alerts == []
        assert any(x.reason == "persistence_evidence_weak" for x in out)


class TestPersistenceTolerance:
    def test_short_upright_blip_does_not_reset_persistence(self) -> None:
        d = Driver()
        d.run(1.0, lying_lateral(), slow())
        d.run(0.2, upright(), slow())  # derau keypoint, di bawah grace_s
        d.run(5.0, lying_lateral(), slow())
        assert len(d.alerts) == 1

    def test_long_upright_interruption_resets_persistence(self) -> None:
        d = Driver()
        d.run(2.0, lying_lateral(), slow())
        d.run(1.0, upright(), slow())  # melewati grace_s 0,5
        out = d.run(1.5, lying_lateral(), slow())
        # Hitungan dimulai ulang, jadi 1,5 detik belum cukup.
        assert d.alerts == []
        assert out[-1].lying_for_s < d.cfg.timing.persistence_s

    def test_flickering_posture_fails_the_lying_fraction(self) -> None:
        """Bergantian tergeletak dan tegak terus-menerus bukan jatuh yang bertahan."""
        d = Driver()
        for _ in range(100):
            d.run(0.05, lying_lateral(), slow())
            d.run(0.05, upright(), slow())
        assert d.alerts == []


class TestOcclusion:
    def test_fallen_person_stays_fallen_through_long_occlusion(self) -> None:
        """Orang yang tertutup bukan orang yang sudah bangun."""
        d = Driver()
        d.run(6.0, lying_lateral(), slow())
        d.run(30.0, unknown(), slow())
        assert d.machine.state is FallState.FALLEN
        assert len(d.alerts) == 1


class TestRecovery:
    def test_get_up_recovers_without_second_alert(self) -> None:
        d = Driver()
        d.run(6.0, lying_lateral(), slow())
        d.run(4.0, upright(), slow())

        assert len(d.alerts) == 1
        assert FallState.RECOVERED in d.states
        assert d.machine.state is FallState.UPRIGHT

    def test_recovered_returns_to_upright_after_hold(self) -> None:
        d = Driver()
        d.run(6.0, lying_lateral(), slow())
        d.run(1.6, upright(), slow())
        assert d.machine.state is FallState.RECOVERED
        d.run(1.5, upright(), slow())
        assert d.machine.state is FallState.UPRIGHT

    def test_relapse_during_recovered_goes_back_to_falling(self) -> None:
        d = Driver()
        d.run(6.0, lying_lateral(), slow())
        d.run(1.6, upright(), slow())
        assert d.machine.state is FallState.RECOVERED
        out = d.run(0.1, lying_lateral(), slow())
        assert out[0].reason == "relapse"
        assert d.machine.state is FallState.FALLING

    def test_second_fall_inside_cooldown_is_suppressed(self) -> None:
        d = Driver()
        d.run(6.0, lying_lateral(), slow())
        d.run(4.0, upright(), slow())
        out = d.run(6.0, lying_lateral(), slow())

        assert len(d.alerts) == 1
        assert any(x.reason == "cooldown_suppressed" for x in out)
        assert d.machine.state is FallState.FALLEN

    def test_second_fall_after_cooldown_alerts_again(self) -> None:
        cfg = FallConfig(timing={"cooldown_s": 5.0})
        d = Driver(cfg)
        d.run(6.0, lying_lateral(), slow())
        d.run(4.0, upright(), slow())
        d.run(8.0, upright(), slow())
        d.run(6.0, lying_lateral(), slow())
        assert len(d.alerts) == 2


class TestVelocityHint:
    def test_fast_descent_enters_falling_with_reason(self) -> None:
        d = Driver()
        out = d.run(0.1, upright(), fast())
        assert out[0].state is FallState.FALLING
        assert out[0].reason == "fast_descent"

    def test_disabled_velocity_never_triggers(self) -> None:
        cfg = FallConfig(timing={"falling_velocity": None})
        d = Driver(cfg)
        d.run(2.0, upright(), fast())
        assert d.states == {FallState.UPRIGHT}

    def test_velocity_across_a_detection_gap_is_ignored(self) -> None:
        gap = TemporalFeatures(hip_velocity_norm=5.0, contiguous=False, n_samples=10)
        d = Driver()
        d.run(2.0, upright(), gap)
        assert d.states == {FallState.UPRIGHT}

    def test_upward_motion_never_triggers(self) -> None:
        rising = TemporalFeatures(hip_velocity_norm=-3.0, contiguous=True, n_samples=10)
        d = Driver()
        d.run(2.0, upright(), rising)
        assert d.states == {FallState.UPRIGHT}


class TestDecisionFields:
    def test_changed_flag_and_previous_state(self) -> None:
        d = Driver()
        first = d.run(0.05, upright(), slow())[0]
        assert first.changed is False
        entering = d.run(0.05, lying_lateral(), slow())[0]
        assert entering.changed is True
        assert entering.previous_state is FallState.UPRIGHT
        assert entering.state is FallState.FALLING

    def test_lying_for_grows_then_is_zero_when_not_lying(self) -> None:
        d = Driver()
        out = d.run(1.0, lying_lateral(), slow())
        assert out[-1].lying_for_s > out[0].lying_for_s
        assert d.run(0.1, upright(), slow())[0].lying_for_s > 0
        assert Driver().run(0.1, upright(), slow())[0].lying_for_s == 0.0


class TestTimeHandling:
    def test_backwards_timestamp_is_rejected(self) -> None:
        m = FallStateMachine(FallConfig())
        m.update(1.0, upright())
        with pytest.raises(ValueError, match="mundur"):
            m.update(0.5, upright())

    def test_equal_timestamps_are_allowed(self) -> None:
        m = FallStateMachine(FallConfig())
        m.update(1.0, upright())
        m.update(1.0, upright())

    def test_alert_time_is_fps_independent(self) -> None:
        times = []
        for fps in (10.0, 20.0, 30.0):
            d = Driver(fps=fps)
            d.run(1.0, upright(), slow())
            d.run(8.0, lying_lateral(), slow())
            times.append(d.alerts[0].timestamp_s)
        assert max(times) - min(times) < 0.2


class TestConfigIntegration:
    def test_shipped_yaml_matches_code_defaults(self, repo_root) -> None:
        shipped = AppConfig.from_yaml(repo_root / "config" / "default.yaml").fall
        assert shipped.posture == PostureConfig()
        assert shipped.timing == FallConfig().timing

    def test_unknown_timing_key_rejected(self) -> None:
        with pytest.raises(Exception):
            FallConfig(timing={"persistance_s": 3.0})

    def test_default_persistence_is_within_planned_range(self) -> None:
        assert 3.0 <= FallConfig().timing.persistence_s <= 5.0
