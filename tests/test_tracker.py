"""Test PersonTracker dan TrackState.

Semua dengan pose sintetis, tanpa model. Yang dijaga: identitas stabil, riwayat
dibatasi WAKTU bukan jumlah frame, dan lubang deteksi bisa dikenali.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from pfv.config import AppConfig
from pfv.io.frames import Frame, FrameTransform
from pfv.perception.base import N_KEYPOINTS, Keypoint, PersonPose
from pfv.tracking import PersonTracker, TrackSample, TrackState
from pfv_test_helpers import make_keypoints, standing_keypoints

FPS = 20.0


def pose_at(x: float, conf: float = 0.9, **kp_overrides: float) -> PersonPose:
    """Orang berkotak 40x160 dengan pojok kiri-atas di (x, 100)."""
    kp = make_keypoints(conf=0.9, **kp_overrides)
    kp[:, 0] += x
    return PersonPose((x, 100.0, x + 40.0, 260.0), kp, conf)


def frame(i: int, fps: float = FPS) -> Frame:
    return Frame(
        index=i,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp_s=i / fps,
        transform=FrameTransform(),
        source_size=(720, 480),
    )


class TestIdentity:
    def test_single_person_keeps_one_id(self) -> None:
        tracker = PersonTracker(fps=FPS)
        ids = set()
        for i in range(20):
            for p in tracker.update([pose_at(50 + i * 4)], frame(i)):
                ids.add(p.track_id)
        assert len(ids) == 1

    def test_two_people_crossing_apart_keep_distinct_ids(self) -> None:
        tracker = PersonTracker(fps=FPS)
        first = {}
        for i in range(30):
            out = tracker.update([pose_at(50 + i * 5), pose_at(400 - i * 5)], frame(i))
            for p in out:
                first.setdefault(round(p.pose.bbox[0]) // 1000, set()).add(p.track_id)
        assert len(tracker.active_ids) == 2

    def test_each_pose_gets_its_own_track_via_src_index(self) -> None:
        """Pemetaan balik memakai src_index, bukan urutan keluaran."""
        tracker = PersonTracker(fps=FPS)
        left_ids, right_ids = set(), set()
        for i in range(15):
            out = tracker.update([pose_at(50 + i * 3), pose_at(450 - i * 3)], frame(i))
            for p in out:
                (left_ids if p.pose.bbox[0] < 250 else right_ids).add(p.track_id)
        assert len(left_ids) == 1 and len(right_ids) == 1
        assert left_ids != right_ids

    def test_id_survives_short_dropout(self) -> None:
        """Orang membelakangi kamera sesaat tidak boleh mengganti identitasnya."""
        tracker = PersonTracker(fps=FPS, lost_track_buffer=30)
        ids: List[int] = []
        for i in range(12):
            poses = [] if i in (5, 6) else [pose_at(50 + i * 4)]
            out = tracker.update(poses, frame(i))
            ids.extend(p.track_id for p in out)
        assert len(set(ids)) == 1

    def test_empty_frame_returns_empty_list(self) -> None:
        assert PersonTracker(fps=FPS).update([], frame(0)) == []

    def test_label(self) -> None:
        tracker = PersonTracker(fps=FPS)
        out = tracker.update([pose_at(50)], frame(0))
        assert out[0].label == f"ID:{out[0].track_id}"


class TestHistory:
    def test_window_is_bounded_by_time_not_frame_count(self) -> None:
        tracker = PersonTracker(fps=FPS, window_s=1.0)
        for i in range(60):
            tracker.update([pose_at(50 + i * 2)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        assert state is not None
        # 1 detik pada 20 fps = 21 sampel inklusif (t=0 sampai t=1.0).
        assert len(state) == 21
        assert state.span_s == pytest.approx(1.0)

    def test_same_window_holds_fewer_samples_at_lower_fps(self) -> None:
        tracker = PersonTracker(fps=10.0, window_s=1.0)
        for i in range(40):
            tracker.update([pose_at(50 + i * 2)], frame(i, fps=10.0))
        state = tracker.state(tracker.active_ids[0])
        assert len(state) == 11

    def test_age_counts_from_first_sighting(self) -> None:
        tracker = PersonTracker(fps=FPS, window_s=0.5)
        for i in range(40):
            tracker.update([pose_at(50 + i * 2)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        assert state.age_s == pytest.approx(39 / FPS)
        assert state.span_s < state.age_s

    def test_series_accessors_match_length(self) -> None:
        tracker = PersonTracker(fps=FPS)
        for i in range(10):
            tracker.update([pose_at(50)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        n = len(state)
        assert len(state.hip_centre_series()) == n
        assert len(state.shoulder_centre_series()) == n
        assert len(state.bbox_series()) == n
        assert len(state.aspect_ratio_series()) == n
        assert len(state.keypoint_series(Keypoint.NOSE)) == n

    def test_invalid_keypoint_stays_none_in_series(self) -> None:
        """None dipertahankan, bukan dilewati, agar pemanggil tahu ada yang hilang."""
        tracker = PersonTracker(fps=FPS)
        for i in range(5):
            kwargs = {"left_hip": 0.1} if i == 2 else {}
            tracker.update([pose_at(50, **kwargs)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        hips = [h for _, h in state.hip_centre_series()]
        assert hips[2] is None
        assert all(h is not None for k, h in enumerate(hips) if k != 2)

    def test_torso_valid_ratio(self) -> None:
        tracker = PersonTracker(fps=FPS)
        for i in range(10):
            kwargs = {"left_hip": 0.1} if i < 4 else {}
            tracker.update([pose_at(50, **kwargs)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        assert state.torso_valid_ratio() == pytest.approx(0.6)


class TestContiguity:
    def test_continuous_history_is_contiguous(self) -> None:
        tracker = PersonTracker(fps=FPS)
        for i in range(10):
            tracker.update([pose_at(50)], frame(i))
        state = tracker.state(tracker.active_ids[0])
        assert state.is_contiguous(max_gap_s=0.2)

    def test_dropout_creates_a_gap_that_is_detected(self) -> None:
        """Kecepatan yang melintasi lubang tampak seperti gerakan mendadak."""
        tracker = PersonTracker(fps=FPS)
        for i in range(12):
            poses = [] if i in (5, 6, 7) else [pose_at(50 + i)]
            tracker.update(poses, frame(i))
        state = tracker.state(tracker.active_ids[0])
        assert not state.is_contiguous(max_gap_s=0.1)
        assert state.is_contiguous(max_gap_s=0.5)

    def test_short_history_is_trivially_contiguous(self) -> None:
        state = TrackState(track_id=1)
        assert state.is_contiguous(max_gap_s=0.1)
        state.append(TrackSample(0, 0.0, pose_at(50)))
        assert state.is_contiguous(max_gap_s=0.1)


class TestLifecycle:
    def test_stale_tracks_are_forgotten(self) -> None:
        tracker = PersonTracker(fps=FPS, prune_after_s=1.0)
        for i in range(5):
            tracker.update([pose_at(50)], frame(i))
        assert len(tracker) == 1
        for i in range(5, 60):
            tracker.update([], frame(i))
        assert len(tracker) == 0

    def test_reset_clears_everything(self) -> None:
        tracker = PersonTracker(fps=FPS)
        for i in range(5):
            tracker.update([pose_at(50)], frame(i))
        tracker.reset()
        assert len(tracker) == 0
        assert tracker.active_ids == []

    def test_state_for_unknown_id_is_none(self) -> None:
        assert PersonTracker(fps=FPS).state(999) is None


class TestValidation:
    def test_non_positive_fps_rejected(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            PersonTracker(fps=0)

    def test_non_positive_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_s"):
            PersonTracker(fps=FPS, window_s=0)


class TestTrackingConfig:
    def test_defaults_match_the_shipped_yaml(self, repo_root) -> None:
        cfg = AppConfig.from_yaml(repo_root / "config" / "default.yaml")
        assert cfg.tracking.window_s == 3.0
        assert cfg.tracking.lost_track_buffer == 30
        assert cfg.tracking.max_gap_s == 0.2

    def test_unknown_key_rejected(self, tmp_path) -> None:
        from pydantic import ValidationError

        path = tmp_path / "c.yaml"
        path.write_text("tracking:\n  windows_s: 3\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            AppConfig.from_yaml(path)

    def test_matching_threshold_default_is_the_measured_choice(self, repo_root) -> None:
        """0.95 dipilih dari pengukuran 100 klip; kode dan YAML tidak boleh berbeda."""
        assert AppConfig().tracking.minimum_matching_threshold == 0.95
        shipped = AppConfig.from_yaml(repo_root / "config" / "default.yaml")
        assert shipped.tracking.minimum_matching_threshold == 0.95

    def test_tracker_builds_from_config(self) -> None:
        cfg = AppConfig().tracking
        tracker = PersonTracker(
            fps=FPS,
            window_s=cfg.window_s,
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            prune_after_s=cfg.prune_after_s,
        )
        assert len(tracker) == 0


class TestTwoPersonCrossing:
    """Regresi untuk pertukaran ID antar orang pada ambang pencocokan 0.95.

    Ambang itu dipilih dari data satu-orang-per-frame. Test ini mengunci bahwa
    kasus paling dasar -- dua orang berpapasan lurus -- tetap mempertahankan
    identitas. Skenario yang lebih sulit (saling menutupi, berhenti) diukur di
    scripts/measure_crossing.py dan menunjukkan ~3% pertukaran yang tidak
    bergantung pada ambang, jadi sengaja TIDAK dikunci di sini.
    """

    @pytest.mark.parametrize("y_gap", [0.0, 25.0, 50.0])
    @pytest.mark.parametrize("threshold", [0.8, 0.95])
    def test_straight_pass_keeps_both_identities(self, y_gap: float, threshold: float) -> None:
        tracker = PersonTracker(fps=FPS, minimum_matching_threshold=threshold)
        first: dict = {}
        last: dict = {}
        n = 60
        for i in range(n):
            ax = 100.0 + 6.0 * i
            bx = 400.0 - 6.0 * i
            out = tracker.update([pose_at(ax), pose_at(bx)], frame(i))
            for t in out:
                who = "A" if abs(t.pose.bbox[0] - ax) < abs(t.pose.bbox[0] - bx) else "B"
                if i == 2:
                    first[who] = t.track_id
                if i == n - 1:
                    last[who] = t.track_id

        assert 100.0 + 6.0 * (n - 1) > 400.0 - 6.0 * (n - 1), "adegan harus berpapasan"
        assert first["A"] != first["B"]
        assert first == last
