"""Test kontrak pose: PersonPose, indeks keypoint COCO, pemilihan device.

Semuanya tanpa model dan tanpa GPU, jadi cepat dan bisa jalan di mana pun.
Keputusan yang paling dijaga di sini: keypoint ber-confidence rendah DITANDAI
invalid, bukan dibuang dan bukan diganti nol.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfv.io.frames import FrameTransform
from pfv.perception.base import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    HEAD_KEYPOINTS,
    KNEE_KEYPOINTS,
    N_KEYPOINTS,
    TORSO_KEYPOINTS,
    Keypoint,
    PersonPose,
    PoseError,
    PoseEstimator,
    keypoints_from_xy_conf,
    resolve_device,
)
from pfv_test_helpers import lying_keypoints, make_keypoints, standing_keypoints


class TestKeypointConstants:
    def test_seventeen_keypoints(self) -> None:
        assert N_KEYPOINTS == 17
        assert len(Keypoint) == 17
        assert len(COCO_KEYPOINT_NAMES) == 17

    def test_coco_order_is_exact(self) -> None:
        """Salah indeks di sini menghasilkan fitur salah tanpa error apa pun."""
        assert Keypoint.NOSE == 0
        assert Keypoint.LEFT_SHOULDER == 5
        assert Keypoint.RIGHT_SHOULDER == 6
        assert Keypoint.LEFT_HIP == 11
        assert Keypoint.RIGHT_HIP == 12
        assert Keypoint.LEFT_KNEE == 13
        assert Keypoint.RIGHT_KNEE == 14
        assert Keypoint.RIGHT_ANKLE == 16

    def test_groups_match_their_purpose(self) -> None:
        assert set(HEAD_KEYPOINTS) == {0, 1, 2, 3, 4}
        assert set(TORSO_KEYPOINTS) == {5, 6, 11, 12}
        # Lutut memisahkan BERLUTUT dari JATUH; CAUCAFall memuat berlutut sebagai ADL.
        assert set(KNEE_KEYPOINTS) == {13, 14}

    def test_skeleton_edges_are_in_range(self) -> None:
        for a, b in COCO_SKELETON:
            assert 0 <= a < N_KEYPOINTS
            assert 0 <= b < N_KEYPOINTS
            assert a != b

    def test_skeleton_connects_the_torso(self) -> None:
        edges = {frozenset(e) for e in COCO_SKELETON}
        assert frozenset({Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER}) in edges
        assert frozenset({Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP}) in edges
        assert frozenset({Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP}) in edges


class TestPersonPoseShape:
    def test_shape_is_always_17x3(self) -> None:
        pose = PersonPose((0, 0, 10, 40), make_keypoints(), 0.9)
        assert pose.keypoints.shape == (N_KEYPOINTS, 3)
        assert pose.valid.shape == (N_KEYPOINTS,)

    def test_wrong_shape_is_rejected(self) -> None:
        with pytest.raises(PoseError, match=r"\(17, 3\)"):
            PersonPose((0, 0, 1, 1), np.zeros((16, 3), dtype=np.float32), 0.9)
        with pytest.raises(PoseError, match=r"\(17, 3\)"):
            PersonPose((0, 0, 1, 1), np.zeros((17, 2), dtype=np.float32), 0.9)

    def test_empty_pose_keeps_the_shape(self) -> None:
        """Bentuk tidak pernah berubah, bahkan ketika semuanya tidak terdeteksi."""
        pose = PersonPose.empty()
        assert pose.keypoints.shape == (N_KEYPOINTS, 3)
        assert pose.n_valid() == 0

    def test_list_input_is_converted(self) -> None:
        pose = PersonPose((0, 0, 1, 1), make_keypoints().tolist(), 0.5)
        assert isinstance(pose.keypoints, np.ndarray)
        assert pose.keypoints.dtype == np.float32


class TestValidityMarking:
    def test_low_confidence_marked_invalid_not_dropped(self) -> None:
        pose = PersonPose((0, 0, 10, 40), make_keypoints(left_hip=0.1), 0.9, keypoint_conf=0.5)

        assert not pose.valid[Keypoint.LEFT_HIP]
        assert pose.keypoint(Keypoint.LEFT_HIP) is None
        # Nilai mentahnya TETAP ADA: tidak dibuang, hanya ditandai.
        assert pose.keypoints[Keypoint.LEFT_HIP, 0] != 0
        assert pose.keypoint_conf_value(Keypoint.LEFT_HIP) == pytest.approx(0.1)

    def test_threshold_is_inclusive(self) -> None:
        pose = PersonPose((0, 0, 1, 1), make_keypoints(nose=0.5), 0.9, keypoint_conf=0.5)
        assert pose.valid[Keypoint.NOSE]

    def test_threshold_can_be_changed_without_reinference(self) -> None:
        """Ambang diterapkan di luar inference; itu yang membuatnya murah disetel."""
        kp = make_keypoints(nose=0.4)
        strict = PersonPose((0, 0, 1, 1), kp, 0.9, keypoint_conf=0.5)
        loose = PersonPose((0, 0, 1, 1), kp, 0.9, keypoint_conf=0.3)
        assert not strict.valid[Keypoint.NOSE]
        assert loose.valid[Keypoint.NOSE]

    def test_has_requires_all_listed(self) -> None:
        pose = PersonPose((0, 0, 1, 1), make_keypoints(left_hip=0.1), 0.9)
        assert pose.has(Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER)
        assert not pose.has(Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP)

    def test_has_with_no_arguments_is_false(self) -> None:
        assert not PersonPose((0, 0, 1, 1), make_keypoints(), 0.9).has()

    def test_n_valid_counts_correctly(self) -> None:
        pose = PersonPose(
            (0, 0, 1, 1), make_keypoints(nose=0.1, left_hip=0.2), 0.9, keypoint_conf=0.5
        )
        assert pose.n_valid() == 15


class TestMidpointsAndTorso:
    def test_midpoint_averages_two_keypoints(self) -> None:
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        kp[Keypoint.LEFT_HIP] = (10.0, 100.0, 0.9)
        kp[Keypoint.RIGHT_HIP] = (30.0, 120.0, 0.9)
        pose = PersonPose((0, 0, 50, 200), kp, 0.9)
        assert pose.hip_centre() == pytest.approx((20.0, 110.0))

    def test_midpoint_requires_both_sides(self) -> None:
        """Mentoleransi satu sisi akan memiringkan vektor trunk dan merusak sudutnya."""
        pose = PersonPose((0, 0, 1, 1), make_keypoints(left_hip=0.1), 0.9)
        assert pose.hip_centre() is None
        assert pose.shoulder_centre() is not None

    def test_full_torso_requires_four_keypoints(self) -> None:
        assert PersonPose((0, 0, 1, 1), make_keypoints(), 0.9).has_full_torso()
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            pose = PersonPose((0, 0, 1, 1), make_keypoints(**{name: 0.1}), 0.9)
            assert not pose.has_full_torso(), name

    def test_empty_pose_has_no_torso(self) -> None:
        assert not PersonPose.empty().has_full_torso()


class TestBoundingBoxGeometry:
    def test_width_height_and_ratio(self) -> None:
        pose = PersonPose((10.0, 20.0, 40.0, 140.0), make_keypoints(), 0.9)
        assert pose.bbox_width == pytest.approx(30.0)
        assert pose.bbox_height == pytest.approx(120.0)
        assert pose.bbox_aspect_ratio == pytest.approx(4.0)

    def test_standing_above_one_lying_below_one(self) -> None:
        """Audit CAUCAFall: median 2,46 berjalan versus 0,59 tergeletak."""
        standing = PersonPose((0, 0, 40, 160), make_keypoints(), 0.9)
        lying = PersonPose((0, 0, 160, 50), make_keypoints(), 0.9)
        assert standing.bbox_aspect_ratio > 1.0
        assert lying.bbox_aspect_ratio < 1.0

    def test_zero_width_returns_none_not_infinity(self) -> None:
        pose = PersonPose((10.0, 20.0, 10.0, 140.0), make_keypoints(), 0.9)
        assert pose.bbox_aspect_ratio is None


class TestToSource:
    def test_maps_box_and_keypoints_back(self) -> None:
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        kp[Keypoint.NOSE] = (100.0, 50.0, 0.9)
        pose = PersonPose((10.0, 20.0, 110.0, 220.0), kp, 0.8)
        transform = FrameTransform(scale=0.5, pad_x=0, pad_y=0)

        mapped = pose.to_source(transform)
        assert mapped.bbox == pytest.approx((20.0, 40.0, 220.0, 440.0))
        assert mapped.keypoint(Keypoint.NOSE) == pytest.approx((200.0, 100.0))

    def test_accounts_for_letterbox_padding(self) -> None:
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        kp[Keypoint.NOSE] = (20.0, 30.0, 0.9)
        pose = PersonPose((0.0, 0.0, 10.0, 10.0), kp, 0.8)
        transform = FrameTransform(scale=0.5, pad_x=20, pad_y=10)

        mapped = pose.to_source(transform)
        assert mapped.keypoint(Keypoint.NOSE) == pytest.approx((0.0, 40.0))

    def test_confidence_and_validity_preserved(self) -> None:
        pose = PersonPose(
            (0, 0, 10, 40), make_keypoints(left_hip=0.1), 0.77, keypoint_conf=0.5
        )
        mapped = pose.to_source(FrameTransform(scale=0.5))
        assert mapped.confidence == pytest.approx(0.77)
        assert mapped.keypoint_conf == pytest.approx(0.5)
        assert not mapped.valid[Keypoint.LEFT_HIP]
        # Keypoint invalid juga dipetakan, agar bentuk (17, 3) tetap utuh.
        assert mapped.keypoints.shape == (N_KEYPOINTS, 3)

    def test_identity_transform_changes_nothing(self) -> None:
        pose = PersonPose((1.0, 2.0, 3.0, 4.0), make_keypoints(), 0.9)
        mapped = pose.to_source(FrameTransform())
        assert mapped.bbox == pytest.approx(pose.bbox)
        np.testing.assert_allclose(mapped.keypoints, pose.keypoints)


class TestKeypointsFromXyConf:
    def test_combines_separate_arrays(self) -> None:
        xy = [[float(i), float(i + 1)] for i in range(N_KEYPOINTS)]
        conf = [0.9] * N_KEYPOINTS
        kp = keypoints_from_xy_conf(xy, conf)
        assert kp.shape == (N_KEYPOINTS, 3)
        assert kp[3, 2] == pytest.approx(0.9)

    def test_missing_conf_treated_as_fully_valid(self) -> None:
        """Model tanpa confidence per keypoint tidak memberi dasar untuk menyaring."""
        xy = [[0.0, 0.0]] * N_KEYPOINTS
        kp = keypoints_from_xy_conf(xy, None)
        assert np.all(kp[:, 2] == 1.0)

    def test_wrong_shape_rejected(self) -> None:
        with pytest.raises(PoseError, match=r"\(17, 2\)"):
            keypoints_from_xy_conf([[0.0, 0.0]] * 16, None)


class TestResolveDevice:
    def test_auto_returns_a_known_device(self) -> None:
        assert resolve_device("auto") in {"mps", "cuda", "cpu"}

    def test_cpu_always_honoured(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_unavailable_device_degrades_to_cpu_not_error(self, monkeypatch, caplog) -> None:
        """Lebih baik lambat daripada tidak jalan, terutama di kernel Kaggle."""
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("mps") == "cpu"
        assert resolve_device("cuda") == "cpu"
        assert resolve_device("auto") == "cpu"

    def test_blank_treated_as_auto(self) -> None:
        assert resolve_device("") in {"mps", "cuda", "cpu"}


class TestProtocolConformance:
    def test_person_pose_estimator_protocol_is_runtime_checkable(self) -> None:
        class Dummy:
            @property
            def device(self) -> str:
                return "cpu"

            def estimate(self, image: np.ndarray) -> list:
                return []

        assert isinstance(Dummy(), PoseEstimator)

    def test_object_missing_estimate_is_not_an_estimator(self) -> None:
        class Incomplete:
            @property
            def device(self) -> str:
                return "cpu"

        assert not isinstance(Incomplete(), PoseEstimator)


class TestRealisticPostures:
    """Pose sintetis yang menyerupai postur nyata.

    Dipakai Task 5 untuk menguji ``trunk_angle`` tanpa menjalankan model, jadi
    bentuk datanya dikunci di sini sejak sekarang.
    """

    def test_standing_trunk_is_near_vertical(self) -> None:
        pose = PersonPose((88, 50, 112, 210), standing_keypoints(), 0.9)
        shoulder = pose.shoulder_centre()
        hip = pose.hip_centre()
        assert shoulder is not None and hip is not None

        dx = abs(hip[0] - shoulder[0])
        dy = abs(hip[1] - shoulder[1])
        # Vertikal berarti perpindahan tegak jauh lebih besar dari mendatar.
        assert dy > 3 * dx

    def test_lying_trunk_is_near_horizontal(self) -> None:
        pose = PersonPose((100, 280, 260, 320), lying_keypoints(), 0.9)
        shoulder = pose.shoulder_centre()
        hip = pose.hip_centre()
        assert shoulder is not None and hip is not None

        dx = abs(hip[0] - shoulder[0])
        dy = abs(hip[1] - shoulder[1])
        assert dx > 3 * dy

    def test_both_postures_have_full_torso_and_knees(self) -> None:
        for kp in (standing_keypoints(), lying_keypoints()):
            pose = PersonPose((0, 0, 100, 100), kp, 0.9)
            assert pose.has_full_torso()
            assert pose.has(Keypoint.LEFT_KNEE, Keypoint.RIGHT_KNEE)
