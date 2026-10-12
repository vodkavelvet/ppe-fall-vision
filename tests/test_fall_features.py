"""Test fitur deteksi jatuh.

Sebagian test di sini mengunci **mode KEGAGALAN** yang terukur, bukan hanya
perilaku yang benar. Itu disengaja: temuan utama proyek ini adalah bahwa sudut
trunk gagal untuk jatuh searah sumbu kamera, dan kalau suatu saat seseorang
"memperbaiki" fitur itu tanpa menyadari batasnya, test ini yang akan memberi tahu.

Semua memakai pose sintetis, tanpa model dan tanpa GPU.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from pfv.fall import (
    FallFeatures,
    StaticFeatures,
    TemporalFeatures,
    bbox_diagonal,
    compute_static,
    compute_temporal,
    head_hip_dy_norm,
    hip_ankle_dy_norm,
    hip_knee_dy_norm,
    keypoint_spread_y_norm,
    trunk_angle_deg,
    trunk_length_norm,
    trunk_vector,
)
from pfv.fall.features import MIN_BBOX_PX, mean_valid_y
from pfv.perception.base import N_KEYPOINTS, Keypoint, PersonPose
from pfv.tracking import TrackSample, TrackState
from pfv_test_helpers import (
    bending_keypoints,
    forward_fall_bbox,
    forward_fall_keypoints,
    lying_bbox,
    lying_keypoints,
    make_keypoints,
    standing_bbox,
    standing_keypoints,
)


def standing() -> PersonPose:
    return PersonPose(standing_bbox(), standing_keypoints(), 0.9)


def bending() -> PersonPose:
    return PersonPose(standing_bbox(), bending_keypoints(), 0.9)


def lying() -> PersonPose:
    return PersonPose(lying_bbox(), lying_keypoints(), 0.9)


def forward_fall() -> PersonPose:
    return PersonPose(forward_fall_bbox(), forward_fall_keypoints(), 0.9)


# ------------------------------------------------------------------- bantuan


class TestBboxDiagonal:
    def test_pythagoras(self) -> None:
        assert bbox_diagonal((0.0, 0.0, 30.0, 40.0)) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "bbox",
        [
            (0.0, 0.0, 1.0, 100.0),  # terlalu sempit
            (0.0, 0.0, 100.0, 1.0),  # terlalu pendek
            (10.0, 10.0, 10.0, 10.0),  # degenerat
            (10.0, 10.0, 5.0, 5.0),  # terbalik
        ],
    )
    def test_degenerate_box_returns_none(self, bbox) -> None:
        assert bbox_diagonal(bbox) is None

    def test_threshold_is_the_documented_constant(self) -> None:
        assert bbox_diagonal((0.0, 0.0, MIN_BBOX_PX, MIN_BBOX_PX)) is not None
        assert bbox_diagonal((0.0, 0.0, MIN_BBOX_PX - 0.1, MIN_BBOX_PX)) is None


class TestMeanValidY:
    def test_averages_valid_only(self) -> None:
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        kp[Keypoint.LEFT_ANKLE] = (10.0, 100.0, 0.9)
        kp[Keypoint.RIGHT_ANKLE] = (20.0, 200.0, 0.1)  # tidak valid
        pose = PersonPose((0, 0, 50, 300), kp, 0.9)
        assert mean_valid_y(pose, Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE) == pytest.approx(100.0)

    def test_one_side_is_enough_for_limbs(self) -> None:
        """Berbeda dari pusat trunk: untuk anggota badan hanya ketinggian yang dipakai."""
        pose = PersonPose(standing_bbox(), standing_keypoints(), 0.9)
        kp = pose.keypoints.copy()
        kp[Keypoint.RIGHT_ANKLE, 2] = 0.1
        partial = PersonPose(standing_bbox(), kp, 0.9)
        assert hip_ankle_dy_norm(partial) is not None

    def test_none_when_nothing_valid(self) -> None:
        pose = PersonPose.empty()
        assert mean_valid_y(pose, Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE) is None


class TestTrunkVector:
    def test_points_from_shoulders_to_hips(self) -> None:
        vec = trunk_vector(standing())
        assert vec is not None
        # Sumbu y gambar tumbuh ke bawah, jadi pinggul di bawah bahu = y positif.
        assert vec[1] > 0

    def test_requires_both_shoulders(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.LEFT_SHOULDER, 2] = 0.1
        assert trunk_vector(PersonPose(standing_bbox(), kp, 0.9)) is None

    def test_requires_both_hips(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.RIGHT_HIP, 2] = 0.1
        assert trunk_vector(PersonPose(standing_bbox(), kp, 0.9)) is None


# ---------------------------------------------------------------- sudut trunk


class TestTrunkAngle:
    def test_standing_is_near_vertical(self) -> None:
        assert trunk_angle_deg(standing()) == pytest.approx(0.0, abs=5.0)

    def test_lateral_lying_is_near_horizontal(self) -> None:
        assert trunk_angle_deg(lying()) == pytest.approx(90.0, abs=5.0)

    def test_unsigned_left_and_right_tilt_are_equal(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.LEFT_HIP, 0] += 40
        kp[Keypoint.RIGHT_HIP, 0] += 40
        right = trunk_angle_deg(PersonPose(standing_bbox(), kp, 0.9))

        kp = standing_keypoints()
        kp[Keypoint.LEFT_HIP, 0] -= 40
        kp[Keypoint.RIGHT_HIP, 0] -= 40
        left = trunk_angle_deg(PersonPose(standing_bbox(), kp, 0.9))

        assert right == pytest.approx(left)

    def test_coincident_centres_return_none_not_zero(self) -> None:
        """Mengembalikan 0 akan terbaca sebagai 'tegak sempurna', padahal tak terdefinisi."""
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        for i in (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER,
                  Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP):
            kp[i] = (100.0, 150.0, 0.9)
        assert trunk_angle_deg(PersonPose((0, 0, 50, 300), kp, 0.9)) is None

    def test_missing_keypoints_return_none(self) -> None:
        assert trunk_angle_deg(PersonPose.empty()) is None

    def test_FAILS_to_separate_camera_axis_fall_from_standing(self) -> None:
        """Mode kegagalan terukur: 3 dari 10 subjek, dikunci di sini.

        Orang yang jatuh ke arah kamera tetap tampak tegak di bidang gambar, jadi
        sudut trunk hampir tidak bergerak. Ini alasan fitur ini tidak boleh
        dipakai sendirian.
        """
        upright = trunk_angle_deg(standing())
        fallen = trunk_angle_deg(forward_fall())
        assert upright is not None and fallen is not None
        assert abs(fallen - upright) < 15.0, (
            "Kalau test ini gagal, sudut trunk ternyata MEMISAHKAN jatuh searah "
            "kamera. Verifikasi ulang dengan scripts/analyse_per_subject.py sebelum "
            "mengubah asumsi Task 6."
        )

    def test_does_separate_lateral_fall_from_standing(self) -> None:
        """Pembanding: untuk jatuh menyamping fitur ini justru sangat kuat (9-10/10)."""
        upright = trunk_angle_deg(standing())
        fallen = trunk_angle_deg(lying())
        assert fallen - upright > 60.0


# -------------------------------------------------------------- panjang trunk


class TestTrunkLength:
    def test_camera_axis_fall_shortens_the_trunk(self) -> None:
        """Fitur ini yang menangkap jatuh searah kamera: 7 dari 10 subjek."""
        assert trunk_length_norm(forward_fall()) < trunk_length_norm(standing())

    def test_is_normalised_by_diagonal_so_scale_invariant(self) -> None:
        small = PersonPose(standing_bbox(), standing_keypoints(), 0.9)
        big = PersonPose(
            standing_bbox(x=200.0, top=100.0, height=320.0),
            standing_keypoints(x=200.0, top=100.0, height=320.0),
            0.9,
        )
        assert trunk_length_norm(small) == pytest.approx(trunk_length_norm(big), abs=0.02)

    def test_bending_also_shortens_the_trunk(self) -> None:
        """Batasan jujur: membungkuk memendekkan trunk lebih dari jatuh ke depan.

        Karena itu panjang trunh tidak bisa dipakai sendirian untuk membedakan
        jatuh ke depan dari mengambil objek. Dicatat sebagai pengganggu nyata.
        """
        assert trunk_length_norm(bending()) < trunk_length_norm(forward_fall())

    def test_none_without_torso(self) -> None:
        assert trunk_length_norm(PersonPose.empty()) is None


# ------------------------------------------------- selisih tinggi anggota badan


class TestHipAnkleDy:
    def test_lateral_lying_is_near_zero(self) -> None:
        """Saat tergeletak, pinggul dan kaki berada pada ketinggian sama."""
        assert hip_ankle_dy_norm(lying()) == pytest.approx(0.0, abs=0.05)

    def test_separates_camera_axis_fall_from_standing_and_bending(self) -> None:
        """Fitur dengan cakupan terbaik: 7/10 untuk jatuh ke depan, 9-10/10 menyamping."""
        fallen = hip_ankle_dy_norm(forward_fall())
        assert fallen < hip_ankle_dy_norm(standing())
        assert fallen < hip_ankle_dy_norm(bending())

    def test_bending_keeps_hips_high(self) -> None:
        """Inilah yang membedakan membungkuk dari tergeletak."""
        assert hip_ankle_dy_norm(bending()) > 0.3

    def test_none_without_ankles(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.LEFT_ANKLE, 2] = 0.1
        kp[Keypoint.RIGHT_ANKLE, 2] = 0.1
        assert hip_ankle_dy_norm(PersonPose(standing_bbox(), kp, 0.9)) is None


class TestHipKneeDy:
    def test_lateral_lying_is_near_zero(self) -> None:
        assert hip_knee_dy_norm(lying()) == pytest.approx(0.0, abs=0.05)

    def test_separates_camera_axis_fall_from_standing(self) -> None:
        assert hip_knee_dy_norm(forward_fall()) < hip_knee_dy_norm(standing())

    def test_none_without_knees(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.LEFT_KNEE, 2] = 0.1
        kp[Keypoint.RIGHT_KNEE, 2] = 0.1
        assert hip_knee_dy_norm(PersonPose(standing_bbox(), kp, 0.9)) is None


class TestHeadHipDy:
    def test_lateral_lying_is_near_zero(self) -> None:
        assert head_hip_dy_norm(lying()) == pytest.approx(0.0, abs=0.05)

    def test_FAILS_to_separate_bending_from_camera_axis_fall(self) -> None:
        """Mode kegagalan terukur: 4 dari 10 subjek untuk jatuh ke depan.

        Orang yang membungkuk juga menurunkan kepala ke dekat pinggul, bahkan
        lebih rendah daripada orang yang jatuh ke depan.
        """
        assert head_hip_dy_norm(bending()) < head_hip_dy_norm(forward_fall())

    def test_none_without_nose(self) -> None:
        kp = standing_keypoints()
        kp[Keypoint.NOSE, 2] = 0.1
        assert head_hip_dy_norm(PersonPose(standing_bbox(), kp, 0.9)) is None


class TestKeypointSpreadY:
    def test_strong_for_lateral_lying(self) -> None:
        """Fitur terkuat untuk jatuh menyamping: 10 dari 10 subjek."""
        assert keypoint_spread_y_norm(lying()) < 0.3
        assert keypoint_spread_y_norm(standing()) > 0.7

    def test_weaker_for_camera_axis_fall(self) -> None:
        """Hanya 6 dari 10 subjek: tubuh yang memendek tetap menyisakan rentang vertikal."""
        fallen = keypoint_spread_y_norm(forward_fall())
        assert fallen < keypoint_spread_y_norm(standing())
        # Tetapi jaraknya ke membungkuk sangat sempit, jadi tidak bisa diandalkan.
        assert abs(fallen - keypoint_spread_y_norm(bending())) < 0.1

    def test_requires_minimum_keypoints(self) -> None:
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        kp[Keypoint.NOSE] = (10.0, 10.0, 0.9)
        kp[Keypoint.LEFT_HIP] = (10.0, 60.0, 0.9)
        assert keypoint_spread_y_norm(PersonPose((0, 0, 50, 100), kp, 0.9)) is None

    def test_computed_from_keypoints_not_bbox(self) -> None:
        """Tidak mewarisi kesalahan detektor saat kotak melebar karena lengan."""
        pose = standing()
        wide = PersonPose((0.0, 50.0, 400.0, 210.0), standing_keypoints(), 0.9)
        assert keypoint_spread_y_norm(wide) != pytest.approx(
            keypoint_spread_y_norm(pose), abs=0.05
        )


# ----------------------------------------------------- normalisasi: diagonal


class TestNormalisationChoice:
    def test_diagonal_not_height_is_used(self) -> None:
        """Tinggi kotak ikut runtuh saat tergeletak dan akan menyembunyikan efeknya.

        Dibuktikan pada pose tergeletak, di mana tinggi kotak jauh lebih kecil
        dari lebarnya. Kalau pembaginya tinggi, nilainya meledak di atas 1 dan
        kehilangan makna; dengan diagonal, nilainya tetap dalam rentang wajar.
        """
        pose = lying()
        x1, y1, x2, y2 = pose.bbox
        height = y2 - y1
        vec = trunk_vector(pose)
        assert vec is not None

        by_height = float(np.hypot(*vec)) / height
        by_diagonal = trunk_length_norm(pose)

        assert by_height > 1.5, "kotak tergeletak harus jauh lebih lebar dari tinggi"
        assert by_diagonal < 0.5
        assert by_diagonal == pytest.approx(
            float(np.hypot(*vec)) / float(np.hypot(x2 - x1, height))
        )

    def test_features_are_scale_invariant(self) -> None:
        a = PersonPose(standing_bbox(), standing_keypoints(), 0.9)
        b = PersonPose(
            standing_bbox(x=400.0, top=20.0, height=480.0),
            standing_keypoints(x=400.0, top=20.0, height=480.0),
            0.9,
        )
        fa, fb = compute_static(a).as_dict(), compute_static(b).as_dict()
        for name in fa:
            if fa[name] is None or fb[name] is None:
                continue
            assert fa[name] == pytest.approx(fb[name], abs=0.03), name


# --------------------------------------------------------------- compute_static


class TestComputeStatic:
    def test_all_fields_present_for_complete_pose(self) -> None:
        feats = compute_static(standing())
        assert feats.n_available == 7
        assert all(v is not None for v in feats.as_dict().values())

    def test_never_raises_on_empty_pose(self) -> None:
        feats = compute_static(PersonPose.empty())
        assert feats.n_available == 0
        assert all(v is None for v in feats.as_dict().values())

    def test_partial_pose_yields_partial_features(self) -> None:
        kp = standing_keypoints()
        for i in (Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE, Keypoint.NOSE):
            kp[i, 2] = 0.1
        feats = compute_static(PersonPose(standing_bbox(), kp, 0.9))
        assert feats.hip_ankle_dy_norm is None
        assert feats.head_hip_dy_norm is None
        assert feats.trunk_angle_deg is not None

    def test_is_frozen(self) -> None:
        feats = compute_static(standing())
        with pytest.raises(Exception):
            feats.trunk_angle_deg = 1.0  # type: ignore[misc]


# -------------------------------------------------------------- fitur temporal


def history(
    poses: List[PersonPose], *, fps: float = 20.0, gap_after: Optional[int] = None
) -> TrackState:
    """Bangun TrackState dari daftar pose. ``gap_after`` menyisipkan lubang waktu."""
    state = TrackState(track_id=1, window_s=10.0)
    t = 0.0
    for i, pose in enumerate(poses):
        state.append(TrackSample(frame_index=i, timestamp_s=t, pose=pose))
        t += 1.0 / fps
        if gap_after is not None and i == gap_after:
            t += 1.0  # lubang 1 detik
    return state


def descending(n: int = 20, drop_per_frame: float = 4.0) -> List[PersonPose]:
    """Orang yang pinggulnya turun tetap, untuk menguji kecepatan."""
    poses = []
    for i in range(n):
        kp = standing_keypoints(top=50.0 + i * drop_per_frame)
        bbox = standing_bbox(top=50.0 + i * drop_per_frame)
        poses.append(PersonPose(bbox, kp, 0.9))
    return poses


class TestTemporalFeatures:
    def test_empty_history(self) -> None:
        feats = compute_temporal(TrackState(track_id=1))
        assert feats.n_samples == 0
        assert feats.hip_velocity_norm is None

    def test_single_sample_has_no_velocity(self) -> None:
        feats = compute_temporal(history([standing()]))
        assert feats.n_samples == 1
        assert feats.hip_velocity_norm is None

    def test_static_person_has_near_zero_velocity(self) -> None:
        feats = compute_temporal(history([standing() for _ in range(20)]))
        assert feats.hip_velocity_norm == pytest.approx(0.0, abs=1e-6)

    def test_descending_hip_gives_positive_velocity(self) -> None:
        """Positif berarti TURUN, karena sumbu y gambar tumbuh ke bawah."""
        feats = compute_temporal(history(descending()))
        assert feats.hip_velocity_norm is not None
        assert feats.hip_velocity_norm > 0

    def test_rising_hip_gives_negative_velocity(self) -> None:
        feats = compute_temporal(history(list(reversed(descending()))))
        assert feats.hip_velocity_norm < 0

    def test_velocity_is_scale_invariant(self) -> None:
        """Orang yang sama jatuh dengan kecepatan sama harus memberi nilai sama,
        tidak peduli jaraknya ke kamera."""
        near = compute_temporal(history(descending(drop_per_frame=8.0)))
        poses = [
            PersonPose(
                standing_bbox(x=100.0, top=50.0 + i * 4.0, height=80.0),
                standing_keypoints(x=100.0, top=50.0 + i * 4.0, height=80.0),
                0.9,
            )
            for i in range(20)
        ]
        far = compute_temporal(history(poses))
        assert near.hip_velocity_norm == pytest.approx(far.hip_velocity_norm, rel=0.2)

    def test_gap_in_history_suppresses_velocity(self) -> None:
        """Kecepatan yang melintasi lubang deteksi tampak seperti gerakan mendadak."""
        feats = compute_temporal(history(descending(), gap_after=5), max_gap_s=0.2)
        assert feats.contiguous is False
        assert feats.hip_velocity_norm is None
        assert feats.trunk_angle_rate is None
        assert feats.motion_energy is None

    def test_generous_max_gap_allows_velocity(self) -> None:
        feats = compute_temporal(history(descending(), gap_after=5), max_gap_s=2.0)
        assert feats.contiguous is True
        assert feats.hip_velocity_norm is not None

    def test_slope_fit_resists_single_frame_noise(self) -> None:
        """Regresi dipakai alih-alih (akhir - awal) justru untuk kasus ini."""
        clean = descending()
        noisy = list(clean)
        kp = standing_keypoints(top=50.0 + 10 * 4.0)
        kp[Keypoint.LEFT_HIP, 1] += 120
        kp[Keypoint.RIGHT_HIP, 1] += 120
        noisy[10] = PersonPose(standing_bbox(top=50.0 + 10 * 4.0), kp, 0.9)

        a = compute_temporal(history(clean)).hip_velocity_norm
        b = compute_temporal(history(noisy)).hip_velocity_norm
        assert b == pytest.approx(a, rel=0.25)

    def test_trunk_angle_rate_positive_when_tipping_over(self) -> None:
        poses = []
        for i in range(20):
            kp = standing_keypoints()
            kp[Keypoint.LEFT_HIP, 0] += i * 4
            kp[Keypoint.RIGHT_HIP, 0] += i * 4
            poses.append(PersonPose(standing_bbox(), kp, 0.9))
        feats = compute_temporal(history(poses))
        assert feats.trunk_angle_rate is not None
        assert feats.trunk_angle_rate > 0

    def test_motion_energy_zero_when_still(self) -> None:
        feats = compute_temporal(history([standing() for _ in range(10)]))
        assert feats.motion_energy == pytest.approx(0.0, abs=1e-6)

    def test_motion_energy_positive_when_moving(self) -> None:
        feats = compute_temporal(history(descending()))
        assert feats.motion_energy > 0

    def test_motion_energy_ignores_keypoints_changing_validity(self) -> None:
        """Keypoint yang muncul atau hilang bukan gerakan."""
        a = standing()
        kp = standing_keypoints()
        kp[Keypoint.LEFT_ANKLE, 2] = 0.1
        b = PersonPose(standing_bbox(), kp, 0.9)
        feats = compute_temporal(history([a, b, a, b]))
        assert feats.motion_energy == pytest.approx(0.0, abs=1e-6)

    def test_window_and_sample_count_reported(self) -> None:
        feats = compute_temporal(history(descending(n=11)))
        assert feats.n_samples == 11
        assert feats.window_s == pytest.approx(10 / 20.0)


class TestFallFeatures:
    def test_merges_static_and_temporal(self) -> None:
        combined = FallFeatures(
            static=compute_static(standing()),
            temporal=compute_temporal(history(descending())),
        )
        merged = combined.as_dict()
        assert "trunk_angle_deg" in merged
        assert "hip_velocity_norm" in merged
        assert len(merged) == 10


class TestDocumentedComplementarity:
    """Inti temuan: fitur yang bekerja untuk satu arah jatuh gagal untuk arah lain.

    Dikunci sebagai test supaya asumsi Task 6 tidak bergeser tanpa disadari.
    """

    def test_lateral_fall_is_caught_by_angle_not_by_trunk_length(self) -> None:
        up, down = standing(), lying()
        assert trunk_angle_deg(down) - trunk_angle_deg(up) > 60.0
        # Panjang trunk justru NAIK saat tergeletak menyamping.
        assert trunk_length_norm(down) > trunk_length_norm(up)

    def test_camera_axis_fall_is_caught_by_length_and_limbs_not_by_angle(self) -> None:
        up, down = standing(), forward_fall()
        assert abs(trunk_angle_deg(down) - trunk_angle_deg(up)) < 15.0
        assert trunk_length_norm(down) < trunk_length_norm(up)
        assert hip_ankle_dy_norm(down) < hip_ankle_dy_norm(up)

    def test_no_single_feature_separates_both_directions(self) -> None:
        """Dinyatakan eksplisit per fitur, dalam satuan masing-masing.

        Membandingkan perubahan relatif untuk sudut tidak sah, karena sudut
        berdiri bisa nol sehingga pembaginya nol.
        """
        up, lateral, axial = standing(), lying(), forward_fall()

        # Sudut trunk: kuat menyamping (puluhan derajat), nyaris nol searah kamera.
        assert trunk_angle_deg(lateral) - trunk_angle_deg(up) > 60.0
        assert abs(trunk_angle_deg(axial) - trunk_angle_deg(up)) < 15.0

        # Panjang trunk: menyusut searah kamera, tetapi justru NAIK menyamping,
        # jadi arah perubahannya berlawanan dan tidak bisa dipakai satu ambang.
        assert trunk_length_norm(axial) < trunk_length_norm(up)
        assert trunk_length_norm(lateral) > trunk_length_norm(up)
