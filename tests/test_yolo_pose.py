"""Test pembungkus YOLO-pose.

Konversi hasil diuji dengan objek palsu, tanpa memuat model dan tanpa GPU.
Integration test atas frame CAUCAFall nyata ada di bawah dan di-skip bila bundel
atau bobotnya tidak ada di mesin ini.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from pfv.perception.base import N_KEYPOINTS, Keypoint, PoseError
from pfv.perception.yolo_pose import DEFAULT_MODEL, YoloPoseEstimator, _to_numpy


class FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: Optional[np.ndarray]) -> None:
        self.xy = xy
        self.conf = conf


class FakeResult:
    def __init__(self, boxes: Optional[FakeBoxes], keypoints: Optional[FakeKeypoints]) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


def fake_result(n_people: int = 1, with_conf: bool = True) -> FakeResult:
    xyxy = np.array([[10.0, 20.0, 50.0, 180.0]] * n_people, dtype=np.float32)
    det_conf = np.array([0.9] * n_people, dtype=np.float32)
    xy = np.zeros((n_people, N_KEYPOINTS, 2), dtype=np.float32)
    for p in range(n_people):
        for k in range(N_KEYPOINTS):
            xy[p, k] = (10.0 + k, 20.0 + k)
    kp_conf = np.full((n_people, N_KEYPOINTS), 0.9, dtype=np.float32) if with_conf else None
    return FakeResult(FakeBoxes(xyxy, det_conf), FakeKeypoints(xy, kp_conf))


class TestConstruction:
    def test_defaults(self) -> None:
        est = YoloPoseEstimator()
        assert est.model_name == DEFAULT_MODEL
        assert est.weights_path == Path("models") / DEFAULT_MODEL

    def test_bare_name_resolved_into_models_dir(self, tmp_path: Path) -> None:
        est = YoloPoseEstimator(model="yolo11s-pose.pt", models_dir=tmp_path)
        assert est.weights_path == tmp_path / "yolo11s-pose.pt"

    def test_explicit_path_is_respected(self) -> None:
        est = YoloPoseEstimator(model="weights/custom.pt")
        assert est.weights_path == Path("weights/custom.pt")

    @pytest.mark.parametrize("conf", [0.0, -0.1, 1.5])
    def test_invalid_conf_rejected(self, conf: float) -> None:
        with pytest.raises(PoseError, match="conf"):
            YoloPoseEstimator(conf=conf)

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_invalid_keypoint_conf_rejected(self, value: float) -> None:
        with pytest.raises(PoseError, match="keypoint_conf"):
            YoloPoseEstimator(keypoint_conf=value)

    def test_invalid_imgsz_rejected(self) -> None:
        with pytest.raises(PoseError, match="imgsz"):
            YoloPoseEstimator(imgsz=0)

    def test_model_not_loaded_on_construction(self) -> None:
        """Lazy load: membuat objek tidak boleh menarik torch atau membaca bobot."""
        est = YoloPoseEstimator()
        assert est._model is None


class TestConvert:
    def test_single_person(self) -> None:
        est = YoloPoseEstimator(keypoint_conf=0.5)
        poses = est._convert(fake_result(1))

        assert len(poses) == 1
        p = poses[0]
        assert p.bbox == pytest.approx((10.0, 20.0, 50.0, 180.0))
        assert p.confidence == pytest.approx(0.9)
        assert p.keypoints.shape == (N_KEYPOINTS, 3)
        assert p.n_valid() == N_KEYPOINTS

    def test_multiple_people(self) -> None:
        poses = YoloPoseEstimator()._convert(fake_result(3))
        assert len(poses) == 3

    def test_keypoint_threshold_applied_from_estimator(self) -> None:
        result = fake_result(1)
        result.keypoints.conf[0, Keypoint.LEFT_HIP] = 0.2

        strict = YoloPoseEstimator(keypoint_conf=0.5)._convert(result)[0]
        loose = YoloPoseEstimator(keypoint_conf=0.1)._convert(result)[0]

        assert not strict.valid[Keypoint.LEFT_HIP]
        assert loose.valid[Keypoint.LEFT_HIP]
        # Nilai mentah tidak disaring di estimator; hanya ditandai di PersonPose.
        assert strict.keypoints[Keypoint.LEFT_HIP, 0] != 0

    def test_missing_keypoint_conf_treated_as_valid(self) -> None:
        poses = YoloPoseEstimator()._convert(fake_result(1, with_conf=False))
        assert poses[0].n_valid() == N_KEYPOINTS

    def test_no_detection_returns_empty_list(self) -> None:
        """Frame tanpa orang itu normal, bukan kesalahan."""
        est = YoloPoseEstimator()
        empty_boxes = FakeBoxes(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        empty_kp = FakeKeypoints(np.zeros((0, N_KEYPOINTS, 2), np.float32), None)
        assert est._convert(FakeResult(empty_boxes, empty_kp)) == []

    def test_missing_attributes_return_empty_list(self) -> None:
        est = YoloPoseEstimator()
        assert est._convert(FakeResult(None, None)) == []
        assert est._convert(object()) == []

    def test_non_pose_model_output_is_rejected_with_a_clear_message(self) -> None:
        """Memakai detektor biasa, bukan varian -pose, harus gagal keras."""
        est = YoloPoseEstimator()
        boxes = FakeBoxes(
            np.array([[0.0, 0.0, 1.0, 1.0]], np.float32), np.array([0.9], np.float32)
        )
        wrong = FakeKeypoints(np.zeros((1, 5, 2), np.float32), None)
        with pytest.raises(PoseError, match="varian -pose"):
            est._convert(FakeResult(boxes, wrong))


class TestEstimateValidation:
    def test_empty_image_rejected(self) -> None:
        est = YoloPoseEstimator()
        with pytest.raises(PoseError, match="gambar kosong"):
            est.estimate(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(PoseError, match="gambar kosong"):
            est.estimate(None)  # type: ignore[arg-type]


class TestToNumpy:
    def test_passes_through_numpy(self) -> None:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(_to_numpy(arr), arr)

    def test_handles_torch_like_object(self) -> None:
        class TorchLike:
            def __init__(self, data: np.ndarray) -> None:
                self._data = data

            def cpu(self) -> "TorchLike":
                return self

            def numpy(self) -> np.ndarray:
                return self._data

        arr = np.ones((2, 2), dtype=np.float32)
        np.testing.assert_allclose(_to_numpy(TorchLike(arr)), arr)

    def test_handles_plain_list(self) -> None:
        np.testing.assert_allclose(_to_numpy([[1.0, 2.0]]), np.array([[1.0, 2.0]]))


class TestRealInferenceIfAvailable:
    """Integration atas frame CAUCAFall nyata.

    Di-skip bila bundel atau bobotnya tidak ada, supaya suite tetap jalan di mesin
    lain dan di kernel Kaggle.
    """

    CLIP = Path("Dataset CAUCAFall/CAUCAFall/Subject.1/Fall backwards")

    @pytest.fixture(autouse=True)
    def _requirements(self, in_repo_root: Path) -> None:
        if not (in_repo_root / self.CLIP).is_dir():
            pytest.skip("bundel CAUCAFall tidak ada")
        if not (in_repo_root / "models" / DEFAULT_MODEL).exists():
            pytest.skip("bobot pose belum diunduh")

    def test_detects_a_person_on_a_real_frame(self) -> None:
        import cv2

        from pfv.data.caucafall import load_clip

        clip = load_clip(self.CLIP)
        image = cv2.imread(str(clip.frames[0].image_path), cv2.IMREAD_COLOR)
        assert image is not None

        est = YoloPoseEstimator()
        poses = est.estimate(image)

        assert len(poses) >= 1
        p = max(poses, key=lambda q: q.confidence)
        assert p.keypoints.shape == (N_KEYPOINTS, 3)
        assert p.n_valid() > 0
        assert p.bbox_width > 0 and p.bbox_height > 0

    def test_device_is_resolved_to_something_real(self) -> None:
        est = YoloPoseEstimator()
        assert est.device in {"mps", "cuda", "cpu"}

    def test_weights_land_in_models_dir(self, in_repo_root: Path) -> None:
        """Bobot harus di models/, bukan di cache tersembunyi Ultralytics.

        Itu prasyarat Task 12 ke atas, karena kernel Kaggle sering tanpa internet
        dan bobot harus bisa dilampirkan sebagai berkas biasa.
        """
        est = YoloPoseEstimator()
        assert est.weights_path == Path("models") / DEFAULT_MODEL
        assert (in_repo_root / est.weights_path).exists()
