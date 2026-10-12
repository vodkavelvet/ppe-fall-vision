"""Estimator pose berbasis Ultralytics YOLO-pose.

Satu-satunya modul yang mengimpor ``ultralytics``. Sisa program hanya mengenal
:class:`pfv.perception.base.PersonPose`, sehingga penggantian ke alternatif
Apache-2.0 nanti cukup menambah satu file di sini.

Dua hal yang ditangani secara eksplisit:

* **Bobot disimpan di ``models/``** dengan path yang ditulis di config, bukan di
  cache tersembunyi Ultralytics. Alasannya praktis: Task 12 ke atas berjalan di
  kernel Kaggle yang sering tanpa internet, jadi bobot harus bisa dilampirkan
  sebagai berkas biasa.
* **Keypoint ber-confidence rendah tidak disaring di sini.** Nilainya diteruskan
  apa adanya; penandaan valid atau tidak dilakukan :class:`PersonPose`. Dengan
  begitu ambangnya bisa diubah tanpa menjalankan ulang inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from pfv.perception.base import (
    DEFAULT_KEYPOINT_CONF,
    N_KEYPOINTS,
    PersonPose,
    PoseError,
    resolve_device,
)

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo11n-pose.pt"
DEFAULT_MODELS_DIR = Path("models")


class YoloPoseEstimator:
    """Pembungkus Ultralytics YOLO-pose yang memenuhi Protocol ``PoseEstimator``.

    Model dimuat malas (lazy) pada pemakaian pertama, supaya mengimpor modul ini
    atau membuat objeknya tidak langsung menarik torch dan membaca bobot. Itu yang
    membuat unit test tetap cepat.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        models_dir: Path = DEFAULT_MODELS_DIR,
        device: str = "auto",
        conf: float = 0.25,
        keypoint_conf: float = DEFAULT_KEYPOINT_CONF,
        imgsz: int = 640,
    ) -> None:
        if not 0.0 < conf <= 1.0:
            raise PoseError(f"conf harus di (0, 1], diberikan {conf}")
        if not 0.0 <= keypoint_conf <= 1.0:
            raise PoseError(f"keypoint_conf harus di [0, 1], diberikan {keypoint_conf}")
        if imgsz <= 0:
            raise PoseError(f"imgsz harus positif, diberikan {imgsz}")

        self.model_name = model
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.keypoint_conf = keypoint_conf
        self.imgsz = imgsz

        self._requested_device = device
        self._device: Optional[str] = None
        self._model: Optional["YOLO"] = None

    # ------------------------------------------------------------------ setup

    @property
    def weights_path(self) -> Path:
        """Lokasi bobot. Nama tanpa direktori diselesaikan ke ``models/``."""
        candidate = Path(self.model_name)
        if candidate.parent != Path("."):
            return candidate
        return self.models_dir / candidate.name

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._requested_device)
            logger.info("pose device: %s", self._device)
        return self._device

    def load(self) -> "YOLO":
        """Muat model. Idempoten; pemanggilan berikutnya memakai yang sudah ada."""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise PoseError(
                "ultralytics belum terpasang. Jalankan: "
                "uv pip install -e '.[perception]'"
            ) from exc

        path = self.weights_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            logger.info("memuat bobot pose dari %s", path)
            model = YOLO(str(path))
        else:
            # Ultralytics mengunduh berdasarkan nama berkas. Diberi path lengkap
            # agar bobotnya mendarat di models/ dan bisa dipakai ulang offline.
            logger.info("bobot pose belum ada, mengunduh ke %s", path)
            model = YOLO(str(path))
            if not path.exists():
                # Sebagian versi mengunduh ke direktori kerja; pindahkan agar
                # lokasinya tetap sesuai config.
                fallback = Path(path.name)
                if fallback.exists():
                    fallback.replace(path)
                    logger.info("bobot dipindahkan dari %s ke %s", fallback, path)

        self._model = model
        return model

    # -------------------------------------------------------------- inference

    def estimate(self, image: np.ndarray) -> List[PersonPose]:
        """Deteksi semua orang pada satu gambar BGR.

        Mengembalikan daftar kosong bila tidak ada orang -- itu keadaan normal,
        bukan kesalahan, dan terjadi di banyak frame CAUCAFall saat subjek belum
        masuk ke bidang kamera.
        """
        if image is None or image.size == 0:
            raise PoseError("gambar kosong")

        model = self.load()
        results = model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._convert(results[0])

    def _convert(self, result: object) -> List[PersonPose]:
        """Ubah satu hasil Ultralytics menjadi daftar :class:`PersonPose`.

        Dipisahkan dari :meth:`estimate` supaya bisa diuji dengan objek hasil
        palsu, tanpa memuat model dan tanpa GPU.
        """
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        xyxy = np.asarray(_to_numpy(boxes.xyxy), dtype=np.float32)
        det_conf = np.asarray(_to_numpy(boxes.conf), dtype=np.float32)

        kp_xy = np.asarray(_to_numpy(keypoints.xy), dtype=np.float32)
        kp_conf_raw = getattr(keypoints, "conf", None)
        kp_conf = None if kp_conf_raw is None else np.asarray(
            _to_numpy(kp_conf_raw), dtype=np.float32
        )

        if kp_xy.ndim != 3 or kp_xy.shape[1] != N_KEYPOINTS:
            raise PoseError(
                f"keypoints model berbentuk {kp_xy.shape}, diharapkan (N, {N_KEYPOINTS}, 2). "
                "Pastikan model yang dipakai adalah varian -pose, bukan detektor biasa."
            )

        poses: List[PersonPose] = []
        for i in range(kp_xy.shape[0]):
            kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
            kp[:, :2] = kp_xy[i, :, :2]
            # Model tanpa confidence per keypoint dianggap valid penuh: tidak ada
            # dasar untuk menyaring, dan menebak nol akan menandai semuanya invalid.
            kp[:, 2] = 1.0 if kp_conf is None else kp_conf[i]

            poses.append(
                PersonPose(
                    bbox=(
                        float(xyxy[i, 0]),
                        float(xyxy[i, 1]),
                        float(xyxy[i, 2]),
                        float(xyxy[i, 3]),
                    ),
                    keypoints=kp,
                    confidence=float(det_conf[i]) if i < det_conf.size else 0.0,
                    keypoint_conf=self.keypoint_conf,
                )
            )
        return poses

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"YoloPoseEstimator(model={self.model_name!r}, "
            f"device={self._device or self._requested_device!r}, "
            f"conf={self.conf}, keypoint_conf={self.keypoint_conf}, imgsz={self.imgsz})"
        )


def _to_numpy(value: object) -> np.ndarray:
    """Ambil array numpy dari tensor torch maupun array biasa."""
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        return numpy_fn()
    return np.asarray(value)
