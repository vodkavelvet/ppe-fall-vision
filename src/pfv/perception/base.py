"""Kontrak pose estimation. Tanpa dependensi ke model atau framework mana pun.

Keputusan paling penting di modul ini: **keypoint ber-confidence rendah DITANDAI
invalid, bukan dibuang dan bukan diganti nol.**

Alasannya langsung mengenai deteksi jatuh. Kalau keypoint pinggul yang tidak
terdeteksi diganti ``(0, 0)``, maka ``trunk_angle`` akan menghitung sudut ke sudut
kiri-atas frame dan menghasilkan nilai yang tampak masuk akal padahal tidak
berarti apa-apa. Fitur jatuh lalu berubah drastis hanya karena pinggul sesaat
tertutup, dan itu muncul sebagai false positive yang nyaris mustahil dilacak.
Dengan ditandai invalid, Task 5 bisa mengembalikan ``None`` secara jujur dan Task 6
memilih status ``UNKNOWN`` alih-alih menebak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from pfv.io.frames import FrameTransform

N_KEYPOINTS = 17
"""Jumlah keypoint pada skema COCO, yang dipakai model pose pretrained."""


class Keypoint(IntEnum):
    """Indeks keypoint COCO.

    Dipakai sebagai nama alih-alih angka ajaib, karena salah indeks di sini akan
    menghasilkan fitur yang salah tanpa error apa pun.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


COCO_KEYPOINT_NAMES: Tuple[str, ...] = tuple(k.name.lower() for k in Keypoint)

#: Keypoint yang membentuk region kepala. Dipakai Task 8 untuk memastikan helm
#: benar-benar DIPAKAI, bukan sekadar bbox-nya bertumpang tindih dengan orangnya.
HEAD_KEYPOINTS: Tuple[Keypoint, ...] = (
    Keypoint.NOSE,
    Keypoint.LEFT_EYE,
    Keypoint.RIGHT_EYE,
    Keypoint.LEFT_EAR,
    Keypoint.RIGHT_EAR,
)

#: Keypoint yang membentuk torso. Sumber ``trunk_angle`` (Task 5) sekaligus anchor
#: pemeriksaan vest (Task 8).
TORSO_KEYPOINTS: Tuple[Keypoint, ...] = (
    Keypoint.LEFT_SHOULDER,
    Keypoint.RIGHT_SHOULDER,
    Keypoint.LEFT_HIP,
    Keypoint.RIGHT_HIP,
)

#: Lutut. Inilah yang memisahkan BERLUTUT dari JATUH -- dan CAUCAFall memang
#: memuat berlutut sebagai ADL, jadi tanpa ini false positive-nya pasti muncul.
KNEE_KEYPOINTS: Tuple[Keypoint, ...] = (Keypoint.LEFT_KNEE, Keypoint.RIGHT_KNEE)

#: Pasangan keypoint untuk menggambar skeleton.
COCO_SKELETON: Tuple[Tuple[Keypoint, Keypoint], ...] = (
    # kepala
    (Keypoint.NOSE, Keypoint.LEFT_EYE),
    (Keypoint.NOSE, Keypoint.RIGHT_EYE),
    (Keypoint.LEFT_EYE, Keypoint.LEFT_EAR),
    (Keypoint.RIGHT_EYE, Keypoint.RIGHT_EAR),
    # torso
    (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP),
    # lengan
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW),
    (Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW),
    (Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST),
    # kaki
    (Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE),
    (Keypoint.LEFT_KNEE, Keypoint.LEFT_ANKLE),
    (Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE),
    (Keypoint.RIGHT_KNEE, Keypoint.RIGHT_ANKLE),
)

DEFAULT_KEYPOINT_CONF = 0.5


class PoseError(RuntimeError):
    """Keluaran pose tidak berbentuk seperti yang diharapkan."""


@dataclass(frozen=True)
class PersonPose:
    """Satu orang: kotak, 17 keypoint, dan confidence deteksinya.

    ``keypoints`` selalu berbentuk ``(17, 3)`` dengan kolom ``x, y, conf``.
    Bentuknya tidak pernah berubah, bahkan ketika semua keypoint tidak terdeteksi
    -- lapisan di atas tidak perlu menangani jumlah keypoint yang bervariasi.

    Validitas ditentukan oleh ``keypoint_conf``, bukan oleh keberadaan nilai.
    """

    bbox: Tuple[float, float, float, float]
    """``(x1, y1, x2, y2)`` pada koordinat frame yang diproses."""

    keypoints: np.ndarray
    confidence: float
    keypoint_conf: float = DEFAULT_KEYPOINT_CONF
    valid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        kp = np.asarray(self.keypoints, dtype=np.float32)
        if kp.shape != (N_KEYPOINTS, 3):
            raise PoseError(
                f"keypoints harus berbentuk ({N_KEYPOINTS}, 3), diberikan {kp.shape}"
            )
        object.__setattr__(self, "keypoints", kp)
        object.__setattr__(self, "valid", kp[:, 2] >= self.keypoint_conf)

    # ------------------------------------------------------------------ akses

    def keypoint(self, index: int) -> Optional[Tuple[float, float]]:
        """Titik ``(x, y)``, atau ``None`` bila keypoint-nya tidak valid.

        Sengaja ``Optional``: pemanggil dipaksa memikirkan kasus tidak terdeteksi,
        alih-alih menerima koordinat palsu secara diam-diam.
        """
        if not self.valid[index]:
            return None
        x, y = self.keypoints[index, 0], self.keypoints[index, 1]
        return float(x), float(y)

    def keypoint_conf_value(self, index: int) -> float:
        return float(self.keypoints[index, 2])

    def has(self, *indices: int) -> bool:
        """``True`` bila SEMUA keypoint yang diminta valid."""
        return bool(np.all(self.valid[list(indices)])) if indices else False

    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    def midpoint(self, a: int, b: int) -> Optional[Tuple[float, float]]:
        """Titik tengah dua keypoint, atau ``None`` bila salah satu tidak valid.

        Dipakai untuk pusat bahu dan pusat pinggul, yaitu dua ujung vektor trunk
        pada Task 5. Menoleransi satu sisi saja akan membuat vektornya miring
        mengikuti sisi yang terdeteksi dan merusak sudutnya, jadi keduanya wajib.
        """
        pa, pb = self.keypoint(a), self.keypoint(b)
        if pa is None or pb is None:
            return None
        return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0

    def shoulder_centre(self) -> Optional[Tuple[float, float]]:
        return self.midpoint(Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER)

    def hip_centre(self) -> Optional[Tuple[float, float]]:
        return self.midpoint(Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP)

    def has_full_torso(self) -> bool:
        """Kedua bahu dan kedua pinggul valid.

        Prasyarat ``trunk_angle`` (Task 5), dan juga syarat sebelum pelanggaran
        vest boleh dilaporkan (Task 8) -- torso yang tidak jelas harus menghasilkan
        ``UNKNOWN``, bukan tuduhan.
        """
        return self.has(*TORSO_KEYPOINTS)

    # --------------------------------------------------------------- geometri

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_aspect_ratio(self) -> Optional[float]:
        """Rasio tinggi/lebar kotak. ``None`` bila lebarnya nol.

        Fitur deteksi jatuh. Audit CAUCAFall: median 2,46 saat berjalan dan 0,59
        saat tergeletak, jadi pemisahannya sekitar 4,2 kali. Itu juga alasan
        penskalaan frame tidak boleh mendistorsi rasio aspek.
        """
        w = self.bbox_width
        return self.bbox_height / w if w > 0 else None

    def to_source(self, transform: FrameTransform) -> "PersonPose":
        """Petakan ke koordinat frame ASLI.

        Deteksi berjalan pada gambar yang sudah diskalakan, tetapi event, crop
        bukti, dan klip harus merujuk frame asli. Confidence tidak tersentuh;
        yang dipetakan hanya geometri, termasuk keypoint yang tidak valid agar
        bentuk ``(17, 3)`` tetap utuh.
        """
        kp = self.keypoints.copy()
        for i in range(N_KEYPOINTS):
            x, y = transform.to_source(float(kp[i, 0]), float(kp[i, 1]))
            kp[i, 0], kp[i, 1] = x, y
        return PersonPose(
            bbox=transform.to_source_box(self.bbox),
            keypoints=kp,
            confidence=self.confidence,
            keypoint_conf=self.keypoint_conf,
        )

    @classmethod
    def empty(cls, keypoint_conf: float = DEFAULT_KEYPOINT_CONF) -> "PersonPose":
        """Orang tanpa satu pun keypoint valid. Berguna untuk test dan fallback."""
        return cls(
            bbox=(0.0, 0.0, 0.0, 0.0),
            keypoints=np.zeros((N_KEYPOINTS, 3), dtype=np.float32),
            confidence=0.0,
            keypoint_conf=keypoint_conf,
        )


@runtime_checkable
class PoseEstimator(Protocol):
    """Antarmuka estimator pose.

    Implementasi apa pun hanya perlu mengembalikan daftar :class:`PersonPose`
    pada koordinat gambar yang diberikan. Tidak ada tipe khusus framework yang
    boleh menembus batas ini.
    """

    @property
    def device(self) -> str:
        """Device yang benar-benar dipakai, mis. ``"mps"`` atau ``"cpu"``."""
        ...

    def estimate(self, image: np.ndarray) -> List[PersonPose]:
        """Deteksi semua orang pada satu gambar BGR. Kosong bila tak ada orang."""
        ...


def resolve_device(requested: str = "auto") -> str:
    """Pilih device inference.

    ``auto`` mencoba MPS lalu jatuh ke CPU. Permintaan eksplisit yang tidak
    tersedia diturunkan ke CPU dengan peringatan, bukan digagalkan -- lebih baik
    lambat daripada tidak jalan, terutama di kernel Kaggle yang headless.
    """
    import logging

    logger = logging.getLogger(__name__)
    choice = (requested or "auto").strip().lower()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise PoseError(
            "torch belum terpasang. Jalankan: uv pip install -e '.[perception]'"
        ) from exc

    mps_ok = torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    if choice == "auto":
        if mps_ok:
            return "mps"
        if cuda_ok:
            return "cuda"
        return "cpu"

    if choice == "mps" and not mps_ok:
        logger.warning("device 'mps' diminta tetapi tidak tersedia; memakai cpu")
        return "cpu"
    if choice == "cuda" and not cuda_ok:
        logger.warning("device 'cuda' diminta tetapi tidak tersedia; memakai cpu")
        return "cpu"
    return choice


def keypoints_from_xy_conf(
    xy: Sequence[Sequence[float]], conf: Optional[Sequence[float]]
) -> np.ndarray:
    """Susun array ``(17, 3)`` dari koordinat dan confidence terpisah.

    Sebagian keluaran model memisahkan keduanya. Bila ``conf`` tidak ada, keypoint
    dianggap valid penuh -- itu satu-satunya asumsi yang masuk akal, dan model
    yang tidak melaporkan confidence memang tidak memberi dasar untuk menyaring.
    """
    arr = np.asarray(xy, dtype=np.float32)
    if arr.shape[0] != N_KEYPOINTS or arr.shape[1] < 2:
        raise PoseError(f"xy harus ({N_KEYPOINTS}, 2), diberikan {arr.shape}")

    out = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    out[:, :2] = arr[:, :2]
    out[:, 2] = 1.0 if conf is None else np.asarray(conf, dtype=np.float32)
    return out
