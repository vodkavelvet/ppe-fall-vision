"""Penggambaran overlay. Dipisahkan agar bisa dipakai CLI, dashboard, dan klip bukti.

Satu pilihan yang disengaja: **keypoint tidak valid tetap digambar**, sebagai titik
abu-abu kecil. Kalau keypoint yang hilang tidak terlihat sama sekali, Anda akan
menduga modelnya bekerja baik padahal pinggulnya sedang tidak terdeteksi -- dan
itu justru kondisi yang membuat ``trunk_angle`` tidak tersedia.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from pfv.perception.base import (
    COCO_SKELETON,
    TORSO_KEYPOINTS,
    Keypoint,
    PersonPose,
)

Colour = Tuple[int, int, int]

COLOUR_BONE: Colour = (0, 220, 255)
COLOUR_TORSO: Colour = (0, 160, 255)
COLOUR_KEYPOINT: Colour = (255, 255, 255)
COLOUR_INVALID: Colour = (120, 120, 120)
COLOUR_BOX: Colour = (0, 200, 0)
COLOUR_BOX_NO_TORSO: Colour = (0, 165, 255)

_TORSO_BONES = frozenset(
    {
        (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER),
        (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP),
        (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP),
        (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP),
    }
)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    colour: Colour = (0, 255, 0),
    scale: float = 0.6,
) -> None:
    """Teks dengan garis luar gelap, agar terbaca di latar terang maupun gelap."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_pose(
    image: np.ndarray,
    pose: PersonPose,
    *,
    label: Optional[str] = None,
    draw_box: bool = True,
    draw_invalid: bool = True,
    draw_trunk: bool = True,
) -> None:
    """Gambar satu orang: kotak, skeleton, dan vektor trunk.

    Vektor trunk digambar karena itulah yang menjadi ``trunk_angle`` di Task 5.
    Melihatnya langsung jauh lebih cepat untuk menilai apakah fiturnya akan
    bekerja daripada membaca angka.
    """
    if draw_box:
        x1, y1, x2, y2 = (int(round(v)) for v in pose.bbox)
        # Kotak berubah warna bila torso tidak lengkap: itu penanda bahwa
        # trunk_angle TIDAK akan tersedia untuk frame ini.
        colour = COLOUR_BOX if pose.has_full_torso() else COLOUR_BOX_NO_TORSO
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        if label:
            draw_text(image, label, (x1, max(14, y1 - 6)), colour)

    for a, b in COCO_SKELETON:
        pa, pb = pose.keypoint(a), pose.keypoint(b)
        if pa is None or pb is None:
            continue
        bone = (a, b) in _TORSO_BONES or (b, a) in _TORSO_BONES
        cv2.line(
            image,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            COLOUR_TORSO if bone else COLOUR_BONE,
            3 if bone else 2,
            cv2.LINE_AA,
        )

    for index in range(len(pose.valid)):
        x, y = pose.keypoints[index, 0], pose.keypoints[index, 1]
        if pose.valid[index]:
            cv2.circle(image, (int(round(x)), int(round(y))), 3, COLOUR_KEYPOINT, -1, cv2.LINE_AA)
        elif draw_invalid and (x != 0 or y != 0):
            cv2.circle(image, (int(round(x)), int(round(y))), 2, COLOUR_INVALID, 1, cv2.LINE_AA)

    if draw_trunk:
        shoulder = pose.shoulder_centre()
        hip = pose.hip_centre()
        if shoulder is not None and hip is not None:
            cv2.arrowedLine(
                image,
                (int(round(shoulder[0])), int(round(shoulder[1]))),
                (int(round(hip[0])), int(round(hip[1]))),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.15,
            )


def draw_poses(
    image: np.ndarray,
    poses: Sequence[PersonPose],
    *,
    labels: Optional[Iterable[str]] = None,
    **kwargs: object,
) -> None:
    """Gambar beberapa orang sekaligus."""
    label_list = list(labels) if labels is not None else [None] * len(poses)
    for pose, label in zip(poses, label_list):
        draw_pose(image, pose, label=label, **kwargs)  # type: ignore[arg-type]
