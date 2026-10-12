"""Detector di belakang Protocol, supaya implementasinya bisa ditukar.

Ini bukan kerapian belaka. Ultralytics berlisensi AGPL-3.0: aman untuk riset dan
penggunaan internal, tetapi menular bila produknya didistribusikan. Dengan
antarmuka di :mod:`pfv.perception.base`, penggantian ke alternatif Apache-2.0
seperti RTMPose/MMPose atau YOLOX tidak menyentuh pipeline sama sekali.
"""

from pfv.perception.base import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    HEAD_KEYPOINTS,
    KNEE_KEYPOINTS,
    N_KEYPOINTS,
    TORSO_KEYPOINTS,
    Keypoint,
    PersonPose,
    PoseEstimator,
)

__all__ = [
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON",
    "HEAD_KEYPOINTS",
    "KNEE_KEYPOINTS",
    "N_KEYPOINTS",
    "TORSO_KEYPOINTS",
    "Keypoint",
    "PersonPose",
    "PoseEstimator",
]
