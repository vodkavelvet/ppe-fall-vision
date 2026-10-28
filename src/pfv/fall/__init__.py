"""Fitur dan penalaran deteksi jatuh.

Semua di :mod:`pfv.fall.features`, :mod:`pfv.fall.posture`, dan
:mod:`pfv.fall.state_machine` adalah FUNGSI MURNI atas keypoint dan waktu: tanpa
OpenCV, tanpa model, tanpa berkas. Itu yang membuatnya bisa diuji dengan pose
sintetis dalam milidetik, dan itu penting karena kesalahan di lapisan ini tidak
memunculkan error -- hanya angka yang salah.

Hanya :mod:`pfv.fall.monitor` dan :mod:`pfv.fall.replay` yang menyentuh tracker.
"""

from pfv.fall.features import (
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
from pfv.fall.monitor import FallMonitor
from pfv.fall.posture import Posture, PostureResult, classify_posture
from pfv.fall.state_machine import FallDecision, FallState, FallStateMachine

__all__ = [
    "FallDecision",
    "FallFeatures",
    "FallMonitor",
    "FallState",
    "FallStateMachine",
    "Posture",
    "PostureResult",
    "StaticFeatures",
    "TemporalFeatures",
    "bbox_diagonal",
    "classify_posture",
    "compute_static",
    "compute_temporal",
    "head_hip_dy_norm",
    "hip_ankle_dy_norm",
    "hip_knee_dy_norm",
    "keypoint_spread_y_norm",
    "trunk_angle_deg",
    "trunk_length_norm",
    "trunk_vector",
]
