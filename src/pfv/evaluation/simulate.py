"""Jalankan state machine atas fitur yang sudah di-cache."""

from __future__ import annotations

from typing import Optional

from pfv.config import FallConfig
from pfv.evaluation.cache import ClipData
from pfv.evaluation.metrics import ClipOutcome
from pfv.fall.state_machine import FallStateMachine, PostureFn


def simulate_clip(
    clip: ClipData, cfg: FallConfig, posture_fn: Optional[PostureFn] = None
) -> ClipOutcome:
    """Satu mesin per track, persis seperti :class:`~pfv.fall.monitor.FallMonitor`.

    Tidak ada pemrosesan ulang fitur; hanya state machine yang dijalankan, sehingga
    cukup cepat untuk grid search.
    """
    alerts = []
    for seq in clip.tracks:
        machine = FallStateMachine(cfg, posture_fn=posture_fn)
        for t, static, temporal in zip(seq.times, seq.static, seq.temporal):
            if machine.update(t, static, temporal).alert:
                alerts.append(t)
    alerts.sort()

    first = alerts[0] if alerts else None
    latency = None
    if first is not None and clip.onset_s is not None:
        latency = first - clip.onset_s

    return ClipOutcome(
        name=clip.name,
        subject=clip.subject,
        activity=clip.activity,
        is_fall=clip.is_fall,
        is_camera_axis=clip.is_camera_axis,
        lying_s=clip.lying_s,
        n_alerts=len(alerts),
        first_alert_s=first,
        latency_s=latency,
    )
