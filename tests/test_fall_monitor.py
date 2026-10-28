"""Test FallMonitor dan putar ulang: state machine bersama tracker nyata."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from pfv.config import FallConfig
from pfv.fall import FallMonitor, FallState
from pfv.fall.replay import NATIVE_FPS, clip_activity, clip_subject, replay_clip
from pfv.io.frames import Frame, FrameTransform
from pfv.perception.base import PersonPose
from pfv.tracking import PersonTracker
from pfv_test_helpers import (
    lying_bbox,
    lying_keypoints,
    standing_bbox,
    standing_keypoints,
)

FPS = 20.0


def frame(i: int) -> Frame:
    return Frame(
        index=i, image=np.zeros((2, 2, 3), np.uint8), timestamp_s=i / FPS,
        transform=FrameTransform(), source_size=(720, 480),
    )


def standing_pose() -> PersonPose:
    return PersonPose(standing_bbox(), standing_keypoints(), 0.9)


def lying_pose() -> PersonPose:
    # Diletakkan di lokasi yang sama dengan orang berdiri, agar tracker
    # mempertahankan identitas bila ia mau.
    return PersonPose(
        lying_bbox(y=190.0, left=40.0), lying_keypoints(y=190.0, left=40.0), 0.9
    )


class TestMonitorWithRealTracker:
    def _run(self, poses: List[PersonPose]) -> List[dict]:
        cfg = FallConfig()
        tracker = PersonTracker(fps=FPS)
        monitor = FallMonitor(cfg)
        out = []
        for i, pose in enumerate(poses):
            tracked = tracker.update([pose], frame(i))
            out.append(monitor.update(tracker, tracked, i / FPS))
        return out

    def test_standing_person_never_alerts(self) -> None:
        decisions = self._run([standing_pose()] * 100)
        assert not any(d.alert for step in decisions for d in step.values())

    def test_person_who_lies_down_alerts_exactly_once(self) -> None:
        decisions = self._run([standing_pose()] * 20 + [lying_pose()] * 120)
        alerts = [d for step in decisions for d in step.values() if d.alert]
        assert len(alerts) == 1
        assert alerts[0].state is FallState.FALLEN

    def test_empty_frames_produce_no_decisions(self) -> None:
        tracker = PersonTracker(fps=FPS)
        monitor = FallMonitor(FallConfig())
        assert monitor.update(tracker, tracker.update([], frame(0)), 0.0) == {}

    def test_machines_for_dropped_tracks_are_forgotten(self) -> None:
        cfg = FallConfig()
        tracker = PersonTracker(fps=FPS, prune_after_s=1.0)
        monitor = FallMonitor(cfg)
        for i in range(5):
            monitor.update(tracker, tracker.update([standing_pose()], frame(i)), i / FPS)
        assert len(monitor) == 1
        for i in range(5, 80):
            monitor.update(tracker, tracker.update([], frame(i)), i / FPS)
        assert len(monitor) == 0

    def test_reset(self) -> None:
        tracker = PersonTracker(fps=FPS)
        monitor = FallMonitor(FallConfig())
        monitor.update(tracker, tracker.update([standing_pose()], frame(0)), 0.0)
        monitor.reset()
        assert len(monitor) == 0


POSE_DIR = Path("data/interim/caucafall/pose")
TRAIN = (1, 2, 3, 4, 5, 6)


class TestReplayOnRealKeypointsIfPresent:
    """Memutar ulang keypoint tersimpan. Di-skip bila tidak ada di mesin ini.

    Sengaja HANYA subjek latih, dan sengaja tidak mengunci angka recall. Mengunci
    recall akan mengubah test ini menjadi alat untuk menyetel ambang, padahal itu
    pekerjaan Task 7b. Yang dikunci hanya kontrak: replay deterministik, tidak
    pernah menyentuh subjek held-out, dan ADL klasik tidak memicu alert.
    """

    @pytest.fixture(autouse=True)
    def _needs_data(self, in_repo_root: Path) -> None:
        if not any((in_repo_root / POSE_DIR).glob("s01_*.npz")):
            pytest.skip("keypoint tersimpan tidak ada; jalankan scripts/extract_pose.sh")

    def _alerts(self, name: str) -> int:
        cfg = FallConfig()
        path = POSE_DIR / f"{name}.npz"
        monitor = FallMonitor(cfg)
        n = 0
        for step in replay_clip(path):
            for d in monitor.update(step.tracker, step.tracked, step.timestamp_s).values():
                n += int(d.alert)
        return n

    def test_replay_is_deterministic(self) -> None:
        assert self._alerts("s01_fall_right") == self._alerts("s01_fall_right")

    def test_replay_helpers_parse_clip_names(self) -> None:
        path = POSE_DIR / "s03_fall_forward.npz"
        assert clip_subject(path) == 3
        assert clip_activity(path) == "fall forward"

    @pytest.mark.parametrize("name", ["s01_walk", "s02_hop", "s05_kneel", "s03_kneel"])
    def test_plain_activities_do_not_alert(self, name: str) -> None:
        path = POSE_DIR / f"{name}.npz"
        if not path.exists():
            pytest.skip(f"{name} tidak ada")
        assert self._alerts(name) == 0

    def test_a_clear_lateral_fall_alerts(self) -> None:
        assert self._alerts("s01_fall_right") >= 1


class TestHeldOutGuard:
    """Penjaga subjek held-out pada skrip evaluasi."""

    def test_replay_script_refuses_heldout_subjects(self, in_repo_root: Path, capsys) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "replay_fall", in_repo_root / "scripts" / "replay_fall.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        rc = module.main(["--subjects", "9-10"])
        out = capsys.readouterr().out
        assert rc == 2
        assert "DITOLAK" in out

    def test_replay_script_refuses_a_mixed_range_that_includes_heldout(
        self, in_repo_root: Path, capsys
    ) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "replay_fall2", in_repo_root / "scripts" / "replay_fall.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        assert module.main(["--subjects", "1-8"]) == 2
        assert "DITOLAK" in capsys.readouterr().out
