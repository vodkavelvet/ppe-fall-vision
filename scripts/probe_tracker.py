#!/usr/bin/env python
"""Probe API ByteTrack di supervision 0.30.5.

``sv.ByteTrack`` di versi ini dibungkus dekorator deprecation, sehingga
``inspect.signature`` hanya memperlihatkan signature wrapper-nya. Jadi
perilakunya harus diperiksa dengan menjalankan, bukan dengan membaca signature.

Yang perlu dipastikan sebelum Task 4 ditulis:

1. Parameter konstruktor apa yang benar-benar diterima.
2. Apakah ``update_with_detections`` mengembalikan deteksi dengan urutan yang
   sama, atau menyaring yang tidak terpasangkan. Ini menentukan cara memetakan
   ``tracker_id`` kembali ke ``PersonPose`` -- kalau urutannya berubah tanpa
   penanda, pemetaannya akan salah secara diam-diam.
3. Apakah ``data`` dict ikut terbawa, karena itu cara paling aman membawa indeks
   asli melewati tracker.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")

import supervision as sv  # noqa: E402


def main() -> int:
    print(f"supervision {sv.__version__}")

    print("\n=== parameter konstruktor yang diterima ===")
    for kwargs in (
        {},
        {"frame_rate": 20},
        {"track_activation_threshold": 0.25},
        {"lost_track_buffer": 30},
        {"minimum_matching_threshold": 0.8},
        {"minimum_consecutive_frames": 1},
        {
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 60,
            "minimum_matching_threshold": 0.8,
            "frame_rate": 20,
            "minimum_consecutive_frames": 1,
        },
    ):
        try:
            sv.ByteTrack(**kwargs)
            print(f"  OK      {kwargs}")
        except TypeError as exc:
            print(f"  TOLAK   {kwargs} -> {exc}")

    print("\n=== apakah data dict terbawa dan urutan terjaga ===")
    tracker = sv.ByteTrack(frame_rate=20)

    # Dua orang bergerak berlawanan arah, 6 frame.
    for step in range(6):
        left_x = 50.0 + step * 20
        right_x = 400.0 - step * 20
        xyxy = np.array(
            [[left_x, 100.0, left_x + 40, 260.0], [right_x, 100.0, right_x + 40, 260.0]],
            dtype=np.float32,
        )
        det = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.9, 0.85], dtype=np.float32),
            class_id=np.zeros(2, dtype=int),
            data={"src_index": np.arange(2)},
        )
        out = tracker.update_with_detections(det)

        ids = out.tracker_id.tolist() if out.tracker_id is not None else None
        src = out.data.get("src_index") if out.data else None
        src = src.tolist() if hasattr(src, "tolist") else src
        print(
            f"  frame {step}: masuk 2, keluar {len(out)}, tracker_id={ids}, "
            f"src_index={src}"
        )

    print("\n=== perilaku saat tidak ada deteksi ===")
    empty = sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        class_id=np.zeros(0, dtype=int),
        data={"src_index": np.zeros(0, dtype=int)},
    )
    out = tracker.update_with_detections(empty)
    print(f"  masuk 0, keluar {len(out)}")

    print("\n=== apakah track_id bertahan setelah deteksi hilang sesaat ===")
    tracker2 = sv.ByteTrack(frame_rate=20, lost_track_buffer=30)
    seen = []
    for step in range(10):
        if step in (4, 5):  # orang hilang 2 frame
            det = empty
        else:
            x = 50.0 + step * 15
            det = sv.Detections(
                xyxy=np.array([[x, 100.0, x + 40, 260.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.zeros(1, dtype=int),
                data={"src_index": np.arange(1)},
            )
        out = tracker2.update_with_detections(det)
        ids = out.tracker_id.tolist() if out.tracker_id is not None else []
        seen.append(ids)
    print(f"  id per frame: {seen}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
