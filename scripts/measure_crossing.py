#!/usr/bin/env python
"""Ukur risiko pertukaran ID antar DUA orang pada ambang pencocokan berbeda.

CAUCAFall hanya berisi satu orang per frame, jadi menaikkan
``minimum_matching_threshold`` ke 0,95 terbukti memperbaiki pemutusan track tetapi
risiko sampingannya -- menukar identitas dua orang berdekatan -- tidak bisa diuji
di sana. Skrip ini mengujinya dengan adegan sintetis yang kebenarannya diketahui.

Adegan: dua orang berkotak 40x160 berjalan saling mendekati lalu berpapasan.
Variasi: jarak vertikal antar orang, kecepatan, dan derau posisi (deterministik).
Pertukaran dihitung dengan membandingkan ID tiap orang SESUDAH berpapasan terhadap
ID-nya SEBELUM berpapasan; kebenaran diketahui karena posisinya kita yang buat.

Ini adegan ideal: kotak bergerak lurus, sementara orang sungguhan berhenti,
berbalik, dan saling menutupi. Hasilnya batas bawah risiko, bukan perkiraannya.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

from pfv.io.frames import Frame, FrameTransform
from pfv.perception.base import N_KEYPOINTS, PersonPose
from pfv.tracking import PersonTracker

FPS = 20.0
W, H = 40.0, 160.0


def person(x: float, y: float, rng: np.random.Generator, noise: float) -> PersonPose:
    jitter = rng.normal(0.0, noise, size=2) if noise > 0 else np.zeros(2)
    x, y = x + jitter[0], y + jitter[1]
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 0], kp[:, 1], kp[:, 2] = x + W / 2, y + H / 2, 0.9
    return PersonPose((x, y, x + W, y + H), kp, 0.9)


SCENARIOS = ("pass", "occlude", "pause")


def run(
    threshold: float, y_gap: float, speed: float, noise: float, seed: int, scenario: str
) -> Tuple[bool, bool]:
    """Kembalikan ``(terjadi_pertukaran, ada_id_baru)`` untuk satu adegan.

    ``pass``    -- berpapasan lurus (mudah; Kalman memprediksinya dengan baik)
    ``occlude`` -- B tidak terdeteksi selama kotaknya menumpuk A, seperti detektor
                   nyata yang menggabungkan dua orang menjadi satu kotak
    ``pause``   -- B berhenti di dalam kotak A selama 6 frame lalu lanjut, sehingga
                   prediksi kecepatan konstan salah tepat saat identitas ditentukan
    """
    rng = np.random.default_rng(seed)
    tracker = PersonTracker(fps=FPS, minimum_matching_threshold=threshold)

    n = 90
    start_gap = 300.0
    first_ids: Dict[str, int] = {}
    last_ids: Dict[str, int] = {}
    all_ids = set()

    paused_at = None
    for i in range(n):
        # A mulai di kiri bergerak ke kanan, B mulai di kanan bergerak ke kiri.
        ax = 100.0 + speed * i
        step_b = i
        if scenario == "pause":
            # B berhenti begitu mencapai tumpang tindih dengan A.
            if paused_at is None and abs(ax - (100.0 + start_gap - speed * i)) < W:
                paused_at = i
            if paused_at is not None:
                step_b = paused_at + max(0, i - paused_at - 6)
        bx = 100.0 + start_gap - speed * step_b

        overlap = abs(ax - bx) < W and abs(y_gap) < H / 2
        poses = [person(ax, 100.0, rng, noise)]
        if not (scenario == "occlude" and overlap):
            poses.append(person(bx, 100.0 + y_gap, rng, noise))
        frame = Frame(
            index=i, image=np.zeros((2, 2, 3), np.uint8),
            timestamp_s=i / FPS, transform=FrameTransform(), source_size=(720, 480),
        )
        out = tracker.update(poses, frame)
        for t in out:
            all_ids.add(t.track_id)
            # Kecocokan dengan kebenaran lewat jarak terdekat ke posisi sebenarnya.
            who = "A" if abs(t.pose.bbox[0] - ax) < abs(t.pose.bbox[0] - bx) else "B"
            if i == 2:
                first_ids[who] = t.track_id
            if i == n - 1:
                last_ids[who] = t.track_id

    # Kedua orang harus sudah berpapasan di frame terakhir, kalau tidak ukurannya
    # kosong: skenario kecepatan rendah pernah lolos 0% hanya karena tidak sempat
    # bertemu.
    assert ax > bx, f"adegan tidak berpapasan (speed={speed})"

    swapped = any(first_ids.get(k) is not None and last_ids.get(k) is not None
                  and first_ids[k] != last_ids[k] for k in ("A", "B"))
    return swapped, len(all_ids) > 2


def main() -> int:
    gaps = (0.0, 10.0, 25.0, 50.0)
    speeds = (4.0, 6.0, 8.0)
    noises = (0.0, 2.0, 4.0)
    seeds = range(5)

    print(f"{'ambang':>8s} {'adegan':>8s} {'skenario':>9s} {'tertukar':>11s} {'ID baru':>11s}")
    for scenario in SCENARIOS:
        for threshold in (0.8, 0.9, 0.95, 0.99):
            total = swapped = fragmented = 0
            for gap, speed, noise, seed in itertools.product(gaps, speeds, noises, seeds):
                s, f = run(threshold, gap, speed, noise, seed, scenario)
                total += 1
                swapped += s
                fragmented += f
            print(f"{threshold:8.2f} {scenario:>8s} {total:9d} "
                  f"{swapped:5d} ({100*swapped/total:3.0f}%) "
                  f"{fragmented:5d} ({100*fragmented/total:3.0f}%)")
        print()

    print("gap 0 = berpapasan sejajar sempurna (kasus terburuk untuk IoU).")
    print("tertukar = identitas A/B berbeda antara awal dan akhir.")
    print("ID baru  = muncul lebih dari 2 track_id (track terputus).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
