#!/usr/bin/env python
"""Putar ulang keypoint tersimpan melalui tracker + state machine, per klip.

Ini PRATINJAU Task 7b, bukan evaluasi akhirnya. Tujuannya melihat perilaku state
machine pada data nyata: klip jatuh mana yang menghasilkan alert, seberapa lambat,
dan klip ADL mana yang menghasilkan alarm palsu.

**Penjaga subjek.** Secara bawaan hanya subjek latih (1-6) yang diputar. Subjek
validasi (7-8) dan uji (9-10) ditolak kecuali ``--include-heldout`` diberikan.
Angka dari subjek uji hanya boleh dilihat SEKALI, di akhir Task 7b, setelah semua
ambang dibekukan. Melihatnya lebih awal lalu menyetel ulang ambang membuat angka
uji tidak lagi tak terlihat.

Definisi hasil per klip jatuh, per KEJADIAN (bukan per frame):

* terdeteksi : ada alert pada klip itu
* latensi    : waktu alert dikurangi awal periode tergeletak pada ground truth.
               Negatif berarti alert mendahului label, yang terjadi bila label
               mulai terlambat dari saat orang sebenarnya sudah jatuh.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from pfv.config import load_config
from pfv.fall import FallMonitor
from pfv.fall.replay import clip_activity, clip_subject, replay_clip

TRAIN = (1, 2, 3, 4, 5, 6)
HELDOUT = (7, 8, 9, 10)
FALL_ACTIVITIES = ("fall backwards", "fall forward", "fall left", "fall right", "fall sitting")


def parse_subjects(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_clip(path: Path, cfg) -> Dict[str, object]:
    monitor = FallMonitor(cfg.fall, max_gap_s=cfg.tracking.max_gap_s)
    t = cfg.tracking
    tracker_kwargs = dict(
        window_s=t.window_s,
        track_activation_threshold=t.track_activation_threshold,
        lost_track_buffer=t.lost_track_buffer,
        minimum_matching_threshold=t.minimum_matching_threshold,
        minimum_consecutive_frames=t.minimum_consecutive_frames,
        prune_after_s=t.prune_after_s,
    )

    flags = np.load(path)["fall_flags"]
    onset_idx = int(np.argmax(flags)) if flags.any() else None
    onset_s = onset_idx / 20.0 if onset_idx is not None else None

    alerts: List[float] = []
    ids = set()
    for step in replay_clip(path, keypoint_conf=cfg.pose.keypoint_conf, tracker_kwargs=tracker_kwargs):
        for tid in (p.track_id for p in step.tracked):
            ids.add(tid)
        for d in monitor.update(step.tracker, step.tracked, step.timestamp_s).values():
            if d.alert:
                alerts.append(d.timestamp_s)

    return {
        "alerts": alerts,
        "onset_s": onset_s,
        "n_ids": len(ids),
        "duration_s": len(flags) / 20.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--subjects", default="1-6")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--include-heldout", action="store_true",
                        help="izinkan subjek 7-10. Lihat peringatan di docstring.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    subjects = parse_subjects(args.subjects)
    blocked = sorted(set(subjects) & set(HELDOUT))
    if blocked and not args.include_heldout:
        print(
            f"DITOLAK: subjek {blocked} adalah subjek validasi/uji. Angkanya hanya boleh "
            "dilihat sekali di akhir Task 7b, setelah semua ambang dibekukan. Pakai "
            "--include-heldout bila memang itu yang dimaksud.",
        )
        return 2
    if blocked:
        print(f"PERINGATAN: memutar subjek held-out {blocked}. Jangan menyetel ambang "
              "berdasarkan angka ini.\n")

    cfg = load_config(args.config)
    rows = []
    for path in sorted(args.pose_dir.glob("*.npz")):
        if clip_subject(path) not in subjects:
            continue
        r = run_clip(path, cfg)
        r.update(subject=clip_subject(path), activity=clip_activity(path), name=path.stem)
        rows.append(r)

    fall_rows = [r for r in rows if r["activity"] in FALL_ACTIVITIES]
    adl_rows = [r for r in rows if r["activity"] not in FALL_ACTIVITIES]

    print(f"Subjek {subjects} | {len(fall_rows)} klip jatuh, {len(adl_rows)} klip ADL")
    print(f"persistence_s={cfg.fall.timing.persistence_s}  "
          f"orientasi>={cfg.fall.posture.min_orientation_votes}  "
          f"anggota badan>={cfg.fall.posture.min_limb_votes}\n")

    print("=" * 78)
    print("KLIP JATUH: terdeteksi per KEJADIAN dan latensi dari awal periode tergeletak")
    print("=" * 78)
    print(f"{'aktivitas':16s} {'terdeteksi':>11s} {'latensi median':>16s} {'latensi min..maks':>20s} {'>1 alert':>9s}")
    latencies_all: List[float] = []
    n_detected = 0
    for activity in FALL_ACTIVITIES:
        group = [r for r in fall_rows if r["activity"] == activity]
        if not group:
            continue
        hit = [r for r in group if r["alerts"]]
        lat = [r["alerts"][0] - r["onset_s"] for r in hit if r["onset_s"] is not None]
        latencies_all += lat
        n_detected += len(hit)
        multi = sum(1 for r in group if len(r["alerts"]) > 1)
        med = f"{statistics.median(lat):6.1f}s" if lat else "-"
        span = f"{min(lat):.1f}..{max(lat):.1f}s" if lat else "-"
        print(f"{activity:16s} {len(hit):>5d}/{len(group):<5d} {med:>16s} {span:>20s} {multi:>9d}")
        if args.verbose:
            for r in group:
                if not r["alerts"]:
                    print(f"     TERLEWAT {r['name']}  (id {r['n_ids']})")

    print("-" * 78)
    total = len(fall_rows)
    print(f"{'SEMUA':16s} {n_detected:>5d}/{total:<5d}  "
          f"recall per kejadian {n_detected / total:.0%}" if total else "tidak ada klip jatuh")
    if latencies_all:
        print(f"latensi dari awal periode tergeletak: median {statistics.median(latencies_all):.1f}s, "
              f"p90 {np.quantile(latencies_all, 0.9):.1f}s, maks {max(latencies_all):.1f}s")

    print()
    print("=" * 78)
    print("KLIP ADL: alarm palsu per KLIP")
    print("=" * 78)
    print(f"{'aktivitas':16s} {'alarm palsu':>14s}")
    false_clips = 0
    for activity in sorted({r["activity"] for r in adl_rows}):
        group = [r for r in adl_rows if r["activity"] == activity]
        bad = [r for r in group if r["alerts"]]
        false_clips += len(bad)
        print(f"{activity:16s} {len(bad):>8d}/{len(group):<5d}")
        if args.verbose:
            for r in bad:
                print(f"     ALARM {r['name']} pada t={r['alerts'][0]:.1f}s dari {r['duration_s']:.1f}s")
    print("-" * 78)
    adl_minutes = sum(r["duration_s"] for r in adl_rows) / 60.0
    print(f"klip ADL dengan alarm palsu: {false_clips}/{len(adl_rows)}  "
          f"({adl_minutes:.1f} menit ADL)")
    if adl_minutes > 0:
        print(f"setara {false_clips / adl_minutes * 60:.0f} alarm palsu/jam, tetapi dengan "
              f"{false_clips} kejadian ketidakpastiannya sangat lebar; laporkan per klip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
