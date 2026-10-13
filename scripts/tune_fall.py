#!/usr/bin/env python
"""Cari dan bekukan ambang untuk model penuh dan semua baseline.

Memakai subjek latih (1-6) dan validasi (7-8) SAJA. Subjek uji (9-10) tidak
dimuat: ``load_clips`` menolaknya, dan ada test yang mengunci itu.

Keluaran ``eval/frozen_models.json`` memuat parameter terpilih tiap model, jejak
seleksi, daftar klip yang dipakai, dan hash. ``evaluate_fall.py`` menolak berjalan
bila isinya berubah sejak dibekukan.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pfv.config import load_config
from pfv.evaluation import baselines
from pfv.evaluation.search import (
    MAX_TRAIN_FALSE_ALARM_CLIPS,
    RECALL_TOLERANCE_CLIPS,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VALIDATION_SUBJECTS,
    candidates,
    fingerprint,
    load_clips,
    select,
    spec_to_json,
)


def summary_json(s):
    return {
        "recall": [s.recall.k, s.recall.n],
        "achievable_recall": [s.achievable_recall.k, s.achievable_recall.n],
        "false_alarm_clips": [s.false_alarm_clips.k, s.false_alarm_clips.n],
        "camera_axis_recall": [s.camera_axis_recall.k, s.camera_axis_recall.n],
        "latency_median_s": s.latency_median_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--out", type=Path, default=Path("eval/frozen_models.json"))
    args = parser.parse_args()

    cfg = load_config()
    base = cfg.fall

    started = time.perf_counter()
    train = load_clips(args.pose_dir, TRAIN_SUBJECTS, cfg)
    validation = load_clips(args.pose_dir, VALIDATION_SUBJECTS, cfg)
    print(f"fitur ter-cache: {len(train)} klip latih, {len(validation)} klip validasi "
          f"({time.perf_counter() - started:.0f}s)")
    print(f"subjek uji {TEST_SUBJECTS} TIDAK dimuat\n")

    frozen = {}
    for model in baselines.MODEL_ORDER:
        specs = candidates(model)
        t0 = time.perf_counter()
        chosen, shortlist, n_eligible = select(specs, train, validation, base)
        print(f"{baselines.MODEL_LABELS[model]}")
        print(f"  kandidat {len(specs)} | lolos batas alarm palsu latih {n_eligible} | "
              f"shortlist {len(shortlist)} | {time.perf_counter() - t0:.0f}s")
        print(f"  terpilih : {chosen.spec.label()}")
        print(f"  latih    : recall {chosen.train.recall.k}/{chosen.train.recall.n}, "
              f"alarm palsu {chosen.train.false_alarm_clips.k}/{chosen.train.false_alarm_clips.n}")
        v = chosen.validation
        print(f"  validasi : recall {v.recall.k}/{v.recall.n}, "
              f"alarm palsu {v.false_alarm_clips.k}/{v.false_alarm_clips.n}\n")

        frozen[model] = {
            "spec": spec_to_json(chosen.spec),
            "train": summary_json(chosen.train),
            "validation": summary_json(chosen.validation),
            "n_candidates": len(specs),
            "n_eligible": n_eligible,
            "n_shortlist": len(shortlist),
        }

    payload = {
        "persistence_s": base.timing.persistence_s,
        "selection_rule": {
            "max_train_false_alarm_clips": MAX_TRAIN_FALSE_ALARM_CLIPS,
            "recall_tolerance_clips": RECALL_TOLERANCE_CLIPS,
        },
        "train_clips": sorted(c.name for c in train),
        "validation_clips": sorted(c.name for c in validation),
        "models": frozen,
    }
    payload["fingerprint"] = fingerprint(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"dibekukan: {args.out}  (hash {payload['fingerprint']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
