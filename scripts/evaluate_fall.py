#!/usr/bin/env python
"""Evaluasi akhir pada subjek UJI (9-10). Dijalankan SEKALI.

Prasyarat, semuanya diperiksa di sini dan bukan sekadar disarankan:

1. ``eval/frozen_models.json`` ada dan hash isinya cocok. Ambang dibekukan oleh
   ``tune_fall.py`` sebelum skrip ini ada artinya.
2. ``eval/gate.yaml`` ada. Kriteria gerbang harus sudah tertulis sebelum angka uji
   terlihat; hash-nya dicatat bersama hasilnya.
3. Konfigurasi beku ini belum pernah dievaluasi pada subjek uji. Kunci ada di
   ``eval/test_runs.jsonl``. Ulangan hanya lewat ``--allow-rerun`` dan tercatat
   sebagai ulangan.

Catatan dicatat SEBELUM hasil dicetak, sehingga skrip yang macet setelah angka
terlihat tidak bisa diulang secara diam-diam.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from pfv.config import load_config
from pfv.evaluation import baselines
from pfv.evaluation.gate import (
    TestRunLockedError,
    check_lock,
    file_hash,
    judge,
    load_gate,
    record_run,
)
from pfv.evaluation.search import (
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VALIDATION_SUBJECTS,
    evaluate,
    fingerprint,
    load_clips,
    spec_from_json,
)


def verify_frozen(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = data.pop("fingerprint", None)
    if stored != fingerprint(data):
        raise SystemExit(
            f"DITOLAK: {path} berubah sejak dibekukan (hash tersimpan {stored}, "
            f"hash sekarang {fingerprint(data)}). Jalankan ulang scripts/tune_fall.py."
        )
    data["fingerprint"] = stored
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("data/interim/caucafall/pose"))
    parser.add_argument("--frozen", type=Path, default=Path("eval/frozen_models.json"))
    parser.add_argument("--gate", type=Path, default=Path("eval/gate.yaml"))
    parser.add_argument("--log", type=Path, default=Path("eval/test_runs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/test_results.json"))
    parser.add_argument("--allow-rerun", action="store_true")
    args = parser.parse_args(argv)

    frozen = verify_frozen(args.frozen)
    gate = load_gate(args.gate)
    cfg = load_config()
    base = cfg.fall

    if frozen["persistence_s"] != base.timing.persistence_s:
        raise SystemExit("DITOLAK: persistence_s di config berbeda dari yang dibekukan.")

    try:
        previous = check_lock(args.log, frozen["fingerprint"], allow_rerun=args.allow_rerun)
    except TestRunLockedError as exc:
        print(f"DITOLAK: {exc}")
        return 2

    # Dicatat SEBELUM hasil dicetak.
    record_run(
        args.log, fingerprint=frozen["fingerprint"], gate_hash=file_hash(args.gate),
        rerun=previous is not None, verdict=None,
    )
    if previous is not None:
        print("PERINGATAN: ini ULANGAN. Angka di bawah tidak lagi tak terlihat.\n")

    test = load_clips(args.pose_dir, TEST_SUBJECTS, cfg, allow_test=True)
    insample = load_clips(args.pose_dir, TRAIN_SUBJECTS + VALIDATION_SUBJECTS, cfg)
    n_fall = sum(c.is_fall for c in test)
    print(f"Subjek uji {TEST_SUBJECTS}: {n_fall} klip jatuh, {len(test) - n_fall} klip ADL")
    print(f"persistence_s = {base.timing.persistence_s} (tidak dituning)")
    print(f"konfigurasi beku {frozen['fingerprint']} | gerbang {file_hash(args.gate)}\n")

    results = {}
    for model in baselines.MODEL_ORDER:
        spec = spec_from_json(frozen["models"][model]["spec"])
        summary, outcomes = evaluate(spec, test, base)
        in_summary, _ = evaluate(spec, insample, base)
        results[model] = (spec, summary, in_summary, outcomes)

    print("=" * 100)
    print("HASIL PADA SUBJEK UJI (interval Wilson 95%)")
    print("=" * 100)
    print(f"{'model':38s} {'recall (mungkin)':>26s} {'alarm palsu':>26s} {'latensi med':>12s}")
    for model in baselines.MODEL_ORDER:
        _, s, _, _ = results[model]
        lat = "-" if s.latency_median_s is None else f"{s.latency_median_s:.1f}s"
        print(f"{baselines.MODEL_LABELS[model]:38s} {s.achievable_recall.format():>26s} "
              f"{s.false_alarm_clips.format():>26s} {lat:>12s}")

    print("\nRecall mentah (semua klip jatuh, termasuk yang secara struktural tak mungkin):")
    for model in baselines.MODEL_ORDER:
        print(f"  {baselines.MODEL_LABELS[model]:38s} {results[model][1].recall.format()}")

    print("\n" + "=" * 100)
    print("PER ARAH JATUH. Uji (n=2 per aktivitas, TIDAK untuk inferensi) | latih+validasi (in-sample)")
    print("=" * 100)
    print(f"{'model':38s} {'searah kamera: uji':>20s} {'menyamping: uji':>18s} "
          f"{'searah kamera: in-sample':>26s}")
    for model in baselines.MODEL_ORDER:
        _, s, ins, _ = results[model]
        print(f"{baselines.MODEL_LABELS[model]:38s} {s.camera_axis_recall.format():>20s} "
              f"{s.lateral_recall.format():>18s} {ins.camera_axis_recall.format():>26s}")

    print("\n" + "=" * 100)
    print("PASANGAN: klip jatuh yang berbeda hasilnya antara model penuh dan tiap baseline")
    print("=" * 100)
    full_out = {o.name: o.alerted for o in results["full"][3] if o.is_fall}
    for model in baselines.MODEL_ORDER[:-1]:
        other = {o.name: o.alerted for o in results[model][3] if o.is_fall}
        only_full = sum(1 for k in full_out if full_out[k] and not other[k])
        only_other = sum(1 for k in full_out if other[k] and not full_out[k])
        print(f"  vs {baselines.MODEL_LABELS[model]:38s} hanya penuh {only_full:2d} | hanya baseline {only_other:2d}")

    verdict = judge(results["full"][1], gate)
    print("\n" + "=" * 100)
    print(f"GERBANG (model penuh): {verdict.label}")
    print("=" * 100)
    for c in verdict.criteria:
        print(f"  [{'ya' if c.passed else 'TIDAK':>5s}] {c.name:40s} perlu {c.required:>9s}  teramati {c.observed}")
    if verdict.passed:
        print("\n-> Task 16-18 (model temporal) DILEWATI; kuota GPU untuk PPE.")
    else:
        print("\n-> Task 16-18 (model temporal) DIJALANKAN.")
    print("\nn=20 per kelompok: vonis ini keputusan penyaringan, bukan bukti statistik.")
    print("Generalisasi yang sesungguhnya harus diuji di UR Fall dan Multiple Cameras Fall.")

    record_run(
        args.log, fingerprint=frozen["fingerprint"], gate_hash=file_hash(args.gate),
        rerun=previous is not None, verdict=verdict.label,
    )
    payload = {
        "frozen_fingerprint": frozen["fingerprint"],
        "gate_hash": file_hash(args.gate),
        "verdict": verdict.label,
        "models": {
            m: {
                "spec": frozen["models"][m]["spec"],
                "recall": [results[m][1].recall.k, results[m][1].recall.n],
                "achievable_recall": [results[m][1].achievable_recall.k, results[m][1].achievable_recall.n],
                "false_alarm_clips": [results[m][1].false_alarm_clips.k, results[m][1].false_alarm_clips.n],
                "latency_median_s": results[m][1].latency_median_s,
                "outcomes": [
                    {"clip": o.name, "alerted": o.alerted, "latency_s": o.latency_s}
                    for o in results[m][3]
                ],
            }
            for m in baselines.MODEL_ORDER
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nhasil disimpan: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
