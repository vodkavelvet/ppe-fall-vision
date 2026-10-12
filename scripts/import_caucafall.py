#!/usr/bin/env python
"""Impor CAUCAFall menjadi manifes klip dan ground truth evaluasi.

Yang dihasilkan:

* ``eval/fall_events.csv`` -- satu baris per klip, dengan interval jatuh.
  Inilah ground truth yang dipakai harness evaluasi Task 7b.
* ``data/interim/caucafall/manifest.json`` -- daftar frame terurut per klip
  beserta path gambar dan flag jatuh, agar Task 3-7 tidak perlu memindai ulang
  100 direktori setiap kali dijalankan.

**Yang TIDAK dilakukan, dan alasannya.** Tidak ada transkode ke video, dan file
``.avi`` bawaan dataset tidak disentuh. Audit menunjukkan ``.avi`` hanya 1,7
persen dari ukuran PNG dan tiga di antaranya terpotong, sedangkan estimasi pose
adalah satu-satunya masukan deteksi jatuh. PNG lossless dibaca langsung lewat
:class:`pfv.io.ImageSequenceSource`. Karena itu tidak ada opsi ``--prune-png``:
PNG adalah sumber kebenaran, bukan data sementara.

Skrip ini read-only terhadap dataset. Hanya menulis ke ``eval/`` dan
``data/interim/``.

Logikanya sengaja tipis; verifikasi dan pembentukan baris ada di
:mod:`pfv.data.caucafall` supaya bisa diuji.

Pemakaian::

    .venv/bin/python scripts/import_caucafall.py "Dataset CAUCAFall/CAUCAFall"
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from pfv.data.caucafall import (
    EVENT_CSV_FIELDS,
    NATIVE_FPS,
    NATIVE_SIZE,
    Clip,
    clip_to_row,
    discover_clips,
    verify_bundle,
)


def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EVENT_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(clips: Sequence[Clip], path: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "CAUCAFall",
        "source_doi": "10.17632/7w7fccy7ky.4",
        "licence": "CC BY-NC-SA 4.0 (non-komersial, share-alike)",
        "attribution": (
            "Jose Camilo Eraso, Elena Munoz, Mariela Munoz, Jesus Pinto; "
            "Universidad del Cauca; Universidad de la Amazonia"
        ),
        "root": str(root),
        "native_size": list(NATIVE_SIZE),
        "native_fps": NATIVE_FPS,
        "note": (
            "Frame dibaca dari PNG lossless, bukan dari .avi. Tiga .avi bawaan "
            "terpotong dan ukurannya hanya 1,7% dari PNG. Lihat data/DATASETS.md."
        ),
        "clips": [
            {
                "clip_id": c.clip_id,
                "subject_id": c.subject_id,
                "activity": c.activity,
                "kind": c.kind,
                "n_frames": c.n_frames,
                "fall_intervals": c.fall_intervals(),
                "images": [str(p.relative_to(root)) for p in c.image_paths],
                "fall_flags": [int(f) for f in c.fall_flags],
                "unlabelled_images": [str(p.relative_to(root)) for p in c.unlabelled_images],
            }
            for c in clips
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def summarise(clips: Sequence[Clip]) -> None:
    fall = [c for c in clips if c.kind == "fall"]
    adl = [c for c in clips if c.kind == "adl"]

    print(f"klip         : {len(clips)} (jatuh {len(fall)}, adl {len(adl)})")
    print(f"subjek       : {sorted({c.subject_id for c in clips})}")
    print(f"frame        : {sum(c.n_frames for c in clips)}")
    print(f"durasi jatuh : {sum(c.duration_s for c in fall)/60:.1f} menit")
    print(f"durasi ADL   : {sum(c.duration_s for c in adl)/60:.1f} menit")

    onsets = [c.fall_onset() / c.n_frames for c in fall if c.fall_onset() is not None]
    if onsets:
        print(f"onset jatuh (fraksi klip)         : median {statistics.median(onsets):.2f}")
    ratio = [sum(c.fall_flags) / c.n_frames for c in fall]
    if ratio:
        print(f"proporsi frame fall di klip jatuh : median {statistics.median(ratio):.2f}")
    print(f"PNG tanpa label (dilewati)        : {sum(len(c.unlabelled_images) for c in clips)}")

    # Denominator false alarm per jam. Sengaja dicetak agar sulit dilupakan.
    adl_minutes = sum(c.duration_s for c in adl) / 60
    if adl_minutes > 0:
        print(
            f"\nPERINGATAN: hanya {adl_minutes:.1f} menit ADL. Satu false alarm "
            f"berekstrapolasi menjadi ~{60/adl_minutes:.0f}/jam, jadi metrik false "
            "alarm per jam akan sangat berderau. Laporkan juga per-klip di Task 7b."
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="direktori CAUCAFall yang memuat Subject.*")
    parser.add_argument("--csv", type=Path, default=Path("eval/fall_events.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/interim/caucafall/manifest.json")
    )
    parser.add_argument(
        "--allow-problems",
        action="store_true",
        help="tetap tulis keluaran meskipun invarian audit dilanggar",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.expanduser()
    clips = discover_clips(root)

    problems = verify_bundle(clips)
    if problems:
        print("INVARIAN AUDIT DILANGGAR:", file=sys.stderr)
        for p in problems[:20]:
            print(f"  {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... dan {len(problems) - 20} lainnya", file=sys.stderr)
        if not args.allow_problems:
            print(
                "\nKeluaran tidak ditulis. Jalankan scripts/audit_caucafall.py untuk "
                "rincian, atau pakai --allow-problems bila memang disengaja.",
                file=sys.stderr,
            )
            return 1

    write_csv([clip_to_row(c, root) for c in clips], args.csv)
    write_manifest(clips, args.manifest, root)

    summarise(clips)
    print(f"\nditulis: {args.csv}")
    print(f"ditulis: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
