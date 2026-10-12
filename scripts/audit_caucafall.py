#!/usr/bin/env python
"""Audit bundel CAUCAFall apa adanya, sebelum apa pun diimpor.

Tujuannya memeriksa asumsi, bukan mempercayai deskripsi dataset. Yang diverifikasi:

* 10 subjek x 10 aktivitas, nama konsisten
* setiap folder punya satu ``.avi``, frame ``.png``, dan label ``.txt`` per frame
* jumlah frame ``.avi`` sama dengan jumlah label -- ini yang menentukan apakah
  folder PNG boleh dihapus untuk mengembalikan ruang disk
* ``classes.txt`` konsisten di semua folder
* resolusi dan fps sesuai yang dideskripsikan (1080x960, 23 fps)
* label memuat ground truth temporal lewat ``class_id`` (0=nofall, 1=fall), dan
  distribusinya masuk akal: aktivitas jatuh harus punya frame kelas 1,
  aktivitas ADL seharusnya tidak

Dijalankan sebagai read-only. Tidak menyentuh, memindahkan, atau menghapus apa pun.

Pemakaian::

    .venv/bin/python scripts/audit_caucafall.py "Dataset CAUCAFall/CAUCAFall"
    .venv/bin/python scripts/audit_caucafall.py <root> --json audit.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Nama aktivitas yang diharapkan, dan klasifikasinya.
# ADL di sini bukan pelengkap: berlutut, mengambil objek, dan duduk adalah
# gerakan yang membuat sudut trunk berubah drastis TANPA orang jatuh, yaitu
# sumber false positive nomor satu untuk deteksi berbasis skeleton.
FALL_ACTIVITIES = {
    "Fall backwards",
    "Fall forward",
    "Fall left",
    "Fall right",
    "Fall sitting",
}
ADL_ACTIVITIES = {
    "Hop",
    "Kneel",
    "Pick up object",
    "Sit down",
    "Walk",
}
EXPECTED_ACTIVITIES = FALL_ACTIVITIES | ADL_ACTIVITIES

EXPECTED_SIZE = (1080, 960)
EXPECTED_FPS = 23.0
CLASSES_FILE = "classes.txt"


def png_size(path: Path) -> Optional[Tuple[int, int]]:
    """Baca lebar/tinggi dari header PNG tanpa mendekode gambarnya.

    Mendekode 12 ribu PNG hanya untuk mengetahui ukurannya akan memakan waktu
    lama tanpa alasan; IHDR chunk sudah memuat informasinya di 24 byte pertama.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


@dataclass
class ActivityReport:
    subject: str
    activity: str
    kind: str  # "fall" | "adl" | "unknown"
    relpath: str

    avi_files: List[str] = field(default_factory=list)
    n_png: int = 0
    n_label: int = 0

    avi_frame_count: Optional[int] = None
    avi_fps: Optional[float] = None
    avi_size: Optional[Tuple[int, int]] = None

    png_sizes: Dict[str, int] = field(default_factory=dict)
    label_class_counts: Dict[str, int] = field(default_factory=dict)
    first_fall_frame: Optional[int] = None
    last_fall_frame: Optional[int] = None
    label_prefixes: List[str] = field(default_factory=list)
    has_classes_file: bool = False
    boxes_per_label: Dict[str, int] = field(default_factory=dict)

    unpaired_png: List[str] = field(default_factory=list)
    unpaired_label: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def read_avi_metadata(path: Path) -> Tuple[Optional[int], Optional[float], Optional[Tuple[int, int]]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, None, None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return (
        count if count > 0 else None,
        fps if fps > 0 else None,
        (w, h) if w > 0 and h > 0 else None,
    )


def decode_avi_frame_count(path: Path) -> Optional[int]:
    """Hitung frame dengan benar-benar mendekode.

    ``CAP_PROP_FRAME_COUNT`` sering hanya estimasi dari header dan bisa salah.
    Karena keputusan ``--prune-png`` bergantung pada angka ini, mode teliti
    harus tersedia.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    n = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
    finally:
        cap.release()
    return n


def parse_label_file(path: Path) -> Tuple[List[int], int]:
    """Kembalikan ``(class_id per kotak, jumlah kotak)`` dari satu file label YOLO."""
    classes: List[int] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return classes, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            classes.append(int(float(parts[0])))
        except (ValueError, IndexError):
            continue
    return classes, len(classes)


def audit_activity(
    subject_dir: Path, activity_dir: Path, root: Path, *, strict: bool, sample_png: int
) -> ActivityReport:
    import re

    kind = (
        "fall"
        if activity_dir.name in FALL_ACTIVITIES
        else "adl"
        if activity_dir.name in ADL_ACTIVITIES
        else "unknown"
    )
    rep = ActivityReport(
        subject=subject_dir.name,
        activity=activity_dir.name,
        kind=kind,
        relpath=str(activity_dir.relative_to(root)),
    )

    pngs = sorted(p for p in activity_dir.glob("*.png"))
    avis = sorted(p for p in activity_dir.iterdir() if p.suffix.lower() in {".avi", ".mp4"})
    all_txt = sorted(p for p in activity_dir.glob("*.txt"))
    labels = [p for p in all_txt if p.name != CLASSES_FILE]

    rep.has_classes_file = any(p.name == CLASSES_FILE for p in all_txt)
    rep.avi_files = [p.name for p in avis]
    rep.n_png = len(pngs)
    rep.n_label = len(labels)

    if len(avis) != 1:
        rep.problems.append(f"jumlah file video = {len(avis)}, diharapkan 1")
    if not rep.has_classes_file:
        rep.problems.append("classes.txt tidak ada")

    if avis:
        count, fps, size = read_avi_metadata(avis[0])
        rep.avi_frame_count = decode_avi_frame_count(avis[0]) if strict else count
        rep.avi_fps = fps
        rep.avi_size = size
        if size and tuple(size) != EXPECTED_SIZE:
            rep.problems.append(f"resolusi video {size[0]}x{size[1]}, diharapkan 1080x960")
        if fps and abs(fps - EXPECTED_FPS) > 1.0:
            rep.problems.append(f"fps video {fps:.2f}, diharapkan ~23")

    # Pasangan PNG <-> label berdasarkan stem.
    png_stems = {p.stem for p in pngs}
    label_stems = {p.stem for p in labels}
    rep.unpaired_png = sorted(png_stems - label_stems)[:10]
    rep.unpaired_label = sorted(label_stems - png_stems)[:10]
    if rep.unpaired_png:
        rep.problems.append(f"{len(png_stems - label_stems)} png tanpa label")
    if rep.unpaired_label:
        rep.problems.append(f"{len(label_stems - png_stems)} label tanpa png")

    # Prefiks nama file: dipakai untuk memastikan urutan frame bisa diturunkan
    # dari namanya, karena itu prasyarat memetakan label ke indeks frame video.
    prefixes = Counter()
    for p in labels:
        m = re.match(r"^(.*?)(\d+)$", p.stem)
        prefixes[m.group(1) if m else p.stem] += 1
    rep.label_prefixes = [f"{k}*{v}" for k, v in sorted(prefixes.items())]
    if len(prefixes) > 1:
        rep.problems.append(f"prefiks nama label tidak seragam: {rep.label_prefixes}")

    # Ukuran PNG: hanya sampel, karena seluruhnya ada belasan ribu file.
    step = max(1, len(pngs) // sample_png) if sample_png else 1
    size_counter: Counter = Counter()
    for p in pngs[::step]:
        s = png_size(p)
        size_counter[f"{s[0]}x{s[1]}" if s else "tidak terbaca"] += 1
    rep.png_sizes = dict(size_counter)

    # Ground truth temporal ada di class_id tiap label.
    class_counter: Counter = Counter()
    boxes_counter: Counter = Counter()
    fall_indices: List[int] = []
    for order, p in enumerate(sorted(labels, key=lambda q: q.stem)):
        classes, n_boxes = parse_label_file(p)
        boxes_counter[str(n_boxes)] += 1
        for c in classes:
            class_counter[str(c)] += 1
        if 1 in classes:
            fall_indices.append(order)

    rep.label_class_counts = dict(sorted(class_counter.items()))
    rep.boxes_per_label = dict(sorted(boxes_counter.items()))
    if fall_indices:
        rep.first_fall_frame = fall_indices[0]
        rep.last_fall_frame = fall_indices[-1]

    # Konsistensi semantik: aktivitas jatuh harus punya frame kelas 1, ADL tidak.
    has_fall_label = "1" in rep.label_class_counts
    if kind == "fall" and not has_fall_label:
        rep.problems.append("aktivitas jatuh tetapi tidak ada label kelas 1 (fall)")
    if kind == "adl" and has_fall_label:
        rep.problems.append(
            f"aktivitas ADL tetapi ada {rep.label_class_counts['1']} label kelas 1 (fall)"
        )

    # Inilah keputusan yang menentukan apakah PNG boleh dihapus nanti.
    if rep.avi_frame_count is not None and rep.n_label != rep.avi_frame_count:
        rep.problems.append(
            f"jumlah label {rep.n_label} != frame video {rep.avi_frame_count}"
        )

    return rep


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="direktori CAUCAFall yang memuat Subject.*")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="hitung frame video dengan mendekode, bukan dari header (lebih lambat, lebih akurat)",
    )
    parser.add_argument(
        "--sample-png", type=int, default=6, help="jumlah PNG yang dicek ukurannya per folder (0 = semua)"
    )
    parser.add_argument("--json", type=Path, default=None, help="tulis laporan lengkap ke JSON")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah folder, untuk uji cepat")
    args = parser.parse_args(argv)

    root: Path = args.root.expanduser()
    if not root.is_dir():
        print(f"bukan direktori: {root}", file=sys.stderr)
        return 2

    subjects = sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: (len(d.name), d.name),  # Subject.2 sebelum Subject.10
    )
    if not subjects:
        print(f"tidak ada direktori subjek di {root}", file=sys.stderr)
        return 2

    reports: List[ActivityReport] = []
    for subject in subjects:
        for activity in sorted(d for d in subject.iterdir() if d.is_dir()):
            reports.append(
                audit_activity(subject, activity, root, strict=args.strict, sample_png=args.sample_png)
            )
            if args.limit and len(reports) >= args.limit:
                break
        if args.limit and len(reports) >= args.limit:
            break

    _print_summary(root, subjects, reports)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nlaporan lengkap: {args.json}")

    return 1 if any(r.problems for r in reports) else 0


def _print_summary(root: Path, subjects: List[Path], reports: List[ActivityReport]) -> None:
    print("=" * 78)
    print(f"AUDIT CAUCAFall  {root}")
    print("=" * 78)

    print(f"\nsubjek   : {len(subjects)}")
    print(f"aktivitas: {len(reports)}")

    activity_names = Counter(r.activity for r in reports)
    unexpected = set(activity_names) - EXPECTED_ACTIVITIES
    missing = EXPECTED_ACTIVITIES - set(activity_names)
    print(f"nama aktivitas tak dikenal: {sorted(unexpected) or 'tidak ada'}")
    print(f"nama aktivitas hilang     : {sorted(missing) or 'tidak ada'}")

    n_fall = sum(1 for r in reports if r.kind == "fall")
    n_adl = sum(1 for r in reports if r.kind == "adl")
    print(f"\nklip jatuh: {n_fall}   klip ADL: {n_adl}")

    total_png = sum(r.n_png for r in reports)
    total_label = sum(r.n_label for r in reports)
    total_frames = sum(r.avi_frame_count or 0 for r in reports)
    print(f"total frame PNG   : {total_png}")
    print(f"total file label  : {total_label}")
    print(f"total frame video : {total_frames}")

    fps_values = Counter(f"{r.avi_fps:.2f}" for r in reports if r.avi_fps)
    size_values: Counter = Counter()
    for r in reports:
        if r.avi_size:
            size_values[f"{r.avi_size[0]}x{r.avi_size[1]}"] += 1
    png_size_values: Counter = Counter()
    for r in reports:
        for k, v in r.png_sizes.items():
            png_size_values[k] += v
    print(f"\nfps video      : {dict(fps_values)}")
    print(f"resolusi video : {dict(size_values)}")
    print(f"resolusi PNG   : {dict(png_size_values)}")

    boxes: Counter = Counter()
    for r in reports:
        for k, v in r.boxes_per_label.items():
            boxes[k] += v
    print(f"kotak per label: {dict(sorted(boxes.items()))}")

    cls: Counter = Counter()
    for r in reports:
        for k, v in r.label_class_counts.items():
            cls[k] += v
    print(f"\nlabel per kelas (0=nofall, 1=fall): {dict(sorted(cls.items()))}")

    no_classes = [r.relpath for r in reports if not r.has_classes_file]
    print(f"folder tanpa classes.txt: {len(no_classes)}")
    if no_classes:
        for p in no_classes[:6]:
            print(f"    {p}")
        if len(no_classes) > 6:
            print(f"    ... dan {len(no_classes) - 6} lainnya")

    print("\n" + "-" * 78)
    print("PENTING: jumlah label == jumlah frame video?")
    print("-" * 78)
    match = [r for r in reports if r.avi_frame_count is not None and r.n_label == r.avi_frame_count]
    mismatch = [r for r in reports if r.avi_frame_count is not None and r.n_label != r.avi_frame_count]
    unknown = [r for r in reports if r.avi_frame_count is None]
    print(f"cocok      : {len(match)}")
    print(f"tidak cocok: {len(mismatch)}")
    print(f"tak terbaca: {len(unknown)}")
    for r in mismatch[:10]:
        print(f"    {r.relpath}: label {r.n_label} vs frame {r.avi_frame_count} (png {r.n_png})")
    if len(mismatch) > 10:
        print(f"    ... dan {len(mismatch) - 10} lainnya")

    print("\n" + "-" * 78)
    print("MASALAH")
    print("-" * 78)
    flagged = [r for r in reports if r.problems]
    if not flagged:
        print("tidak ada")
    else:
        grouped: Dict[str, int] = Counter()
        for r in flagged:
            for p in r.problems:
                # Normalisasi angka agar masalah serupa terkelompok.
                import re as _re

                grouped[_re.sub(r"\d+", "N", p)] += 1
        for text, n in grouped.most_common():
            print(f"  [{n:>3}x] {text}")
        print(f"\nfolder bermasalah: {len(flagged)} dari {len(reports)}")
        for r in flagged[:8]:
            print(f"    {r.relpath}: {'; '.join(r.problems)}")
        if len(flagged) > 8:
            print(f"    ... dan {len(flagged) - 8} lainnya")


if __name__ == "__main__":
    raise SystemExit(main())
