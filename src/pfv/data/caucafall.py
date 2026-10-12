"""Pembacaan dataset CAUCAFall.

Modul ini memodelkan bundel apa adanya, termasuk anomalinya. Semua angka dan
perilaku di sini berasal dari audit seluruh 100 folder; lihat
``data/DATASETS.md`` dan ``scripts/audit_caucafall.py``.

Tiga hal yang membentuk desain modul ini:

1. **PNG adalah sumber kebenaran, bukan ``.avi``.** Tiga file ``.avi`` terpotong
   (kekurangan 1, 2, dan 23 frame), dan ukurannya hanya 1,7 persen dari PNG
   sehingga kompresinya berat. Karena pose adalah satu-satunya masukan deteksi
   jatuh, kualitas gambar tidak boleh dikorbankan. ``.avi`` diabaikan sepenuhnya.
2. **Pemasangan PNG ke label lewat stem yang dinormalisasi.** Enam folder punya
   artefak penamaan: ``' - copia'`` dari duplikasi di Windows Explorer berbahasa
   Spanyol, dan sufiks huruf seperti ``'sals800096a'`` dari penamaan manual saat
   anotasi. Dengan normalisasi, 19.999 dari 19.999 label berhasil dipasangkan.
3. **Label adalah ground truth temporal, bukan target training.** Penulis
   merancangnya untuk melatih detektor YOLO ``nofall``/``fall`` per frame. Proyek
   ini tidak melakukan itu, karena jatuh adalah properti temporal. ``class_id``
   dipakai semata untuk mengevaluasi detektor berbasis skeleton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------- konstanta

CLASS_NOFALL = 0
CLASS_FALL = 1

CLASS_NAMES = {CLASS_NOFALL: "nofall", CLASS_FALL: "fall"}

FALL_ACTIVITIES = (
    "Fall backwards",
    "Fall forward",
    "Fall left",
    "Fall right",
    "Fall sitting",
)

#: Berlutut, mengambil objek, dan duduk membuat sudut trunk berubah drastis TANPA
#: orang jatuh. Ini sumber false positive utama untuk deteksi berbasis skeleton,
#: jadi keberadaannya di dataset adalah fitur, bukan pelengkap.
ADL_ACTIVITIES = (
    "Hop",
    "Kneel",
    "Pick up object",
    "Sit down",
    "Walk",
)

ACTIVITIES = FALL_ACTIVITIES + ADL_ACTIVITIES

#: Geometri SEBENARNYA dari bundel terunduh, hasil audit 100 folder. Deskripsi
#: Mendeley menyebut 1080x960 @ 23 fps, tetapi itu spesifikasi kamera; file yang
#: dikirim seragam 720x480 @ 20 fps.
NATIVE_SIZE = (720, 480)
NATIVE_FPS = 20.0

CLASSES_FILENAME = "classes.txt"

_COPY_SUFFIX = re.compile(r"\s*-\s*cop(?:y|ia)(?:\s*\(\d+\))?$", re.IGNORECASE)
_TRAILING_LETTERS = re.compile(r"[a-z]+$", re.IGNORECASE)
_TRAILING_DIGITS = re.compile(r"(\d+)$")


class CaucaFallError(RuntimeError):
    """Struktur dataset tidak seperti yang diharapkan."""


# ------------------------------------------------------------------ nama file


def normalise_stem(stem: str) -> str:
    """Normalkan stem nama file agar PNG dan label bisa dipasangkan.

    Dua artefak penamaan yang ditemukan di audit dihilangkan::

        'cas200091 - copia'  -> 'cas200091'
        'sals800096a'        -> 'sals800096'
        'cams900140w'        -> 'cams900140'

    Huruf hanya dibuang bila berada SETELAH angka, sehingga prefiks seperti
    ``'cas'`` pada nama tanpa angka tidak ikut terhapus.
    """
    s = stem.strip()
    s = _COPY_SUFFIX.sub("", s)
    if _TRAILING_DIGITS.search(_TRAILING_LETTERS.sub("", s)):
        s = _TRAILING_LETTERS.sub("", s)
    return s.lower()


def frame_number(stem: str) -> Optional[int]:
    """Ambil angka di akhir stem, dipakai untuk mengurutkan frame.

    Prefiks berbeda per aktivitas (``cas``, ``cfs``, ``cis``, ``cds``, ``css``,
    ``sals``, ``ars``, ``res``, ``ses``, ``cams``) tetapi seragam di dalam satu
    folder, jadi mengurutkan berdasarkan angka ini benar selama dilakukan
    per folder.
    """
    match = _TRAILING_DIGITS.search(normalise_stem(stem))
    return int(match.group(1)) if match else None


def subject_number(name: str) -> Optional[int]:
    """``'Subject.7'`` -> ``7``."""
    match = re.search(r"(\d+)$", name.strip())
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------- label


@dataclass(frozen=True)
class LabelBox:
    """Satu kotak berlabel dalam format YOLO ternormalisasi."""

    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def is_fall(self) -> bool:
        return self.cls_id == CLASS_FALL

    def to_xyxy(self, width: int, height: int) -> Tuple[float, float, float, float]:
        """Ke koordinat piksel ``(x1, y1, x2, y2)`` pada gambar ``width x height``."""
        half_w = self.w * width / 2.0
        half_h = self.h * height / 2.0
        cx_px = self.cx * width
        cy_px = self.cy * height
        return cx_px - half_w, cy_px - half_h, cx_px + half_w, cy_px + half_h

    def aspect_ratio(self, width: int = NATIVE_SIZE[0], height: int = NATIVE_SIZE[1]) -> Optional[float]:
        """Rasio tinggi/lebar dalam piksel. ``None`` bila lebarnya nol.

        Piksel, bukan nilai ternormalisasi: rasio ternormalisasi akan mengandung
        rasio aspek frame dan membuat angkanya tidak bisa dibandingkan antar
        sumber dengan resolusi berbeda.
        """
        w_px = self.w * width
        h_px = self.h * height
        return h_px / w_px if w_px > 0 else None


def parse_label_file(path: Path) -> List[LabelBox]:
    """Baca satu file label YOLO. Baris rusak dilewati, bukan menggagalkan semua."""
    boxes: List[LabelBox] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append(LabelBox(cls_id=cls_id, cx=cx, cy=cy, w=w, h=h))
    return boxes


# ----------------------------------------------------------------------- klip


@dataclass(frozen=True)
class FrameRecord:
    """Satu frame berlabel di dalam sebuah klip."""

    order: int
    """Posisi berurutan mulai 0. Inilah indeks frame yang dipakai pipeline."""

    frame_number: int
    """Angka asli dari nama file, mis. ``100042``. Hanya untuk penelusuran."""

    image_path: Path
    label_path: Path
    boxes: Tuple[LabelBox, ...]

    @property
    def is_fall(self) -> bool:
        """Frame ini berada dalam periode tergeletak menurut ground truth."""
        return any(b.is_fall for b in self.boxes)


@dataclass
class Clip:
    """Satu folder aktivitas: 10 subjek x 10 aktivitas menghasilkan 100 klip."""

    subject: str
    activity: str
    directory: Path
    frames: List[FrameRecord] = field(default_factory=list)
    unlabelled_images: Tuple[Path, ...] = ()
    """PNG tanpa label. Ada 2 di seluruh bundel dan keduanya dilewati."""

    @property
    def subject_id(self) -> int:
        n = subject_number(self.subject)
        if n is None:
            raise CaucaFallError(f"nama subjek tidak mengandung nomor: {self.subject!r}")
        return n

    @property
    def kind(self) -> str:
        """``'fall'`` atau ``'adl'`` berdasarkan nama aktivitas."""
        if self.activity in FALL_ACTIVITIES:
            return "fall"
        if self.activity in ADL_ACTIVITIES:
            return "adl"
        return "unknown"

    @property
    def clip_id(self) -> str:
        """Pengenal stabil, aman dipakai sebagai nama file."""
        slug = self.activity.lower().replace(" ", "_")
        return f"s{self.subject_id:02d}_{slug}"

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.n_frames / NATIVE_FPS

    @property
    def image_paths(self) -> List[Path]:
        return [f.image_path for f in self.frames]

    @property
    def fall_flags(self) -> List[bool]:
        return [f.is_fall for f in self.frames]

    def fall_intervals(self) -> List[Tuple[int, int]]:
        """Segmen berlabel ``fall`` yang kontinu, sebagai ``(start, end)`` inklusif.

        Audit menunjukkan 50 dari 50 klip jatuh punya **tepat satu** segmen
        kontinu, jadi daftar ini praktis selalu berisi nol atau satu elemen.
        Bentuknya tetap daftar supaya sumber lain yang lebih berantakan, misalnya
        UR Fall atau Multiple Cameras Fall, bisa memakai tipe yang sama.
        """
        intervals: List[Tuple[int, int]] = []
        flags = self.fall_flags
        i = 0
        while i < len(flags):
            if flags[i]:
                j = i
                while j + 1 < len(flags) and flags[j + 1]:
                    j += 1
                intervals.append((i, j))
                i = j + 1
            else:
                i += 1
        return intervals

    def fall_onset(self) -> Optional[int]:
        """Indeks frame saat periode tergeletak dimulai.

        Ini acuan untuk menghitung latensi deteksi. Perlu dicatat bahwa ground
        truth CAUCAFall menandai seluruh periode tergeletak, bukan hanya detik
        jatuhnya, sehingga onset di sini berarti "mulai tampak tergeletak".
        """
        intervals = self.fall_intervals()
        return intervals[0][0] if intervals else None


def load_clip(directory: Path) -> Clip:
    """Baca satu folder aktivitas menjadi :class:`Clip`.

    Pemasangan PNG ke label memakai stem yang dinormalisasi, dan frame diurutkan
    berdasarkan angka di akhir nama file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CaucaFallError(f"bukan direktori: {directory}")

    images: Dict[str, Path] = {}
    for path in directory.glob("*.png"):
        images.setdefault(normalise_stem(path.stem), path)

    labels: Dict[str, Path] = {}
    for path in directory.glob("*.txt"):
        if path.name.lower() == CLASSES_FILENAME:
            continue
        labels.setdefault(normalise_stem(path.stem), path)

    records: List[FrameRecord] = []
    for key, label_path in labels.items():
        image_path = images.get(key)
        if image_path is None:
            raise CaucaFallError(
                f"label tanpa gambar setelah normalisasi: {label_path} (kunci {key!r})"
            )
        number = frame_number(key)
        if number is None:
            raise CaucaFallError(f"nama file tanpa nomor frame: {label_path}")
        records.append(
            FrameRecord(
                order=-1,  # diisi setelah pengurutan
                frame_number=number,
                image_path=image_path,
                label_path=label_path,
                boxes=tuple(parse_label_file(label_path)),
            )
        )

    records.sort(key=lambda r: r.frame_number)
    ordered = [
        FrameRecord(
            order=i,
            frame_number=r.frame_number,
            image_path=r.image_path,
            label_path=r.label_path,
            boxes=r.boxes,
        )
        for i, r in enumerate(records)
    ]

    unlabelled = tuple(
        sorted(path for key, path in images.items() if key not in labels)
    )

    return Clip(
        subject=directory.parent.name,
        activity=directory.name,
        directory=directory,
        frames=ordered,
        unlabelled_images=unlabelled,
    )


def discover_clips(root: Path, *, subjects: Optional[Sequence[int]] = None) -> List[Clip]:
    """Temukan semua klip di bawah direktori yang memuat ``Subject.*``.

    ``subjects`` menyaring berdasarkan nomor subjek. Itu yang dipakai untuk split
    per subjek pada Task 7b dan 17 -- split per KLIP akan membuat orang yang sama
    muncul di train dan test, dan metriknya jadi menyesatkan.
    """
    root = Path(root)
    if not root.is_dir():
        raise CaucaFallError(f"bukan direktori: {root}")

    subject_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("Subject")]
    if not subject_dirs:
        raise CaucaFallError(f"tidak ada direktori Subject.* di {root}")

    wanted = set(subjects) if subjects is not None else None
    clips: List[Clip] = []
    for subject_dir in sorted(subject_dirs, key=lambda d: (subject_number(d.name) or 0)):
        number = subject_number(subject_dir.name)
        if wanted is not None and number not in wanted:
            continue
        for activity_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            clips.append(load_clip(activity_dir))
    return clips


# ------------------------------------------------------- ground truth evaluasi

#: Kolom ``eval/fall_events.csv``. Ini kontrak dengan harness evaluasi Task 7b.
EVENT_CSV_FIELDS = (
    "clip_id",
    "subject_id",
    "activity",
    "kind",
    "n_frames",
    "fps",
    "duration_s",
    "fall_start_frame",
    "fall_end_frame",
    "fall_start_s",
    "fall_end_s",
    "n_fall_frames",
    "directory",
)

#: Invarian yang ditetapkan audit bundel. Dipakai :func:`verify_bundle`.
EXPECTED_N_CLIPS = 100
EXPECTED_N_FALL_CLIPS = 50
EXPECTED_N_ADL_CLIPS = 50
EXPECTED_N_LABELLED_FRAMES = 19999


def clip_to_row(clip: Clip, root: Path) -> Dict[str, object]:
    """Ubah :class:`Clip` menjadi satu baris ground truth.

    Satu baris per klip, bukan per frame. Alasannya: evaluasi Task 7b mengukur
    per KEJADIAN, bukan per frame -- satu kejadian jatuh yang terdeteksi 40 frame
    berturut-turut adalah satu keberhasilan, bukan empat puluh.
    """
    intervals = clip.fall_intervals()
    start: Optional[int] = None
    end: Optional[int] = None
    if intervals:
        # Audit: 50/50 klip jatuh punya tepat satu segmen. Kalau suatu saat ada
        # lebih dari satu, baris ini merangkum dari awal segmen pertama sampai
        # akhir segmen terakhir; flag per frame yang lengkap tetap ada di manifes.
        start, end = intervals[0][0], intervals[-1][1]

    return {
        "clip_id": clip.clip_id,
        "subject_id": clip.subject_id,
        "activity": clip.activity,
        "kind": clip.kind,
        "n_frames": clip.n_frames,
        "fps": NATIVE_FPS,
        "duration_s": round(clip.duration_s, 3),
        "fall_start_frame": "" if start is None else start,
        "fall_end_frame": "" if end is None else end,
        "fall_start_s": "" if start is None else round(start / NATIVE_FPS, 3),
        "fall_end_s": "" if end is None else round(end / NATIVE_FPS, 3),
        "n_fall_frames": sum(clip.fall_flags),
        "directory": str(clip.directory.relative_to(root)),
    }


def verify_bundle(clips: Sequence[Clip]) -> List[str]:
    """Periksa invarian yang ditetapkan audit. Kembalikan daftar pelanggaran.

    Ada supaya kalau bundel diunduh ulang, dipindahkan, atau versinya berganti,
    penyimpangannya terdeteksi saat impor -- bukan muncul sebagai metrik aneh
    beberapa task kemudian.
    """
    problems: List[str] = []

    if len(clips) != EXPECTED_N_CLIPS:
        problems.append(f"jumlah klip {len(clips)}, diharapkan {EXPECTED_N_CLIPS}")

    fall = [c for c in clips if c.kind == "fall"]
    adl = [c for c in clips if c.kind == "adl"]
    if len(fall) != EXPECTED_N_FALL_CLIPS or len(adl) != EXPECTED_N_ADL_CLIPS:
        problems.append(
            f"komposisi jatuh/adl = {len(fall)}/{len(adl)}, "
            f"diharapkan {EXPECTED_N_FALL_CLIPS}/{EXPECTED_N_ADL_CLIPS}"
        )

    unknown = [c.clip_id for c in clips if c.kind == "unknown"]
    if unknown:
        problems.append(f"aktivitas tak dikenal: {unknown[:5]}")

    for c in fall:
        if c.fall_onset() is None:
            problems.append(f"{c.clip_id}: klip jatuh tanpa label fall")
    for c in adl:
        if c.fall_onset() is not None:
            problems.append(f"{c.clip_id}: klip ADL punya label fall")

    total = sum(c.n_frames for c in clips)
    if total != EXPECTED_N_LABELLED_FRAMES:
        problems.append(
            f"total frame berlabel {total}, diharapkan {EXPECTED_N_LABELLED_FRAMES}"
        )

    for c in clips:
        bad = [f.order for f in c.frames if len(f.boxes) != 1]
        if bad:
            problems.append(
                f"{c.clip_id}: {len(bad)} frame dengan jumlah kotak != 1 "
                f"(mis. frame {bad[0]})"
            )
            break

    return problems
