"""Fitur deteksi jatuh dari keypoint. Fungsi murni, tanpa model dan tanpa I/O.

Modul ini dirancang dari hasil pengukuran, bukan dari intuisi. Tiga keputusan di
bawah ini semuanya berasal dari analisis 100 klip CAUCAFall yang terdokumentasi
di ``data/DATASETS.md``.

**Tidak ada satu fitur yang cukup.** Diukur per subjek, kemampuan tiap fitur
memisahkan frame tergeletak dari aktivitas harian milik subjek yang sama:

=====================  ==================  ================
fitur                  jatuh menyamping    jatuh ke depan
=====================  ==================  ================
``keypoint_spread_y``  10/10 subjek        6/10 subjek
``hip_ankle_dy``       9-10/10 subjek      7/10 subjek
``trunk_angle``        9-10/10 subjek      3/10 subjek
``trunk_length``       4/10 subjek         7/10 subjek
=====================  ==================  ================

``trunk_angle`` adalah fitur paling umum di literatur deteksi jatuh berbasis
skeleton, dan ia **gagal untuk 7 dari 10 subjek** pada jatuh searah sumbu kamera:
orang yang jatuh ke arah atau menjauhi kamera tetap tampak tegak di bidang
gambar. Sebaliknya ``trunk_length`` justru buruk untuk jatuh menyamping. Keduanya
saling melengkapi, bukan saling menggantikan, dan karena itu modul ini
mengembalikan SEMUA fitur alih-alih memilih satu. Penggabungannya dilakukan
:mod:`pfv.fall.state_machine` (Task 6) dan ambangnya dicari Task 7b.

**Normalisasi memakai DIAGONAL kotak, bukan tingginya.** Tinggi kotak ikut runtuh
saat orang tergeletak, sehingga memakainya sebagai pembagi akan menyembunyikan
justru efek yang dicari. Diagonal jauh lebih stabil terhadap perubahan postur.

**Keypoint tidak valid menghasilkan ``None``, bukan nol.** Ini aturan yang
diwarisi dari :class:`~pfv.perception.base.PersonPose` dan tidak boleh dilanggar
di sini. Mengganti pinggul yang hilang dengan ``(0, 0)`` membuat sudut trunk
dihitung terhadap sudut kiri-atas frame: angkanya tampak masuk akal, tidak ada
error yang muncul, dan false positive-nya nyaris mustahil dilacak.

Satu beda yang disengaja antara pusat trunk dan anggota badan. Pusat bahu dan
pusat pinggul menuntut KEDUA sisi valid, karena satu sisi saja akan memiringkan
vektor trunk dan merusak sudutnya. Lutut dan pergelangan kaki cukup satu sisi,
karena yang dipakai hanya ketinggiannya dan sisi yang tertutup adalah hal biasa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pfv.perception.base import Keypoint, PersonPose

Point = Tuple[float, float]

#: Di bawah ini kotak dianggap terlalu kecil untuk menghasilkan fitur bermakna.
MIN_BBOX_PX = 2.0

#: Jumlah minimum keypoint valid sebelum sebaran vertikal dihitung. Dengan dua
#: atau tiga titik, "sebaran" hanyalah derau.
MIN_KEYPOINTS_FOR_SPREAD = 4


# --------------------------------------------------------------------- bantuan


def bbox_diagonal(bbox: Sequence[float]) -> Optional[float]:
    """Panjang diagonal kotak, atau ``None`` bila kotaknya degenerat.

    Dipakai sebagai pembagi semua fitur jarak. Lihat catatan modul: memakai tinggi
    kotak akan menyembunyikan pemendekan trunk, karena tingginya ikut runtuh.
    """
    w = float(bbox[2]) - float(bbox[0])
    h = float(bbox[3]) - float(bbox[1])
    if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
        return None
    return math.hypot(w, h)


def mean_valid_y(pose: PersonPose, *indices: int) -> Optional[float]:
    """Rata-rata ketinggian keypoint yang valid. ``None`` bila tak ada yang valid.

    Dipakai untuk lutut dan pergelangan kaki: satu sisi sudah cukup karena hanya
    ketinggiannya yang dipakai. Jangan memakai ini untuk pusat trunk.
    """
    ys = [pose.keypoints[i, 1] for i in indices if pose.valid[i]]
    return float(np.mean(ys)) if ys else None


def trunk_vector(pose: PersonPose) -> Optional[np.ndarray]:
    """Vektor dari pusat bahu ke pusat pinggul, dalam piksel.

    ``None`` bila salah satu pusat tidak tersedia. Keduanya menuntut kedua sisi
    valid; lihat catatan modul.
    """
    shoulder = pose.shoulder_centre()
    hip = pose.hip_centre()
    if shoulder is None or hip is None:
        return None
    return np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]], dtype=np.float64)


# ------------------------------------------------------------- fitur per frame


def trunk_angle_deg(pose: PersonPose) -> Optional[float]:
    """Sudut trunk terhadap vertikal, 0 sampai 90 derajat.

    Nol berarti tegak, 90 berarti mendatar. Nilainya tak bertanda: condong ke kiri
    dan ke kanan menghasilkan angka sama, karena yang relevan adalah seberapa
    mendatar tubuhnya.

    **Peringatan.** Fitur ini gagal untuk jatuh searah sumbu kamera. Terukur:
    sudut naik ke 61-79 derajat untuk jatuh ke samping, belakang, dan dari duduk,
    tetapi tetap sekitar 10 derajat untuk jatuh ke depan, dan hanya 3 dari 10
    subjek menunjukkan arah perubahan yang benar. Jangan dipakai sendirian.
    """
    vec = trunk_vector(pose)
    if vec is None:
        return None
    horizontal = abs(float(vec[0]))
    vertical = abs(float(vec[1]))
    if horizontal < 1e-9 and vertical < 1e-9:
        # Bahu dan pinggul berimpit: sudutnya tidak terdefinisi, dan mengembalikan
        # 0 akan terbaca sebagai "tegak sempurna".
        return None
    return math.degrees(math.atan2(horizontal, vertical))


def trunk_length_norm(pose: PersonPose) -> Optional[float]:
    """Panjang trunk dibagi diagonal kotak.

    Inilah fitur untuk jatuh searah sumbu kamera. Tubuh yang jatuh ke arah atau
    menjauhi kamera MEMENDEK di bidang gambar alih-alih melebar, sehingga sudut
    trunk tidak bergerak tetapi panjangnya menyusut.

    Terukur: 7 dari 10 subjek terpisah dari ADL-nya sendiri pada jatuh ke depan,
    tetapi hanya 4 dari 10 pada jatuh menyamping. Jadi ini fitur khusus, bukan
    fitur umum.
    """
    vec = trunk_vector(pose)
    diag = bbox_diagonal(pose.bbox)
    if vec is None or diag is None:
        return None
    return float(np.hypot(vec[0], vec[1])) / diag


def hip_knee_dy_norm(pose: PersonPose) -> Optional[float]:
    """Selisih ketinggian pinggul dan lutut, dibagi diagonal kotak.

    Membedakan tergeletak dari membungkuk: saat tergeletak pinggul dan lutut
    sama-sama di lantai, sementara saat membungkuk mengambil objek pinggul tetap
    tinggi. Tak bertanda, karena yang relevan besarnya.
    """
    hip = pose.hip_centre()
    knee_y = mean_valid_y(pose, Keypoint.LEFT_KNEE, Keypoint.RIGHT_KNEE)
    diag = bbox_diagonal(pose.bbox)
    if hip is None or knee_y is None or diag is None:
        return None
    return abs(knee_y - hip[1]) / diag


def hip_ankle_dy_norm(pose: PersonPose) -> Optional[float]:
    """Selisih ketinggian pinggul dan pergelangan kaki, dibagi diagonal kotak.

    Fitur dengan cakupan terbaik secara keseluruhan: 9-10 dari 10 subjek untuk
    jatuh menyamping dan 7 dari 10 untuk jatuh ke depan. Alasannya geometris --
    pada posisi tergeletak apa pun, pinggul dan kaki berada pada ketinggian yang
    sama, sedangkan pada semua postur tegak tidak.
    """
    hip = pose.hip_centre()
    ankle_y = mean_valid_y(pose, Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE)
    diag = bbox_diagonal(pose.bbox)
    if hip is None or ankle_y is None or diag is None:
        return None
    return abs(ankle_y - hip[1]) / diag


def head_hip_dy_norm(pose: PersonPose) -> Optional[float]:
    """Selisih ketinggian kepala dan pinggul, dibagi diagonal kotak.

    Berguna untuk jatuh menyamping dan dari duduk (8-9 dari 10 subjek), tetapi
    hampir tidak informatif untuk jatuh ke depan (4 dari 10) -- orang yang
    membungkuk juga menurunkan kepalanya ke dekat pinggul.
    """
    hip = pose.hip_centre()
    diag = bbox_diagonal(pose.bbox)
    if hip is None or diag is None or not pose.valid[Keypoint.NOSE]:
        return None
    return abs(float(pose.keypoints[Keypoint.NOSE, 1]) - hip[1]) / diag


def keypoint_spread_y_norm(pose: PersonPose) -> Optional[float]:
    """Rentang vertikal seluruh keypoint valid, dibagi diagonal kotak.

    Fitur terkuat untuk jatuh menyamping dan ke belakang: 10 dari 10 subjek. Untuk
    jatuh ke depan hanya 6 dari 10, karena tubuh yang memendek ke arah kamera
    tetap menyisakan rentang vertikal.

    Dihitung dari keypoint, bukan dari kotak, sehingga tidak mewarisi kesalahan
    detektor saat kotaknya melebar karena lengan terentang.
    """
    diag = bbox_diagonal(pose.bbox)
    if diag is None:
        return None
    ys = pose.keypoints[pose.valid, 1]
    if ys.size < MIN_KEYPOINTS_FOR_SPREAD:
        return None
    return float(ys.max() - ys.min()) / diag


# ------------------------------------------------------------- fitur temporal


def _fit_slope(times: Sequence[float], values: Sequence[float]) -> Optional[float]:
    """Kemiringan regresi linear kuadrat terkecil, satuan nilai per detik.

    Dipakai alih-alih ``(akhir - awal) / dt`` karena jauh lebih tahan terhadap
    derau satu frame, dan menangani timestamp yang jaraknya tidak seragam.
    """
    if len(times) < 2:
        return None
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    span = float(t[-1] - t[0])
    if span <= 0:
        return None
    t = t - t[0]
    denom = float(((t - t.mean()) ** 2).sum())
    if denom <= 0:
        return None
    return float(((t - t.mean()) * (v - v.mean())).sum() / denom)


@dataclass(frozen=True)
class StaticFeatures:
    """Fitur dari satu frame. Setiap medan ``None`` bila datanya tidak memadai."""

    bbox_aspect_ratio: Optional[float] = None
    trunk_angle_deg: Optional[float] = None
    trunk_length_norm: Optional[float] = None
    hip_knee_dy_norm: Optional[float] = None
    hip_ankle_dy_norm: Optional[float] = None
    head_hip_dy_norm: Optional[float] = None
    keypoint_spread_y_norm: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return dict(self.__dict__)

    @property
    def n_available(self) -> int:
        """Berapa fitur yang berhasil dihitung. Nol berarti frame tidak terpakai."""
        return sum(1 for v in self.__dict__.values() if v is not None)


@dataclass(frozen=True)
class TemporalFeatures:
    """Fitur dari riwayat satu orang.

    ``contiguous`` adalah syarat kelayakan, bukan hiasan. Bila riwayatnya berlubang
    karena orangnya sempat tidak terdeteksi, kecepatan yang melintasi lubang itu
    akan tampak seperti gerakan mendadak -- persis pola yang dicari detektor jatuh.
    Pada kasus itu nilai kecepatan sengaja dikosongkan.
    """

    hip_velocity_norm: Optional[float] = None
    """Kecepatan vertikal pusat pinggul, dalam diagonal-kotak per detik.
    POSITIF berarti bergerak TURUN, karena sumbu y gambar tumbuh ke bawah."""

    trunk_angle_rate: Optional[float] = None
    """Laju perubahan sudut trunk, derajat per detik. Positif berarti makin mendatar."""

    motion_energy: Optional[float] = None
    """Rata-rata perpindahan keypoint antar sampel, dinormalisasi dan per detik."""

    window_s: float = 0.0
    n_samples: int = 0
    contiguous: bool = True

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "hip_velocity_norm": self.hip_velocity_norm,
            "trunk_angle_rate": self.trunk_angle_rate,
            "motion_energy": self.motion_energy,
        }


@dataclass(frozen=True)
class FallFeatures:
    """Gabungan fitur per frame dan temporal untuk satu orang pada satu waktu."""

    static: StaticFeatures
    temporal: TemporalFeatures

    def as_dict(self) -> Dict[str, Optional[float]]:
        merged = self.static.as_dict()
        merged.update(self.temporal.as_dict())
        return merged


def compute_static(pose: PersonPose) -> StaticFeatures:
    """Hitung semua fitur per frame. Tidak pernah melempar; medan tak terhitung
    menjadi ``None``."""
    return StaticFeatures(
        bbox_aspect_ratio=pose.bbox_aspect_ratio,
        trunk_angle_deg=trunk_angle_deg(pose),
        trunk_length_norm=trunk_length_norm(pose),
        hip_knee_dy_norm=hip_knee_dy_norm(pose),
        hip_ankle_dy_norm=hip_ankle_dy_norm(pose),
        head_hip_dy_norm=head_hip_dy_norm(pose),
        keypoint_spread_y_norm=keypoint_spread_y_norm(pose),
    )


def _is_contiguous(samples: Sequence[object], max_gap_s: float) -> bool:
    """``True`` bila tidak ada jeda antar sampel yang melebihi ``max_gap_s``."""
    for prev, cur in zip(samples, list(samples)[1:]):
        if cur.timestamp_s - prev.timestamp_s > max_gap_s:
            return False
    return True


def compute_temporal(
    state: "object", *, max_gap_s: float = 0.2, window_s: Optional[float] = None
) -> TemporalFeatures:
    """Hitung fitur temporal dari :class:`~pfv.tracking.TrackState`.

    Diterima sebagai ``object`` dan diakses lewat duck typing supaya modul ini
    tetap bebas dari ketergantungan ke lapisan tracking: fitur harus bisa dihitung
    dari riwayat apa pun, termasuk keypoint hasil ekstraksi yang dibaca dari berkas.

    ``window_s`` membatasi perhitungan ke sampel TERBARU saja. Ini penting karena
    riwayat track menyimpan 3 detik, sedangkan jatuh berlangsung 0,5-1 detik:
    regresi atas seluruh 3 detik akan mengencerkan lonjakan kecepatan menjadi
    hampir tak terlihat. ``None`` memakai semua sampel.
    """
    samples = list(getattr(state, "samples", []) or [])
    if window_s is not None and samples:
        cutoff = samples[-1].timestamp_s - window_s - 1e-6
        samples = [s for s in samples if s.timestamp_s >= cutoff]
    n = len(samples)
    if n == 0:
        return TemporalFeatures(n_samples=0)

    times = [s.timestamp_s for s in samples]
    span = float(times[-1] - times[0]) if n > 1 else 0.0
    contiguous = _is_contiguous(samples, max_gap_s)

    if n < 2 or not contiguous:
        # Riwayat berlubang: laju apa pun yang dihitung di sini akan menyesatkan.
        return TemporalFeatures(
            window_s=span, n_samples=n, contiguous=contiguous
        )

    # Normalisasi memakai diagonal kotak TERBARU, sehingga kecepatan tidak
    # bergantung pada jarak orang ke kamera.
    diag = bbox_diagonal(samples[-1].pose.bbox)

    hip_t: List[float] = []
    hip_y: List[float] = []
    angle_t: List[float] = []
    angle_v: List[float] = []
    for sample in samples:
        hip = sample.pose.hip_centre()
        if hip is not None:
            hip_t.append(sample.timestamp_s)
            hip_y.append(hip[1])
        angle = trunk_angle_deg(sample.pose)
        if angle is not None:
            angle_t.append(sample.timestamp_s)
            angle_v.append(angle)

    hip_slope = _fit_slope(hip_t, hip_y)
    hip_velocity = None
    if hip_slope is not None and diag is not None:
        hip_velocity = hip_slope / diag

    return TemporalFeatures(
        hip_velocity_norm=hip_velocity,
        trunk_angle_rate=_fit_slope(angle_t, angle_v),
        motion_energy=_motion_energy(samples, diag),
        window_s=span,
        n_samples=n,
        contiguous=True,
    )


def _motion_energy(samples: Sequence[object], diag: Optional[float]) -> Optional[float]:
    """Perpindahan rata-rata keypoint yang valid di KEDUA sampel berurutan.

    Keypoint yang berubah status valid tidak dihitung: perpindahannya bukan
    gerakan, melainkan efek deteksi yang muncul atau hilang.
    """
    if diag is None or len(samples) < 2:
        return None

    total = 0.0
    weight = 0.0
    for prev, cur in zip(samples, list(samples)[1:]):
        dt = cur.timestamp_s - prev.timestamp_s
        if dt <= 0:
            continue
        both = prev.pose.valid & cur.pose.valid
        if not bool(both.any()):
            continue
        delta = cur.pose.keypoints[both, :2] - prev.pose.keypoints[both, :2]
        step = float(np.linalg.norm(delta, axis=1).mean())
        total += step / dt
        weight += 1.0

    if weight == 0.0:
        return None
    return (total / weight) / diag
