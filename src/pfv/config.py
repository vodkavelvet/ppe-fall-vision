"""Konfigurasi aplikasi.

Dua hal dipisahkan dengan sengaja:

* :class:`AppConfig` -- setelan non-rahasia, dimuat dari YAML. Ini yang
  di-commit ke repo dan yang di-review saat threshold berubah.
* :class:`Secrets` -- token dan API key, dari environment atau ``.env``.
  Tidak pernah di-commit.

Pemisahan ini juga yang membuat ``train/common/env.py:get_secret()`` pada Task 12
bisa menukar sumber rahasia (Kaggle Secrets vs ``.env``) tanpa menyentuh
setelan biasa.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class ResizeMode(str, Enum):
    """Mode penskalaan frame.

    Tidak ada mode ``stretch``, dan itu disengaja. Menarik 1080x960 (CAUCAFall,
    rasio ~9:8) menjadi 640x640 membuat orang gepeng dan merusak
    ``bbox_aspect_ratio``, yang merupakan fitur deteksi jatuh. Distorsi aspek
    dilarang di level tipe, bukan cuma lewat konvensi.
    """

    NONE = "none"
    LONGEST_SIDE = "longest_side"
    LETTERBOX = "letterbox"


class ResizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ResizeMode = ResizeMode.NONE
    longest_side: Optional[int] = Field(default=None, gt=0)
    letterbox_size: Optional[Tuple[int, int]] = None
    allow_upscale: bool = False

    @model_validator(mode="after")
    def _check_required_field(self) -> "ResizeConfig":
        if self.mode is ResizeMode.LONGEST_SIDE and self.longest_side is None:
            raise ValueError("resize.mode 'longest_side' membutuhkan resize.longest_side")
        if self.mode is ResizeMode.LETTERBOX:
            if self.letterbox_size is None:
                raise ValueError("resize.mode 'letterbox' membutuhkan resize.letterbox_size")
            w, h = self.letterbox_size
            if w <= 0 or h <= 0:
                raise ValueError("resize.letterbox_size harus positif")
        return self


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "0"
    frame_skip: int = Field(default=0, ge=0)
    resize: ResizeConfig = Field(default_factory=ResizeConfig)
    loop: bool = False


class PostureConfig(BaseModel):
    """Ambang yang mengubah fitur satu frame menjadi suara 'tergeletak'.

    NILAI AWAL, bukan hasil akhir. Diusulkan ``scripts/suggest_thresholds.py``
    dari subjek latih 1-6 saja, dengan batas laju salah-positif 10 persen per
    fitur. Task 7b mencari ambang yang sesungguhnya lewat grid search.
    """

    model_config = ConfigDict(extra="forbid")

    trunk_angle_min_deg: float = Field(default=37.0, ge=0.0, le=90.0)
    """Tergeletak bila sudut trunk >= ini. Hanya andal untuk jatuh menyamping."""

    keypoint_spread_y_max: float = Field(default=0.59, gt=0.0)
    hip_ankle_dy_max: float = Field(default=0.27, gt=0.0)
    hip_knee_dy_max: float = Field(default=0.12, gt=0.0)
    trunk_length_max: float = Field(default=0.24, gt=0.0)
    """Tergeletak bila nilainya <= ini. Dipakai untuk jatuh searah kamera."""

    seated_spread_y_min: float = Field(default=0.59, gt=0.0)
    upright_aspect_ratio_min: Optional[float] = Field(default=None, gt=0.0)
    upright_trunk_angle_max_deg: float = Field(default=30.0, ge=0.0, le=90.0)

    seated_knee_angle_max_deg: Optional[float] = Field(default=None, gt=0.0, lt=180.0)
    """V3: bent knees plus vertical spread invalidate short-trunk evidence.
    None retains V1/V2.

    Direct lying evidence (trunk angle or spread) remains eligible. This
    projected angle is a heuristic, not a universal sitting classifier.
    """

    min_orientation_votes: int = Field(default=1, ge=1, le=3)
    """Dari tiga indikator orientasi (sudut, sebaran, panjang trunk)."""

    min_limb_votes: int = Field(default=1, ge=1, le=2)
    """Dari dua indikator tinggi anggota badan (pinggul-pergelangan kaki, pinggul-lutut).

    Kedua kelompok HARUS sama-sama terpenuhi. Menghitung kelima indikator sebagai
    suara setara gagal pada klip nyata: duduk di lantai memenuhi kedua indikator
    anggota badan (dua suara untuk satu fakta), dan membungkuk memenuhi ketiga
    indikator orientasi tanpa pernah tergeletak."""

    min_available_indicators: int = Field(default=3, ge=1, le=5)
    """Di bawah ini frame berstatus UNKNOWN. Bukti yang terlalu tipis tidak boleh
    menghasilkan tuduhan, juga tidak boleh menghasilkan 'aman'."""


class TimingConfig(BaseModel):
    """Parameter waktu state machine, dalam DETIK (bukan frame)."""

    model_config = ConfigDict(extra="forbid")

    require_current_lying_for_alert: bool = False
    """V3 requires a current LYING frame before emitting a new fall alert."""

    persistence_s: float = Field(default=3.0, gt=0)
    """Orang harus tetap tergeletak selama ini sebelum alert. Peredam false positive
    utama: jongkok dan berlutut tidak bertahan setelah beberapa detik."""

    min_lying_fraction: float = Field(default=0.7, gt=0.0, le=1.0)
    """Porsi frame berstatus tergeletak (di antara yang terbaca) selama persistence."""

    min_known_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    """Porsi frame yang postur-nya terbaca (bukan UNKNOWN) selama persistence. Orang
    yang tertutup hampir sepanjang waktu tidak boleh dinyatakan jatuh dari
    sedikit frame yang kebetulan terbaca tergeletak."""

    grace_s: float = Field(default=0.5, ge=0.0)
    """Berapa lama postur tegak boleh menyela sebelum hitungan persistence dibatalkan.
    Menoleransi derau keypoint satu dua frame."""

    falling_timeout_s: float = Field(default=1.5, gt=0)
    """FALLING tanpa pemicu baru selama ini kembali ke UPRIGHT (alarm palsu)."""

    max_unknown_s: float = Field(default=3.0, gt=0)
    """FALLING yang terus UNKNOWN selama ini dibatalkan. FALLEN tidak pernah
    dibatalkan karena UNKNOWN: orang yang tertutup bukan orang yang bangun."""

    recover_s: float = Field(default=1.5, gt=0)
    """Postur tegak selama ini setelah FALLEN dihitung sebagai bangun kembali."""

    recovered_hold_s: float = Field(default=1.0, ge=0.0)
    cooldown_s: float = Field(default=30.0, ge=0.0)
    """Jeda minimum antar alert untuk satu track. Task 9 menambahkan cooldown
    berbasis posisi, karena track yang pecah menghasilkan ID baru."""

    velocity_window_s: float = Field(default=0.5, gt=0)
    falling_velocity: Optional[float] = Field(default=0.6, gt=0)
    """Kecepatan turun pinggul (diagonal-kotak/detik) yang memicu FALLING. ``None``
    menonaktifkan. Hanya petunjuk, tidak pernah syarat FALLEN: terukur lemah.
    Puncak klip ADL median 0,49 mendekati jatuh p90 0,82."""


class FallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Kejadian jatuh berlangsung 0,5-1 detik. Di bawah ~15 fps hanya ada 3-5
    # frame sepanjang kejadian dan hip_vertical_velocity jadi tidak bermakna.
    min_fps: float = Field(default=15.0, gt=0)

    posture: PostureConfig = Field(default_factory=PostureConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)


class PoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "yolo11n-pose.pt"
    models_dir: Path = Path("models")
    device: str = "auto"
    """``auto`` mencoba mps lalu cpu. Bisa dipaksa ``mps``, ``cuda``, ``cpu``."""

    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    """Ambang confidence deteksi ORANG."""

    keypoint_conf: float = Field(default=0.5, ge=0.0, le=1.0)
    """Ambang confidence per KEYPOINT. Di bawah ini keypoint ditandai invalid.

    Diterapkan di luar inference, jadi mengubahnya tidak memerlukan deteksi ulang.
    """

    imgsz: int = Field(default=640, gt=0)
    rotations: list[int] = Field(default_factory=list)
    merge_iou: float = Field(default=0.5, gt=0.0, le=1.0)
    rotation_conf: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_device(self) -> "PoseConfig":
        if any(r not in (-90, 90, 180) for r in self.rotations):
            raise ValueError("pose.rotations hanya menerima -90, 90, 180")
        allowed = {"auto", "cpu", "mps", "cuda"}
        if self.device.strip().lower() not in allowed:
            raise ValueError(f"pose.device harus salah satu dari {sorted(allowed)}")
        return self


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_s: float = Field(default=3.0, gt=0)
    """Panjang riwayat per orang, dalam DETIK (bukan frame), supaya perilakunya
    sama pada sumber 20 fps maupun 30 fps."""

    track_activation_threshold: float = Field(default=0.25, gt=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=0)
    """Berapa frame sebuah track dipertahankan setelah orangnya hilang. Diverifikasi
    empiris: track_id pulih setelah hilang 2 frame."""

    minimum_matching_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    """Lebih tinggi = lebih permisif. 0.95 dipilih dari pengukuran 100 klip CAUCAFall
    (lihat config/default.yaml), dengan risiko pertukaran ID antar orang yang belum
    teruji karena dataset itu hanya berisi satu orang per frame."""

    minimum_consecutive_frames: int = Field(default=1, ge=1)

    prune_after_s: float = Field(default=5.0, gt=0)
    """Riwayat track yang tak terlihat selama ini dilupakan agar memori tidak tumbuh."""

    max_gap_s: float = Field(default=0.2, gt=0)
    """Jeda terbesar yang masih dianggap riwayat berkesinambungan. Kecepatan yang
    melintasi jeda lebih besar tidak boleh dipercaya (Task 5)."""


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: VideoConfig = Field(default_factory=VideoConfig)
    fall: FallConfig = Field(default_factory=FallConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None, **overrides: Any) -> "AppConfig":
        """Muat konfigurasi dari YAML, lalu terapkan ``overrides`` bersarang.

        ``overrides`` memakai notasi titik agar flag CLI bisa menimpa satu nilai
        tanpa harus menyusun ulang seluruh dict, misalnya
        ``AppConfig.from_yaml(p, **{"video.source": "clip.mp4"})``.
        Nilai ``None`` diabaikan supaya flag CLI yang tidak diisi tidak
        menghapus nilai dari YAML.
        """
        data: Dict[str, Any] = {}
        if path is not None:
            text = Path(path).read_text(encoding="utf-8")
            loaded = yaml.safe_load(text)
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(f"{path}: isi YAML harus berupa mapping di level teratas")
                data = loaded

        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_nested(data, dotted.split("."), value)

        return cls(**data)


def _set_nested(target: Dict[str, Any], keys: List[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[keys[-1]] = value


class Secrets(BaseSettings):
    """Rahasia dari environment atau ``.env``.

    Di Kaggle nilai-nilai ini tidak datang dari ``.env`` melainkan dari Kaggle
    Secrets, yang hanya bisa didefinisikan lewat web UI. Jembatannya dibuat di
    Task 12 (``train/common/env.py:get_secret()``); ``kaggle_secrets`` tidak
    berfungsi di luar environment eksekusi Kaggle.
    """

    model_config = SettingsConfigDict(
        env_prefix="PFV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    roboflow_api_key: Optional[str] = None


def load_config(path: Optional[Path] = None, **overrides: Any) -> AppConfig:
    """Muat :class:`AppConfig`, default dari ``config/default.yaml`` bila ada."""
    if path is None and DEFAULT_CONFIG_PATH.exists():
        path = DEFAULT_CONFIG_PATH
    return AppConfig.from_yaml(path, **overrides)
