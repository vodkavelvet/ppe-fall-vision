# ppe-fall-vision

Kepatuhan PPE (helm, vest) dan deteksi jatuh dari video, memakai object detection
untuk PPE dan pose estimation untuk jatuh, difusikan per orang.

Dikembangkan lokal di macOS Apple Silicon. Training dijalankan di Kaggle GPU
lewat push dari repo; bobot hasilnya ditarik kembali untuk inference lokal.

## Status

**Task 1 selesai** — scaffold, abstraksi sumber video, config, CLI `pfv play`.

**Task 2 selesai** — audit CAUCAFall, pembacaan dataset, sumber sekuens gambar,
dan ground truth evaluasi.

**Task 3 selesai** — pose estimation YOLO11n-pose di MPS (`pfv pose`).

**Task 4 selesai** — tracking ByteTrack dengan riwayat per orang berjendela
waktu (`pfv track`). Ambang pencocokan 0,95 dipilih dari pengukuran atas 100 klip;
lihat `data/DATASETS.md`. Pertukaran ID antar orang belum teruji pada data nyata.

**Task 5 selesai** — fitur deteksi jatuh sebagai fungsi murni (`pfv features`).
Tujuh fitur per frame dan tiga fitur temporal, semuanya mengembalikan `None` bila
data tidak memadai. Daya pisah dan ketersediaan tiap fitur sudah diukur per
subjek; lihat `data/DATASETS.md`.

**Task 6 selesai** — state machine jatuh (`pfv fall`): postur dari dua kelompok
indikator yang harus sama-sama setuju, `persistence`, toleransi, dan cooldown.
Pratinjau pada subjek latih: recall 22/30, alarm palsu 0/30. Ambangnya nilai awal,
belum dituning; lihat `data/DATASETS.md` untuk batas yang dibuat dataset dan
temuan soal indikator anggota badan.

**Task 7a selesai** — `pfv doctor` memeriksa kelayakan sebuah sumber.

**Task 7b: evaluasi V1 selesai — LOLOS.** Subjek uji 9–10: recall mentah
9/10, recall pada klip yang memenuhi syarat durasi 8/9, alarm palsu 0/10 klip
ADL, median latensi 3,1 detik. Hasil ada di `eval/test_results.json` dan catatan
run di `eval/test_runs.jsonl`. Sampelnya 10 jatuh dan 10 ADL, bukan 20 per kelas.
Berkas beku dan gerbang tetap dipertahankan sebagai bukti versi V1.

**Perbaikan persepsi V2:** profil `config/fall_robust.yaml` menambahkan pemeriksaan
rotasi dan dukungan lintas orientasi untuk deteksi lemah, serta cache semua
kandidat sebelum tracking. Launcher webcam/demo memakai profil ini. Ini versi
pengembangan setelah diagnosis data uji, bukan hasil evaluasi V1. Pemakaian dan
batasannya ada di [`docs/FALL_ROBUST_V2.md`](docs/FALL_ROBUST_V2.md).

Belum ada PPE dan belum ada validasi lintas-dataset (UR Fall, Multiple Cameras
Fall); itu Task 8 dan seterusnya.

```bash
# Audit bundel apa adanya (read-only)
.venv/bin/python scripts/audit_caucafall.py "Dataset CAUCAFall/CAUCAFall" \
    --json reports/caucafall_audit.json

# Bentuk ground truth + manifes
sh scripts/import.sh

# Putar satu klip dengan overlay ground truth
.venv/bin/pfv clip "Subject.1/Fall backwards"

sh scripts/check.sh      # pytest
```

Audit menemukan dua fakta yang berbeda dari deskripsi Mendeley: resolusinya
**720x480** bukan 1080x960, dan **20 fps** bukan 23. Ada juga tiga `.avi` yang
terpotong (kekurangan 1, 2, dan 23 frame) dan enam folder dengan nama file tidak
berpasangan. Setelah normalisasi stem, **19.999 dari 19.999 label terpakai, nol
kehilangan**. Rinciannya di [`data/DATASETS.md`](data/DATASETS.md).

**Bekerja dalam tim atau memakai Colab?** Baca [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md)
lebih dulu: dataset tidak ada di repo dan diunduh sendiri, lokasinya diberitahukan
lewat `PFV_DATASET_ROOT`.

## Jalankan

Proyek memakai [uv](https://docs.astral.sh/uv/). Versi Python ada di
`.python-version`, jadi `uv` memilihnya otomatis.

```bash
uv venv                 # membuat .venv dengan Python 3.14
uv pip install -e ".[dev]"

.venv/bin/pfv play --source data/raw/sample.mp4      # file
.venv/bin/pfv play --source 0                        # webcam
.venv/bin/pfv play --source rtsp://kamera/stream1    # RTSP
.venv/bin/pfv play --source clip.mp4 --no-window     # headless

.venv/bin/python -m pytest
```

Tekan `q` atau `Esc` untuk keluar dari jendela.

## Keputusan desain yang perlu diketahui sebelum menyentuh kode

**Penskalaan tidak pernah mendistorsi rasio aspek.** `ResizeMode` hanya punya
`none`, `longest_side`, dan `letterbox`. Mode `stretch` sengaja tidak ada.
Alasannya bukan kosmetik: `bbox_aspect_ratio` adalah fitur deteksi jatuh, dan
menarik frame 720x480 milik CAUCAFall menjadi 640x640 membuat orang berdiri
terlihat gepeng sehingga fiturnya rusak secara diam-diam. Audit dataset
menunjukkan pemisahan berdiri versus berbaring hanya 4,2 kali, jadi distorsi
1,33 kali dari penarikan 3:2 ke 1:1 bukan hal sepele.

**Setiap `Frame` membawa `FrameTransform`.** Deteksi berjalan di gambar yang sudah
diskalakan, tetapi event, crop bukti, dan klip harus merujuk koordinat frame
asli. `transform.to_source_box(...)` yang menjembatani. Menyimpannya bersama
frame mencegah kekeliruan koordinat yang sulit dilacak nanti.

**`frame_skip` berbahaya saat mode fall aktif.** Kejadian jatuh berlangsung
0,5–1 detik; `hip_vertical_velocity` butuh beberapa frame dalam rentang itu.
`VideoSource` memberi peringatan di konstruktor, bukan saat iterasi, supaya
terlihat sebelum video habis diproses. Untuk menghemat komputasi, turunkan
resolusi lewat `video.resize`, bukan `frame_skip`.

**Tiga angka fps, artinya berbeda.** `declared_fps` dari metadata sumber;
`sampled_fps` = `declared_fps / step`, yaitu resolusi temporal setelah
`frame_skip` dan inilah yang menentukan apakah fitur kecepatan bermakna;
`effective_fps` terukur dari wall clock, jujur untuk webcam dan RTSP yang sering
jauh di bawah fps yang diklaim.

**Config dan rahasia dipisah.** `AppConfig` dari YAML, di-commit dan di-review.
`Secrets` dari environment atau `.env`, tidak pernah di-commit. Pemisahan ini yang
nanti membuat `get_secret()` pada Task 12 bisa menukar sumber rahasia antara
Kaggle Secrets dan `.env` tanpa menyentuh setelan biasa.

**PNG dibaca langsung, tanpa transkode ke video.** Ini menyimpang dari rencana
awal Task 2, dan alasannya dari data: `.avi` bawaan CAUCAFall hanya 1,7 persen
dari ukuran PNG (6,6 KB versus 400 KB per frame) dan tiga di antaranya terpotong.
Estimasi pose adalah satu-satunya masukan deteksi jatuh, jadi menurunkan kualitas
gambar di pintu masuk akan membatasi semua yang di atasnya.
`ImageSequenceSource` punya antarmuka yang sama dengan `VideoSource`, ada test
yang mengunci kesepadanan itu, dan UR Fall juga berupa sekuens PNG sehingga modul
yang sama melayani keduanya. Konsekuensinya tidak ada opsi `--prune-png`: PNG
adalah sumber kebenaran, bukan data sementara.

**Ground truth adalah interval, bukan per frame.** `eval/fall_events.csv` berisi
satu baris per klip. Evaluasi Task 7b mengukur per kejadian — satu jatuh yang
terdeteksi selama 40 frame berturut-turut adalah satu keberhasilan, bukan empat
puluh. Label CAUCAFall sendiri menandai **seluruh periode tergeletak**, bukan
hanya detik jatuhnya (median 69 persen frame dalam klip jatuh), jadi latensi
deteksi diukur dari awal periode itu.

## Catatan environment

**Python 3.14.7** (arm64), dipasang lewat `uv` ke direktori user — tanpa `sudo`
dan tanpa mengubah sistem. Mesin ini aslinya hanya punya Python 3.9.6 bawaan
macOS yang sudah melewati akhir masa dukungan.

Kenapa 3.14 dan bukan 3.15: 3.15 masih release candidate, dan `torch` 2.14
maupun `supervision` 0.30.5 belum mendukungnya. 3.14 adalah versi stabil terbaru
yang didukung seluruh stack.

Stack Task 3 sudah diverifikasi bisa dipasang bersama di 3.14.7, dengan
`torch.backends.mps` aktif dan matmul di device `mps` berhasil. Versinya dipin di
grup `perception` pada `pyproject.toml`, dipasang saat Task 3 tiba.

`requires-python` disetel `>=3.10`, bukan `>=3.14`. Itu bukan kompromi: image
Kaggle memakai Python sendiri (saat ini 3.11), dan kode yang sama harus jalan di
sana tanpa cabang khusus.

Kode masih memakai `from __future__ import annotations` dan menghindari sintaks
yang khusus versi baru. Itu dipertahankan dengan sengaja untuk alasan yang sama:
kompatibilitas dengan Python di kernel Kaggle.

### Soal Python 3.9 bawaan macOS

Python 3.9.6 di `/usr/bin/python3` **tidak boleh dihapus**, dan memang tidak bisa:
`/usr/bin` dilindungi System Integrity Protection. Binary itu milik `root:wheel`,
berbagi inode dengan puluhan tool lain, dan sebenarnya disediakan Xcode Command
Line Tools — yang juga menyediakan `git`. Menghapusnya butuh mematikan SIP dan
berisiko merusak toolchain sistem, tanpa manfaat apa pun karena proyek ini tidak
memakainya.

Yang dilakukan sebagai gantinya: `pfv` punya penjaga versi yang gagal saat impor
dengan pesan yang langsung bisa ditindaklanjuti kalau dijalankan dengan
interpreter yang salah. Batasnya ada di `pfv.MIN_PYTHON`, dan ada test yang
memastikan nilainya selalu sama dengan `requires-python` di `pyproject.toml`.

## Lisensi dataset

Dataset latih utama, **SH17** dan **CAUCAFall**, keduanya melibatkan lisensi
**CC BY-NC-SA 4.0** — non-komersial dan share-alike. Aman untuk riset dan
prototipe internal; untuk produk komersial keduanya harus diganti. Rinciannya di
[`data/DATASETS.md`](data/DATASETS.md).
