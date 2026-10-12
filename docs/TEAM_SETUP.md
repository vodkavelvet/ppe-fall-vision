# Panduan tim

Repo ini berisi **kode, konfigurasi, dan ground truth kecil**. Repo ini **tidak
berisi** dataset, bobot model, maupun keypoint hasil ekstraksi. Semua itu
dihasilkan ulang atau diunduh sendiri oleh tiap anggota.

## Kenapa dataset tidak ada di repo

CAUCAFall (7,9 GB) dan SH17 berlisensi **CC BY-NC-SA 4.0**. Selain terlalu besar
untuk GitHub, redistribusinya sebaiknya lewat sumber aslinya. Karena itu setiap
anggota mengunduh sendiri dan wajib mencantumkan atribusi di `data/DATASETS.md`.

## Setup di komputer lokal

Prasyarat: Python 3.10 atau lebih baru. Proyek ini dikembangkan di 3.14.7.

```bash
git clone <url-repo> ppe-fall-vision
cd ppe-fall-vision

# uv: https://docs.astral.sh/uv/  (satu binary, tanpa sudo)
uv venv
uv pip install -e ".[dev,perception]"

sh scripts/check.sh        # hasil di reports/check.txt, harus pytest_exit=0
```

Di Windows jalankan perintah `sh` lewat Git Bash atau WSL. `.gitattributes` sudah
memaksa script `.sh` berakhir LF supaya tidak rusak saat checkout.

## Dataset CAUCAFall

1. Unduh dari Mendeley Data, DOI `10.17632/7w7fccy7ky.4` (Version 4, 7,75 GB).
   Tidak bisa diunduh sebagian.
2. Taruh di mana saja, lalu beri tahu proyek lewat variabel lingkungan:

```bash
export PFV_DATASET_ROOT="/path/ke/CAUCAFall"     # direktori yang memuat Subject.*
```

Kalau tidak diset, default-nya `Dataset CAUCAFall/CAUCAFall` di akar repo.

3. Verifikasi bundelnya sama dengan yang diaudit:

```bash
sh scripts/import.sh       # menolak menulis apa pun bila invarian audit dilanggar
```

`eval/fall_events.csv` sudah ada di repo (ground truth, 100 baris). Importer
membuat ulang dan hasilnya harus identik.

## Keypoint hasil ekstraksi

Ekstraksi pose atas 100 klip menghasilkan `data/interim/caucafall/pose/*.npz`
(4,8 MB, tidak di-commit). Cukup dijalankan sekali per mesin:

```bash
sh scripts/extract_pose.sh
```

Waktunya sekitar 7 menit di Apple Silicon (MPS, 51 fps). Di CPU jauh lebih lambat,
dan di GPU CUDA lebih cepat. Karena hanya 4,8 MB, boleh juga dibagikan lewat
Google Drive, asalkan versi bobot dan `keypoint_conf` yang dipakai sama.

## Google Colab

**Belum diverifikasi.** Kami belum menjalankan proyek ini di Colab. Langkah di
bawah masuk akal tetapi perlu dicoba sekali oleh orang pertama yang memakainya,
lalu bagian ini diperbarui dengan hasilnya.

```python
# Sel 1: ambil kode
!git clone <url-repo> ppe-fall-vision
%cd ppe-fall-vision

# Sel 2: pasang. Versi dipin eksak di pyproject.toml; Colab akan memasang ulang
# torch/numpy jika berbeda dari bawaannya.
!pip install -q uv
!uv pip install --system -e ".[dev,perception]"

# Sel 3: dataset dari Google Drive (unggah bundel CAUCAFall ke Drive sekali)
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["PFV_DATASET_ROOT"] = "/content/drive/MyDrive/CAUCAFall"

# Sel 4: uji
!sh scripts/check.sh
!cat reports/check.txt
```

Hal yang perlu dicek orang pertama: versi Python Colab memenuhi `>=3.10`,
`numpy==2.5.3` dan `opencv-python==5.0.0.93` bisa terpasang di sana, dan
device otomatis memilih `cuda` (`pose.device: auto` mencoba mps, cuda, lalu cpu).
Jendela `cv2.imshow` tidak ada di Colab; selalu pakai `--no-window`.

Colab hanya cocok untuk eksperimen dan ekstraksi. **Jangan menyimpan keluaran
penting hanya di Colab**; runtime-nya dihapus berkala. Hasil yang berharga di-commit
ke repo (kode, ground truth) atau disimpan di Drive.

## Cara kerja bersama

Kerja dibagi per jalur supaya jarang bentrok:

| Jalur | Isi | Folder utama |
|---|---|---|
| A | Deteksi jatuh (Task 5, 6, 7a, 7b) | `src/pfv/fall/`, `eval/` |
| B | PPE dan training Kaggle (Task 12-15) | `train/`, `kaggle/` |
| C | Fusi, event, alert, dashboard (Task 8-11) | `src/pfv/fusion/`, `rules/`, `alert/` |

Aturan praktis:

- **Satu cabang per task**, mis. `task5-fall-features`. Jangan commit langsung ke `main`.
- Pull Request sebelum merge; minimal satu orang lain membaca. Untuk Task 5 sampai
  7b yang menjadi inti paper, baca fungsi fiturnya baris demi baris.
- **`scripts/check.sh` harus `pytest_exit=0` sebelum membuka PR.**
- Empat berkas paling sering bentrok: `config/default.yaml`, `src/pfv/cli.py`,
  `src/pfv/perception/base.py`, `pyproject.toml`. Beri tahu tim sebelum mengubahnya
  dan tambahkan di bagian yang berbeda dari orang lain.
- Temuan dicatat di `data/DATASETS.md`, bukan di catatan pribadi.

## Hal yang sudah pernah memakan waktu

- **Jangan** `from tests.conftest import ...`. Paket `ultralytics` memasang
  direktori `tests/` miliknya di site-packages dan membayangi milik kita. Pakai
  `pfv_test_helpers`.
- Di ByteTrack, `minimum_matching_threshold` yang LEBIH TINGGI berarti LEBIH
  PERMISIF. Terbalik dari intuisi.
- Resolusi dan fps CAUCAFall yang sebenarnya **720x480 @ 20 fps**, bukan 1080x960
  @ 23 seperti di deskripsi Mendeley.
- Jangan mengganti keypoint tidak valid dengan nol; tandai invalid.
- Split data **per subjek**, bukan per klip.

## Lisensi

Belum ada berkas `LICENSE`. Tanpa itu, secara bawaan hak ciptanya milik penulis
dan orang lain tidak punya izin memakai kode ini. Keputusan lisensi tergantung
tujuan akhir (skripsi, paper, atau produk) dan perlu disepakati tim. Pertimbangan:

- Dataset: CC BY-NC-SA 4.0, non-komersial.
- Ultralytics: AGPL-3.0, menular bila kode didistribusikan.

Selama belum diputuskan, **jadikan repo GitHub-nya Private**.
