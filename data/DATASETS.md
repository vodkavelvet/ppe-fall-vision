# Dataset: sumber, versi, lisensi, atribusi

Semua dataset di bawah sudah diverifikasi dari halaman resminya. Isi file ini
dilengkapi lebih lanjut pada Task 2 (fall) dan Task 13a (PPE).

> **Peringatan lisensi.** Dua dataset yang jadi tulang punggung proyek ini
> berlisensi **CC BY-NC-SA 4.0**, yaitu **non-komersial** dan **share-alike**.
> Aman untuk riset, tugas, dan prototipe internal. Untuk produk komersial,
> keduanya harus diganti, dan penggantian itu prasyarat — bukan pekerjaan
> belakangan. Share-alike juga berarti dataset turunan, dan kemungkinan besar
> bobot hasil training, terikat lisensi yang sama. Untuk kepastian hukum,
> konsultasikan ke pihak yang berwenang.

---

## Dilatih

### SH17 — PPE (helmet / head / vest)

| | |
|---|---|
| Penulis | Ahmad Mughees dkk. |
| Publikasi | *Journal of Safety Science and Resilience*, DOI `10.1016/j.jnlssr.2024.09.002` (arXiv `2407.04590`) |
| Repo | `github.com/ahmadmughees/SH17dataset` — **hanya kode & dokumentasi, ~5 MB. BUKAN data.** |
| **Lokasi data** | **Kaggle: `mugheesahmad/sh17-dataset-for-ppe-detection`** |
| Ukuran | 8.099 gambar, 75.994 instance, 17 kelas |
| Resolusi | min 1920x1002, maks 8192x5462. Rata-rata 9,38 instance/gambar |
| Objek kecil | 39.764 anotasi < 1% area gambar; 59.025 < 5% area |
| Sumber gambar | Pexels |
| **Lisensi** | **CC BY-NC-SA 4.0** (terverifikasi dari halaman Kaggle) |

17 kelas: `Person`, `Head`, `Face`, `Glasses`, `Face-mask-medical`, `Face-guard`,
`Ear`, `Earmuffs`, `Hands`, `Gloves`, `Foot`, `Shoes`, `Safety-vest`, `Tools`,
`Helmet`, `Medical-suit`, `Safety-suit`.

Catatan penting:

- **Tidak diunduh ke mesin lokal.** Dirujuk lewat `dataset_sources` di
  `kernel-metadata.json`, lalu di-*mount* Kaggle ke `/kaggle/input` sebagai
  direktori read-only. Kode membacanya sebagai file biasa — bukan streaming,
  bukan panggilan API.
- `Head` kemungkinan besar berarti **semua kepala termasuk yang berhelm**, karena
  ini dataset bagian tubuh. Kelas negatif karena itu **diturunkan**: `Head` tanpa
  `Helmet` yang overlap menjadi `head`. **Wajib diverifikasi visual di Task 13a.**
- `Safety-suit` harus ikut dipetakan ke `vest`, kalau tidak pekerja bercoverall
  hi-vis penuh badan jadi false positive.
- Penomoran 1-17 di README adalah daftar bernomor untuk manusia. **ID kelas
  sebenarnya dibaca dari `data.yaml`** di dataset Kaggle.

### CAUCAFall — deteksi jatuh

| | |
|---|---|
| Penulis | Jose Camilo Eraso, Elena Muñoz, Mariela Muñoz, Jesus Pinto |
| Institusi | Universidad del Cauca; Universidad de la Amazonia |
| Sumber | Mendeley Data, DOI `10.17632/7w7fccy7ky.4`, Version 4, 12 Juli 2022 |
| Unduhan | Bundel 7,75 GB, **tidak bisa diunduh sebagian**. Di disk jadi 7,9 GB |
| Kamera | IR HIKVISION, dipasang di **sudut atas dinding**, terhubung DVR, perekaman terpicu gerakan |

Aktivitas: 5 jatuh (depan, belakang, kiri, kanan, dari posisi duduk) dan 5 ADL
(berjalan, melompat, mengambil objek, duduk, **berlutut**).

Kondisi tidak terkontrol: oklusi, cahaya alami/artifisial/**malam tanpa cahaya**,
variasi pakaian, latar bergerak, tekstur lantai berbeda, sudut jatuh berbeda,
jarak kamera berbeda, subjek beda usia/berat/tinggi/kaki dominan.

**Catatan penggunaan.** Label CAUCAFall dirancang penulisnya untuk melatih
detektor YOLO dengan `fall`/`nofall` sebagai kelas objek. **Proyek ini tidak
melakukan itu** — kondisi jatuh adalah properti temporal, bukan penampakan satu
frame. Label mereka dipakai **hanya sebagai ground truth temporal** untuk
mengevaluasi detektor berbasis skeleton.

#### Hasil audit bundel yang benar-benar diunduh

Dijalankan dengan `scripts/audit_caucafall.py` atas seluruh 100 folder. Laporan
lengkap di `reports/caucafall_audit.json`. Beberapa angka **berbeda dari
deskripsi Mendeley**, jadi yang berlaku adalah hasil audit ini.

| | Deskripsi Mendeley | **Kenyataan di disk** |
|---|---|---|
| Resolusi | 1080x960 | **720x480** (video dan PNG, seragam di 100 folder) |
| Frame rate | 23 fps | **20 fps** (seragam di 100 folder) |

Deskripsi Mendeley menyebut 23 fps dan 1080x960 sebagai *kemampuan kamera*, bukan
properti file yang dikirim. Jadi bukan salah, tapi asumsi awal proyek ini salah.

Terverifikasi benar:

- 10 subjek x 10 aktivitas = 100 folder, nama aktivitas seragam di semua subjek
- Tepat **50 klip jatuh dan 50 klip ADL**
- 20.001 frame PNG, 19.999 file label, 19.877 frame video
- **Tepat satu kotak orang per label** di seluruh 19.999 label
- `classes.txt` ada di 100 folder. Tiga varian isi: 50x `nofall\r\nfall\r\n`
  (folder jatuh), 48x `nofall\r\n` dan 2x `nofall` (folder ADL). Konsisten
  dengan datanya, karena folder ADL memang tidak punya label kelas 1
- **Semantik label bersih**: setiap klip jatuh punya frame kelas 1, dan **tidak
  satu pun** klip ADL punya frame kelas 1
- Panjang klip 87–293 frame, median 203

#### Anomali yang harus ditangani importer

**1. Video kekurangan frame terakhir (97 dari 100 folder).** Jumlah label =
jumlah frame video + 1. Diverifikasi dengan mendekode, bukan dari header.
Alignment-nya sudah ditentukan lewat perbandingan piksel pada frame bergerak di
tiga folder berbeda: **`avi[i]` sama dengan `png[i]`**, jadi yang hilang adalah
frame terakhir, bukan yang pertama (MAD offset 0 ≈ 2,27 versus offset 1 ≈ 3,35).

**2. Dua video terpotong lebih parah.** `Subject.2/Fall backwards` kekurangan 2
frame, dan `Subject.6/Fall backwards` kekurangan **23 frame**. Untuk kedua folder
ini alignment tidak bisa diasumsikan.

**3. Enam folder punya nama file yang tidak berpasangan.** Semuanya artefak
penamaan, bukan data hilang:

| Folder | Anomali |
|---|---|
| `Subject.2/Fall backwards` | PNG bernama `cas200091 - copia.png`, labelnya `cas200091.txt` |
| `Subject.8/Hop` | PNG `sals800096.png`, labelnya `sals800096a.txt` |
| `Subject.8/Pick up object` | PNG `res800090.png`, labelnya `res800090a.txt` |
| `Subject.9/Walk` | PNG `cams900140.png`, labelnya `cams900140w.txt` |
| `Subject.5/Kneel` | PNG `ars500236.png` berlebih, tanpa label |
| `Subject.6/Walk` | PNG `cams600260.png` berlebih, tanpa label |

Importer harus memasangkan PNG dan label dengan **stem yang dinormalisasi**, bukan
dengan pencocokan tepat:

```python
def norm(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"\s*-\s*cop(?:y|ia)(?:\s*\(\d+\))?$", "", s)  # ' - copia'
    s = re.sub(r"[a-z]+$", "", s)                              # huruf ekor: '...096a'
    return s
```

Sudah diuji atas seluruh bundel: **19.999 dari 19.999 label berhasil dipasangkan
ke PNG, nol kegagalan.** Yang tersisa hanya 2 PNG tanpa label (0,01 persen dari
PNG) di `Subject.5/Kneel` dan `Subject.6/Walk`, dan keduanya cukup dilewati.

**4. Prefiks nama file berbeda per aktivitas**, tetapi seragam di dalam folder:
`cas` (Fall backwards), `cfs` (Fall forward), `cis` (Fall left), `cds` (Fall
right), `css` (Fall sitting), `sals` (Hop), `ars` (Kneel), `res` (Pick up
object), `ses` (Sit down), `cams` (Walk). Urutan frame harus diturunkan dari
pengurutan di dalam folder, bukan dari pola nama global.

#### Keputusan: PNG adalah sumber kebenaran, bukan `.avi`

Tiga alasan, dan ini membatalkan rencana awal memakai `.avi` langsung:

1. Tiga folder punya `.avi` yang terpotong (kekurangan 1, 2, dan 23 frame), jadi
   pemetaan label ke indeks frame video tidak seragam.
2. `.avi` hanya **1,7 persen** dari ukuran PNG — 0,126 GB versus 7,62 GB, atau
   6,6 KB versus 400 KB per frame. Kompresi sebesar itu akan menurunkan kualitas
   estimasi pose, dan pose adalah satu-satunya masukan deteksi jatuh.
3. Label berpasangan dengan PNG lewat nama file, bukan dengan indeks video.

Konsekuensinya `--prune-png` **tidak boleh** dijalankan sebelum PNG
ditranskode ke video berkualitas tinggi dan hasilnya diverifikasi.

#### Batasan yang harus diakui

- **Lingkungan rumah, bukan industri.** Tidak ada PPE, mesin, atau pekerja. Jadi
  dataset ini memvalidasi *algoritma* deteksi jatuh, bukan kesesuaiannya dengan
  domain pemasangan.
- **Total hanya ~16,7 menit video**: 8,1 menit klip jatuh dan **8,6 menit ADL**.
  Angka ADL itu adalah denominator untuk false alarm per jam. Satu false alarm
  saja sudah berekstrapolasi menjadi sekitar 7 per jam, jadi metrik itu akan
  punya ketidakpastian sangat lebar dan tidak boleh dibaca sebagai angka presisi.
- **Tepat satu orang per frame.** Dataset ini **tidak bisa** memvalidasi
  ketahanan terhadap pertukaran ID antar orang, padahal itu risiko utama Task 4
  dan Task 9. Perlu sumber lain, misalnya Multiple Cameras Fall.
- **Label mencakup seluruh periode berbaring, bukan hanya detik jatuhnya.**
  Median 69 persen frame dalam satu klip jatuh berlabel kelas 1. Jadi ground
  truth-nya adalah "orang dalam keadaan tergeletak", dan definisi latensi
  deteksi harus mengacu pada awal periode itu.
- Hanya 10 subjek, sehingga error bar metrik model temporal Task 17 lebar. Jangan
  percaya selisih akurasi 1–2 persen.
- Jatuh disimulasikan aktor; pingsan dan merosot lambat punya profil berbeda.
- Satu sudut kamera per adegan.

#### Baseline fitur tunggal

Dihitung dari kotak ground truth, sebagai patokan sebelum Task 5 dan 6 ditulis.
Ambang pada rasio tinggi/lebar kotak, dinilai per frame:

| Ambang | Recall frame fall | FP di frame ADL | FP di frame pra-jatuh |
|---|---|---|---|
| 0,8 | 67,4% | 0,7% | 4,0% |
| **0,9** | **71,3%** | **1,7%** | **6,1%** |
| 1,0 | 73,3% | 2,9% | 8,7% |
| 1,2 | 76,6% | 8,6% | 13,6% |

Sekaligus terlihat kenapa satu fitur tidak cukup. Rasio kotak untuk ADL yang
menjadi pengganggu berada di antara berdiri dan berbaring: berlutut median 1,59,
mengambil objek 1,85, duduk 1,78, sementara frame jatuh punya persentil ke-90
sebesar 1,80. Tumpang tindihnya nyata, jadi sudut trunk, kecepatan pinggul,
keypoint lutut, dan `persistence` di Task 5–6 memang diperlukan.

Angka di atas memakai kotak ground truth, bukan kotak hasil deteksi. Kotak dari
YOLO-pose akan berbeda, jadi ini patokan kualitas fitur, bukan prediksi performa
sistem.

Satu catatan geometri: median rasio tinggi/lebar orang berjalan adalah 2,46,
lebih rendah dari 3,5–4,5 yang biasa untuk manusia berdiri dengan piksel persegi.
Penjelasan yang paling mungkin adalah kotak orang berjalan memang lebih lebar
(tangan dan langkah) ditambah pemendekan perspektif karena kamera menunduk dari
sudut atas dinding — bukan pixel aspect ratio non-square. Belum dipastikan, dan
`pfv doctor` di Task 7a yang akan mengukurnya. Yang penting untuk sekarang:
pemisahan berdiri versus berbaring tetap 4,2 kali, jadi fiturnya tetap berguna.

---

#### Hasil pose dan tracking atas CAUCAFall

Diukur dengan YOLO11n-pose di MPS (51 fps pada sumber 20 fps) atas seluruh 100
klip. Keypoint tersimpan di `data/interim/caucafall/pose/` (gitignored), sehingga
Task 5-7 tidak perlu inference ulang. Skrip: `scripts/extract_pose_caucafall.py`.

**Kualitas pose.** Orang terdeteksi 93-100 persen frame, torso lengkap (kedua bahu
dan kedua pinggul valid) 92-100 persen, lutut 91-100 persen. `trunk_angle` praktis
selalu tersedia.

**Fitur gagal untuk jatuh searah sumbu kamera** (`scripts/analyse_fall_direction.py`,
median gabungan antar klip, belum per subjek):

| Fitur | Jatuh menyamping | Jatuh ke depan | Respons tersisa |
|---|---|---|---|
| Rasio aspek kotak | -73% | -21% | 29% |
| Sudut trunk | +387% | -9% | 2% |
| Panjang trunk (ternormalisasi diagonal kotak) | -9% | -15% | 163% |

Sudut trunk naik ke 61-79 derajat untuk jatuh menyamping, belakang, dan dari
duduk, tetapi tetap sekitar 10 derajat untuk jatuh ke depan. Rasio aspek untuk
jatuh ke depan (1,81) tumpang tindih penuh dengan aktivitas harian (1,73-2,05).
Panjang trunk memisahkan (0,22 versus 0,26-0,31) tetapi dengan margin sedang.

Catatan metodologi yang harus diperbaiki sebelum diklaim di paper: angka di atas
median yang digabung, jadi perlu analisis per subjek dengan sebaran. Uji
"di dalam rentang ADL atau tidak" di skrip itu juga terlalu naif untuk sudut
trunk, karena berjalan dan melompat punya sudut lebih rendah lagi (6,2 dan 6,9).

**Tracking (ByteTrack, supervision 0.30.5).** Diukur dengan
`scripts/measure_id_switches.py`; jumlah ID yang sehat per klip adalah 1.

| `minimum_matching_threshold` | Klip dengan >1 ID | Perpindahan ID di sekitar onset |
|---|---|---|
| 0,80 (bawaan) | 28/100 | 9 |
| 0,90 | 13/100 | 3 |
| **0,95 (dipilih)** | **7/100** | **0** |
| 0,99 | 6/100 | 0 |

Klip jatuh paling terdampak (20 dari 50 pada ambang bawaan, versus 8 dari 50 untuk
ADL), karena orang yang jatuh mengubah bentuk kotaknya drastis dalam beberapa
frame. Menaikkan `lost_track_buffer` hampir tidak membantu (28 menjadi 26), jadi
penyebabnya kotak yang berubah terlalu jauh, bukan orang yang menghilang.

Di ByteTrack nilai ambang yang LEBIH TINGGI berarti LEBIH PERMISIF. Percobaan
pertama dengan 0,5 memperburuk hasil menjadi 85/100 karena arahnya terbalik.

**Risiko yang belum terselesaikan.** CAUCAFall hanya berisi satu orang per frame,
sehingga pertukaran identitas antar orang tidak bisa diuji. Adegan sintetis dua
orang (`scripts/measure_crossing.py`) menunjukkan 0 persen pertukaran pada
berpapasan lurus dan sekitar 3 persen pada skenario saling menutupi atau
berhenti, dengan angka identik di ambang 0,80 sampai 0,99. Artinya tidak ada
bukti 0,95 menambah risiko, tetapi ini bukan bukti aman: adegannya ideal dan
tidak membedakan ambang. Validasi dengan Multiple Cameras Fall sebelum Task 9.

## Validasi saja — tidak dilatih

### Hard Hat Workers

Dihosting Roboflow: `public.roboflow.com/object-detection/hard-hat-workers`.
7.035 gambar. **Pin versi v5 `raw_HeadHelmetClasses`** (split 70/20/10, kelas
`person` sudah dibuang, gambar mentah tanpa augmentasi).

Semantik `head` di sini **sudah benar** = tanpa helm. Halaman resminya: anotasi
mencakup contoh `person` dan `head` "for when an individual may be present
without a hard hat". Karena itu dataset ini dipakai untuk memverifikasi apakah
derivasi `head` dari SH17 sudah benar, dan sebagai **test eksternal** yang
memprediksi performa lapangan.

- Hindari versi `augmented3x` — Ultralytics sudah melakukan augmentasi sendiri.
- **Waspada:** versi v14 `raw_HeadClassOnly` me-remap `helmet` menjadi `head`.
  Nama kelas sama, artinya terbalik. Nomor versi wajib dipin.
- Atribusi penulis asal belum terverifikasi; halaman yang dibaca tidak
  menyebutkannya. Perlu dicek sebelum publikasi.

### UR Fall Detection

Bogdan Kwolek, Michal Kepski. *Computer Methods and Programs in Biomedicine*
117(3), Desember 2014, hlm. 489–501. `fenix.ur.edu.pl/~mkepski/ds/uf.html`.

30 sekuens jatuh + 40 ADL, PNG image sequence (perlu convert ke mp4), plus data
akselerometer per frame. Setiap stream disimpan sebagai zip terpisah, jadi
**bisa diunduh sebagian**.

**Lisensi: CC BY-NC-SA 4.0**, non-komersial, untuk penggunaan akademik.
Halaman resminya meminta kontak langsung untuk penggunaan komersial.

### Multiple Cameras Fall Dataset

E. Auvinet, C. Rougier, J. Meunier, A. St-Arnaud, J. Rousseau,
"Multiple cameras fall dataset", Technical report 1350, DIRO,
Université de Montréal, Juli 2010. `iro.umontreal.ca/~labimage/Dataset/`.
**Sitasi wajib.**

24 skenario, 8 kamera IP. 22 skenario berisi jatuh dan confounding event;
**2 skenario hanya confounding tanpa jatuh** — set negatif murni untuk mengukur
false alarm.

---

## Ditolak, dengan alasan

Dicatat supaya keputusan ini tidak diulang.

| Dataset | Alasan |
|---|---|
| **CHV** — Wang dkk., *Sensors* 2021, 21, 3478, DOI `10.3390/s21103478`, repo `ZijianWang-ZW/PPE_detection`, 1.330 gambar, Google Drive / Baidu (password `f003`) | Tidak punya kelas kepala tanpa helm. Disimpan sebagai cadangan kalau `vest` dari SH17 kurang realistis untuk lokasi target |
| **UP-Fall** — Martínez-Villaseñor dkk., *Sensors* 19(9), 1988, 2019, DOI `10.3390/s19091988`, Universidad Panamericana | Gambar tersebar di ratusan zip terpisah, tidak ada bundel tunggal. Kekuatannya di IMU 5 titik, bukan vision. Trial 2 & 3 subjek 8 aktivitas 11 hilang |
| **NTU RGB+D 120** — ROSE Lab, Nanyang Technological University | Punya `A43 falling down` plus negatif keras `A42 staggering`, `A80 squat down`, `A8 sit down`. **Tapi skeleton-nya 25 joint Kinect V2, tidak kompatibel dengan 17 keypoint COCO** tanpa pemetaan yang lossy. Jalur benarnya mengekstrak keypoint dari 114.480 video RGB dengan model pose sendiri — terlalu mahal untuk fase ini |
| **Le2i / FDD** | Halaman IMVIA mengembalikan HTTP 404. Anggap tidak tersedia |

---

## Belum terverifikasi

Diperiksa saat task yang relevan dikerjakan.

- Struktur folder internal dataset SH17 di Kaggle (apakah ada `images/`,
  `labels/`, `data.yaml`) — Task 13a
- Jumlah instance per kelas SH17 (hanya tersedia sebagai SVG di repo) — Task 13a
- Apakah SH17 punya split bawaan — Task 13a
- Dataset "Construction Site Safety" di Roboflow Universe, yang kemungkinan
  punya kelas `NO-Hardhat` / `NO-Safety Vest`. Roboflow Universe menolak akses
  otomatis dengan HTTP 403, jadi harus dicek manual lewat browser
- Kuota GPU Kaggle per minggu dan durasi maksimum sesi. Docs Kaggle di-render
  JavaScript sehingga tidak terbaca; cek dashboard akun. Desain
  checkpoint-resume di Task 14 tetap dipertahankan apa pun angkanya
