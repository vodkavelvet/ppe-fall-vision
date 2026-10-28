# Audit fall detection seluruh subjek — 28 Oktober 2026

Audit pengembangan ini memeriksa Subject 1–10: **100 klip, 19.999 frame**, terdiri dari 50 klip jatuh dan 50 klip aktivitas biasa. Semua klip memakai inference baru dan satu konfigurasi V3 yang dibekukan selama pemeriksaan.

Hasil ini **belum menunjukkan semua kasus sudah benar**. Data yang sama telah digunakan untuk pengembangan; angka di bawah bukan hasil test independen baru.

## Hasil utama

| Ukuran | Hasil |
|---|---:|
| Klip jatuh yang menghasilkan alarm | 42/50 (84%) |
| Klip aktivitas biasa dengan alarm palsu | 1/50 (2%) |
| Klip aktivitas biasa yang pernah menampilkan FALLING | 27/50 (54%) |
| Klip jatuh terlewat | 8 |

`FALLING` adalah status dugaan sementara; `FALLEN` adalah konfirmasi yang dapat menghasilkan alarm. Alarm palsu dan status FALLING pada aktivitas normal dilaporkan terpisah. Ketiadaan alarm tidak membuktikan pose selalu terbaca.

## Hasil per subjek

Setiap subjek memiliki 5 klip jatuh dan 5 klip aktivitas biasa.

| Subjek | Jatuh terdeteksi | Alarm palsu pada klip normal | Klip normal dengan FALLING |
|---|---:|---:|---:|
| 1 | 3/5 | 0/5 | 3/5 |
| 2 | 4/5 | 0/5 | 1/5 |
| 3 | 3/5 | 0/5 | 2/5 |
| 4 | 3/5 | 0/5 | 3/5 |
| 5 | 5/5 | 0/5 | 4/5 |
| 6 | 5/5 | 1/5 | 4/5 |
| 7 | 5/5 | 0/5 | 3/5 |
| 8 | 4/5 | 0/5 | 3/5 |
| 9 | 5/5 | 0/5 | 2/5 |
| 10 | 5/5 | 0/5 | 2/5 |

## Kasus yang masih perlu dikerjakan

Alarm palsu: **Subject 6 — Pick up object**, pada detik 6,25.

Klip jatuh yang terlewat:

| Klip | Durasi berlabel tergeletak | Cakupan tracking |
|---|---:|---:|
| `s01_fall_forward` | 3.00 s | 93.7% |
| `s01_fall_left` | 2.15 s | 81.6% |
| `s02_fall_backwards` | 2.90 s | 100.0% |
| `s03_fall_left` | 3.50 s | 70.9% |
| `s03_fall_sitting` | 7.60 s | 40.4% |
| `s04_fall_backwards` | 6.15 s | 100.0% |
| `s04_fall_sitting` | 6.90 s | 100.0% |
| `s08_fall_forward` | 4.15 s | 82.9% |

Tiga dari delapan miss memiliki periode berlabel tergeletak paling lama 3 detik. Lima lainnya memiliki durasi minimal 3,5 detik, sehingga pendeknya klip tidak menjelaskan seluruh miss. Angka cakupan tracking adalah petunjuk diagnosis, bukan bukti penyebab tunggal.

**Subject 8 — Fall left** menghasilkan dua alarm pada detik 5,65 dan 10,1 dengan ID track berbeda. Deduplikasi alarm lintas ID masih perlu diperbaiki. **Subject 8 — Walk** tetap UPRIGHT sepanjang klip yang diperiksa.

## Perbandingan profil pada deteksi pose yang sama

| Profil keputusan | Jatuh terdeteksi | Klip normal dengan alarm | Klip normal dengan FALLING |
|---|---:|---:|---:|
| V2 | 42/50 | 3/50 | 49/50 |
| V3_initial | 42/50 | 1/50 | 47/50 |
| V3_current | 42/50 | 1/50 | 27/50 |

Pada input deteksi yang sama, tidak ada jatuh yang terdeteksi V2 kemudian hilang pada V3 saat audit ini. Perbandingan ini memisahkan efek aturan keputusan dari perubahan model/persepsi.

## Konfigurasi dan verifikasi

- Profil: `config/fall_robust_v3.yaml`; model YOLO11n-pose, orientasi asli dan rotasi ±90°, tracking ByteTrack.
- Konfirmasi memakai persistence 3 detik; pemicu FALLING dari kecepatan saja dinonaktifkan.
- Verifikasi kode terakhir: **574 tes otomatis lolos**. Tes kode dan akurasi dataset adalah dua ukuran berbeda.
- Semua 100 video klip, 10 video gabungan per subjek, dan 100 cache pose diperiksa kelengkapannya.
- Replay cache menghasilkan waktu alarm yang sama dengan pipeline saat inference.
- Semua 44 alarm yang dipancarkan memiliki IoU minimal 0,5 dengan kotak orang pada anotasi. Ini memeriksa lokasi orang, bukan kebenaran aktivitas.

Fingerprint run: `475b73e560878d7fbfe0ba96ca63cb43683924c3a3480e9a5d3913978d710188`.

## Berkas yang disimpan di GitHub

[Hasil lengkap, metadata, provenance, dan perbandingan profil](../eval/development/all_subjects_v3_20261028.json).

Dokumen ini adalah ringkasan manusia; JSON merupakan bukti terstruktur. Dataset, cache pose, bobot model, dan video tetap menjadi artefak lokal yang dapat dihasilkan ulang, bukan bagian commit kode. Hasil V1 dan gerbang evaluasi historis tetap disimpan tanpa perubahan.

## Mengulang pemeriksaan

```bash
.venv/bin/python scripts/extract_pose_caucafall.py \
  "Dataset CAUCAFall/CAUCAFall" \
  --config config/fall_robust_v3.yaml \
  --out data/interim/caucafall/pose_v3_full_recheck

.venv/bin/python scripts/verify_fall_v3.py \
  --pose-dir data/interim/caucafall/pose_v3_full_recheck \
  --out reports/fall_v3_full_recheck.json
```

Perintah kedua membandingkan keputusan V2/V3 dan memeriksa state Subject 8 Walk; berkas output berada di folder reports yang diabaikan Git. Pemeriksaan ini bersifat pengembangan dan dapat diulang, berbeda dari evaluasi V1 yang dikunci.
