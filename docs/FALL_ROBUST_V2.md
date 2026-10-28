# Fall robust V2

Profil `config/fall_robust.yaml` memeriksa gambar asli serta rotasi -90 dan +90
derajat. Koordinat kotak dan keypoint dikembalikan ke gambar asli sebelum NMS
dan tracking. Identitas joint COCO tetap sama. Gambar asli memakai confidence
0,25, sedangkan rotasi memakai 0,5. Kandidat asli di bawah 0,5 harus didukung
kotak dari orientasi lain (IoU minimal 0,5). Pose asli yang didukung tetap
diprioritaskan agar tidak berganti hanya karena confidence rotasi lebih tinggi.
Deteksi lemah kursi pada klip diagnosis tidak mendapat dukungan tersebut.
Ini bukan jaminan bahwa semua objek bukan manusia akan ditolak.

Ambang postur menggunakan hasil tuning model penuh V1; persistence tetap 3 detik.
Profil default V1 dan berkas eval/frozen_models.json, eval/gate.yaml serta hasil
uji V1 tidak berubah. Profil V2 dipakai launcher webcam dan demo. Biaya inference
naik karena ada tiga orientasi per frame; ukur FPS aktual sebelum pemakaian live.

## Menjalankan

```bash
sh run_demo_sitting.sh
sh run_webcam.sh
.venv/bin/pfv fall --config config/fall_robust.yaml --source path/video.mp4
```

## Cache semua deteksi

```bash
.venv/bin/python scripts/extract_pose_caucafall.py \
  "Dataset CAUCAFall/CAUCAFall" --config config/fall_robust.yaml \
  --out data/interim/caucafall/pose_v2
```

Schema 2 menyimpan `frame_offsets`, `all_keypoints`, `all_bboxes`, dan
`all_confidence`, termasuk frame kosong. Replay memakai semua deteksi. Array
ringkasan V1 tetap disediakan untuk statistik lama. Jangan mencampur cache V1
dan V2, atau memakai `--skip-existing` setelah mengganti konfigurasi. Statistik
ringkasan ekstraksi masih dihitung dari kandidat confidence tertinggi, bukan
ukuran kualitas tracking atau ground truth identitas orang.

Perubahan dibuat setelah diagnosis klip subjek 9. Hasil V2 pada subjek 9–10
adalah pemeriksaan regresi/eksplorasi, bukan evaluasi pada data yang belum pernah
dilihat. Validasi generalisasi memerlukan dataset/kamera lain.


## Pemeriksaan pengembangan 27 Oktober 2026

539 tes otomatis lolos. Pada subset 26 klip (5.759 frame), 10/10 klip jatuh
menghasilkan alarm dan 1/16 klip aktivitas biasa menghasilkan alarm palsu.
Alarm palsu `s04_sit_down` juga ada pada V1; V2 belum menyelesaikannya.
`s09_fall_sitting` yang sebelumnya terlewat kini memicu alarm pada detik 7,3.
Hasil ini memakai data yang sudah dilihat saat diagnosis, bukan uji independen.
Konfigurasi dan rincian per klip: `reports/fall_robust_v2_validation.json`.
