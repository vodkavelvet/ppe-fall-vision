"""Antarmuka command line.

Perintah ditambahkan seiring task:

* ``pfv play``     -- Task 1, putar sumber video dengan overlay FPS
* ``pfv clip``     -- Task 2, putar klip CAUCAFall dengan overlay ground truth
* ``pfv pose``     -- Task 3, skeleton 17 keypoint COCO
* ``pfv features`` -- Task 5
* ``pfv doctor``   -- Task 7a
* ``pfv run``      -- Task 11
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import typer

from pfv import __version__
from pfv.config import load_config
from pfv.io import Frame, ImageSequenceSource, VideoSource, VideoSourceError

app = typer.Typer(
    add_completion=False,
    help="Kepatuhan PPE dan deteksi jatuh dari video.",
    no_args_is_help=True,
)

_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = ESC


class FpsMeter:
    """FPS terukur dengan rata-rata bergerak.

    Rata-rata bergerak dipakai supaya angka di overlay tidak berkedip, tetapi
    tetap responsif saat throughput benar-benar berubah.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self, now: float) -> Optional[float]:
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return None
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return None
        return (len(self._timestamps) - 1) / span


def _draw_overlay(frame: Frame, fps: Optional[float], source_fps: Optional[float]) -> None:
    """Gambar overlay diagnostik langsung di atas gambar frame."""
    fps_text = f"{fps:5.1f} fps" if fps is not None else "  ... fps"
    lines = [f"frame {frame.index:>6}  {fps_text}"]

    w, h = frame.size
    src_w, src_h = frame.source_size
    geometry = f"{w}x{h}"
    if (w, h) != (src_w, src_h):
        geometry += f"  (sumber {src_w}x{src_h}, skala {frame.transform.scale:.3f})"
    if source_fps:
        geometry += f"  sumber {source_fps:.1f} fps"
    lines.append(geometry)

    for i, text in enumerate(lines):
        origin = (10, 26 + i * 24)
        # Garis luar gelap dulu supaya teks tetap terbaca di latar terang.
        cv2.putText(frame.image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame.image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Tampilkan log level DEBUG."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def version() -> None:
    """Tampilkan versi."""
    typer.echo(f"ppe-fall-vision {__version__}")


@app.command()
def play(
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help='Path file, index webcam (mis. "0"), atau URL rtsp://. Menimpa config.',
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path file YAML config.", exists=True, dir_okay=False
    ),
    frame_skip: Optional[int] = typer.Option(
        None, "--frame-skip", min=0, help="Lewati N frame di antara frame yang diproses."
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", min=1, help="Berhenti setelah N frame. Berguna untuk verifikasi."
    ),
    window: bool = typer.Option(
        True, "--window/--no-window", help="Tampilkan jendela. --no-window untuk mode headless."
    ),
) -> None:
    """Putar sumber video dengan overlay FPS.

    Ini demo Task 1: membuktikan jalur masuk video bekerja untuk file, webcam,
    dan RTSP di balik satu antarmuka, dengan rasio aspek terjaga.
    """
    cfg = load_config(config, **{"video.source": source, "video.frame_skip": frame_skip})

    try:
        video = VideoSource(
            cfg.video.source,
            frame_skip=cfg.video.frame_skip,
            resize=cfg.video.resize,
            loop=cfg.video.loop,
            fall_mode=cfg.fall.enabled,
            min_fall_fps=cfg.fall.min_fps,
        )
    except VideoSourceError as exc:
        typer.secho(f"Gagal: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(video.describe())

    meter = FpsMeter()
    interrupted = False

    try:
        for frame in video:
            # Wall clock, bukan timestamp video: yang ingin diketahui di overlay
            # adalah throughput pemrosesan, bukan fps nominal rekaman.
            fps = meter.tick(time.perf_counter())
            _draw_overlay(frame, fps, video.declared_fps)

            if window:
                cv2.imshow("pfv play", frame.image)
                if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    interrupted = True
                    break

            if max_frames is not None and video.frames_yielded >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        video.release()
        if window:
            cv2.destroyAllWindows()

    measured = video.effective_fps
    typer.echo(
        f"{video.frames_yielded} frame dalam {video.elapsed_s:.2f}s"
        + (f" | throughput {measured:.1f} fps" if measured else "")
        + (" | dihentikan pengguna" if interrupted else "")
    )


@app.command()
def clip(
    relpath: str = typer.Argument(
        ..., help='Klip CAUCAFall, mis. "Subject.1/Fall backwards".'
    ),
    dataset_root: Path = typer.Option(
        Path("Dataset CAUCAFall/CAUCAFall"),
        "--dataset-root",
        envvar="PFV_DATASET_ROOT",
        help="Direktori yang memuat Subject.*.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, help="Path file YAML config."
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", min=1),
    window: bool = typer.Option(True, "--window/--no-window"),
) -> None:
    """Putar satu klip CAUCAFall dari PNG lossless, dengan overlay ground truth.

    Demo Task 2. Frame dibaca langsung dari PNG, bukan dari ``.avi`` bawaan
    dataset -- tiga di antaranya terpotong dan ukurannya hanya 1,7 persen dari
    PNG, sehingga kompresinya akan membatasi kualitas estimasi pose nanti.

    Overlay menandai periode yang oleh ground truth dilabeli tergeletak. Perlu
    diingat label CAUCAFall menandai SELURUH periode tergeletak, bukan hanya
    detik jatuhnya.
    """
    from pfv.data.caucafall import NATIVE_FPS, CaucaFallError, load_clip

    try:
        loaded = load_clip(Path(dataset_root) / relpath)
    except CaucaFallError as exc:
        typer.secho(f"Gagal: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    cfg = load_config(config)
    onset = loaded.fall_onset()
    typer.echo(
        f"{loaded.clip_id} | {loaded.activity} | {loaded.kind} | "
        f"{loaded.n_frames} frame @ {NATIVE_FPS:g} fps ({loaded.duration_s:.1f}s) | "
        f"onset: {onset if onset is not None else '-'} | "
        f"interval: {loaded.fall_intervals() or '-'}"
    )

    flags = loaded.fall_flags
    source = ImageSequenceSource(
        loaded.image_paths,
        fps=NATIVE_FPS,
        resize=cfg.video.resize,
        fall_mode=cfg.fall.enabled,
        min_fall_fps=cfg.fall.min_fps,
    )

    meter = FpsMeter()
    interrupted = False
    try:
        for frame in source:
            is_fall = flags[frame.index]
            _draw_overlay(frame, meter.tick(time.perf_counter()), source.declared_fps)
            _draw_ground_truth(frame, is_fall, loaded.activity)

            if window:
                cv2.imshow("pfv clip", frame.image)
                if (cv2.waitKey(int(1000 / NATIVE_FPS)) & 0xFF) in _QUIT_KEYS:
                    interrupted = True
                    break
            if max_frames is not None and source.frames_yielded >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        source.release()
        if window:
            cv2.destroyAllWindows()

    typer.echo(
        f"{source.frames_yielded} frame dalam {source.elapsed_s:.2f}s"
        + (" | dihentikan pengguna" if interrupted else "")
    )


@app.command()
def pose(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help='File, index webcam ("0"), atau URL rtsp://.'
    ),
    clip_ref: Optional[str] = typer.Option(
        None,
        "--clip",
        help='Klip CAUCAFall, mis. "Subject.1/Fall backwards". Menampilkan ground truth juga.',
    ),
    dataset_root: Path = typer.Option(
        Path("Dataset CAUCAFall/CAUCAFall"), "--dataset-root", envvar="PFV_DATASET_ROOT"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="auto | mps | cuda | cpu. Menimpa config."
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", min=1),
    window: bool = typer.Option(True, "--window/--no-window"),
) -> None:
    """Gambar skeleton 17 keypoint COCO di atas sumber video atau klip CAUCAFall.

    Demo Task 3. Kotak berubah warna menjadi jingga ketika torso TIDAK lengkap,
    karena di frame seperti itu ``trunk_angle`` (Task 5) tidak akan tersedia.
    Keypoint yang tidak valid tetap digambar sebagai titik abu-abu, supaya
    terlihat apa yang hilang alih-alih tampak seolah semuanya baik.

    Panah magenta adalah vektor trunk dari pusat bahu ke pusat pinggul -- itulah
    yang nanti menjadi fitur sudut trunk.
    """
    from pfv.draw import draw_pose, draw_text
    from pfv.perception.yolo_pose import YoloPoseEstimator

    cfg = load_config(config, **{"video.source": source, "pose.device": device})

    flags: Optional[list] = None
    title = "pfv pose"
    try:
        if clip_ref:
            from pfv.data.caucafall import NATIVE_FPS, load_clip

            loaded = load_clip(Path(dataset_root) / clip_ref)
            flags = loaded.fall_flags
            typer.echo(
                f"{loaded.clip_id} | {loaded.activity} | {loaded.kind} | "
                f"{loaded.n_frames} frame | onset: {loaded.fall_onset()}"
            )
            frames_source = ImageSequenceSource(
                loaded.image_paths,
                fps=NATIVE_FPS,
                resize=cfg.video.resize,
                fall_mode=cfg.fall.enabled,
                min_fall_fps=cfg.fall.min_fps,
            )
        else:
            frames_source = VideoSource(
                cfg.video.source,
                frame_skip=cfg.video.frame_skip,
                resize=cfg.video.resize,
                loop=cfg.video.loop,
                fall_mode=cfg.fall.enabled,
                min_fall_fps=cfg.fall.min_fps,
            )
            typer.echo(frames_source.describe())
    except (VideoSourceError, RuntimeError) as exc:
        typer.secho(f"Gagal: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    estimator = YoloPoseEstimator(
        model=cfg.pose.model,
        models_dir=cfg.pose.models_dir,
        device=cfg.pose.device,
        conf=cfg.pose.conf,
        keypoint_conf=cfg.pose.keypoint_conf,
        imgsz=cfg.pose.imgsz,
    )
    estimator.load()
    typer.echo(f"pose: {cfg.pose.model} di {estimator.device}")

    meter = FpsMeter()
    n_with_person = n_with_torso = 0
    interrupted = False

    try:
        for frame in frames_source:
            poses = estimator.estimate(frame.image)
            if poses:
                n_with_person += 1
                if any(p.has_full_torso() for p in poses):
                    n_with_torso += 1

            for i, p in enumerate(poses):
                ratio = p.bbox_aspect_ratio
                label = f"#{i} {p.confidence:.2f} kp{p.n_valid()}/17"
                if ratio is not None:
                    label += f" ar{ratio:.2f}"
                draw_pose(frame.image, p, label=label)

            _draw_overlay(frame, meter.tick(time.perf_counter()), frames_source.declared_fps)
            draw_text(
                frame.image,
                f"orang: {len(poses)}",
                (10, 74),
                (0, 255, 0) if poses else (0, 165, 255),
            )
            if flags is not None:
                _draw_ground_truth(frame, flags[frame.index], "ground truth")

            if window:
                cv2.imshow(title, frame.image)
                if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    interrupted = True
                    break
            if max_frames is not None and frames_source.frames_yielded >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        frames_source.release()
        if window:
            cv2.destroyAllWindows()

    total = frames_source.frames_yielded
    pct = lambda n: f"{100.0 * n / total:.0f}%" if total else "-"  # noqa: E731
    typer.echo(
        f"{total} frame dalam {frames_source.elapsed_s:.2f}s | "
        f"ada orang: {pct(n_with_person)} | torso lengkap: {pct(n_with_torso)}"
        + (" | dihentikan pengguna" if interrupted else "")
    )


@app.command()
def track(
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    clip_ref: Optional[str] = typer.Option(
        None, "--clip", help='Klip CAUCAFall, mis. "Subject.1/Fall backwards".'
    ),
    dataset_root: Path = typer.Option(
        Path("Dataset CAUCAFall/CAUCAFall"), "--dataset-root", envvar="PFV_DATASET_ROOT"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, dir_okay=False),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", min=1),
    window: bool = typer.Option(True, "--window/--no-window"),
) -> None:
    """Pose + tracking: setiap orang diberi ID yang menempel antar frame.

    Demo Task 4. Label ``ID:n`` ditampilkan bersama panjang riwayat dalam detik.
    Peringatan ``GAP`` muncul bila riwayat punya lubang deteksi, karena kecepatan
    yang melintasi lubang tidak boleh dipercaya.

    Catatan: CAUCAFall hanya berisi satu orang per frame, jadi klip dari sana
    TIDAK menguji ketahanan terhadap pertukaran ID. Pakai webcam dengan dua orang
    atau Multiple Cameras Fall untuk itu.
    """
    from pfv.draw import draw_pose, draw_text
    from pfv.perception.yolo_pose import YoloPoseEstimator
    from pfv.tracking import PersonTracker

    cfg = load_config(config, **{"video.source": source})
    try:
        if clip_ref:
            from pfv.data.caucafall import NATIVE_FPS, load_clip

            loaded = load_clip(Path(dataset_root) / clip_ref)
            frames_source = ImageSequenceSource(
                loaded.image_paths, fps=NATIVE_FPS, resize=cfg.video.resize,
                fall_mode=cfg.fall.enabled, min_fall_fps=cfg.fall.min_fps,
            )
            fps = NATIVE_FPS
            typer.echo(f"{loaded.clip_id} | {loaded.n_frames} frame")
        else:
            frames_source = VideoSource(
                cfg.video.source, frame_skip=cfg.video.frame_skip,
                resize=cfg.video.resize, loop=cfg.video.loop,
                fall_mode=cfg.fall.enabled, min_fall_fps=cfg.fall.min_fps,
            )
            fps = frames_source.declared_fps or 20.0
            typer.echo(frames_source.describe())
    except (VideoSourceError, RuntimeError) as exc:
        typer.secho(f"Gagal: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    t = cfg.tracking
    estimator = YoloPoseEstimator(
        model=cfg.pose.model, models_dir=cfg.pose.models_dir, device=cfg.pose.device,
        conf=cfg.pose.conf, keypoint_conf=cfg.pose.keypoint_conf, imgsz=cfg.pose.imgsz,
    )
    estimator.load()
    tracker = PersonTracker(
        fps=fps, window_s=t.window_s,
        track_activation_threshold=t.track_activation_threshold,
        lost_track_buffer=t.lost_track_buffer,
        minimum_matching_threshold=t.minimum_matching_threshold,
        minimum_consecutive_frames=t.minimum_consecutive_frames,
        prune_after_s=t.prune_after_s,
    )
    typer.echo(f"pose: {cfg.pose.model} di {estimator.device}")

    meter = FpsMeter()
    seen_ids: set = set()
    interrupted = False
    try:
        for frame in frames_source:
            tracked = tracker.update(estimator.estimate(frame.image), frame)
            for person in tracked:
                seen_ids.add(person.track_id)
                state = tracker.state(person.track_id)
                label = person.label
                if state is not None:
                    label += f" {state.span_s:.1f}s"
                    if not state.is_contiguous(max_gap_s=t.max_gap_s):
                        label += " GAP"
                draw_pose(frame.image, person.pose, label=label)

            _draw_overlay(frame, meter.tick(time.perf_counter()), frames_source.declared_fps)
            draw_text(frame.image, f"track aktif: {len(tracked)}", (10, 74))

            if window:
                cv2.imshow("pfv track", frame.image)
                if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    interrupted = True
                    break
            if max_frames is not None and frames_source.frames_yielded >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        frames_source.release()
        if window:
            cv2.destroyAllWindows()

    typer.echo(
        f"{frames_source.frames_yielded} frame dalam {frames_source.elapsed_s:.2f}s | "
        f"jumlah ID berbeda: {len(seen_ids)}"
        + (" | dihentikan pengguna" if interrupted else "")
    )


@app.command()
def features(
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    clip_ref: Optional[str] = typer.Option(
        None, "--clip", help='Klip CAUCAFall, mis. "Subject.3/Fall forward".'
    ),
    dataset_root: Path = typer.Option(
        Path("Dataset CAUCAFall/CAUCAFall"), "--dataset-root", envvar="PFV_DATASET_ROOT"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, dir_okay=False),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", min=1),
    window: bool = typer.Option(True, "--window/--no-window"),
) -> None:
    """Tampilkan nilai fitur deteksi jatuh secara langsung.

    Demo Task 5. Belum ada keputusan jatuh di sini; itu Task 6. Gunanya melihat
    fitur mana yang bergerak untuk gerakan apa.

    Coba dengan webcam: berdiri, jongkok, berlutut, lalu membungkuk mengambil
    sesuatu. Perhatikan bahwa ``trunk_angle`` nyaris tidak bergerak saat Anda
    membungkuk ke arah kamera, sementara ``hip_ankle`` dan ``trunk_len`` bergerak.
    Itu temuan utama proyek ini, terlihat langsung.
    """
    from pfv.draw import draw_pose, draw_text
    from pfv.fall import compute_static, compute_temporal
    from pfv.perception.yolo_pose import YoloPoseEstimator
    from pfv.tracking import PersonTracker

    cfg = load_config(config, **{"video.source": source})
    flags: Optional[list] = None
    try:
        if clip_ref:
            from pfv.data.caucafall import NATIVE_FPS, load_clip

            loaded = load_clip(Path(dataset_root) / clip_ref)
            flags = loaded.fall_flags
            frames_source = ImageSequenceSource(
                loaded.image_paths, fps=NATIVE_FPS, resize=cfg.video.resize
            )
            fps = NATIVE_FPS
            typer.echo(f"{loaded.clip_id} | onset: {loaded.fall_onset()}")
        else:
            frames_source = VideoSource(
                cfg.video.source, frame_skip=cfg.video.frame_skip,
                resize=cfg.video.resize, loop=cfg.video.loop,
            )
            fps = frames_source.declared_fps or 20.0
            typer.echo(frames_source.describe())
    except (VideoSourceError, RuntimeError) as exc:
        typer.secho(f"Gagal: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    estimator = YoloPoseEstimator(
        model=cfg.pose.model, models_dir=cfg.pose.models_dir, device=cfg.pose.device,
        conf=cfg.pose.conf, keypoint_conf=cfg.pose.keypoint_conf, imgsz=cfg.pose.imgsz,
    )
    estimator.load()
    t = cfg.tracking
    tracker = PersonTracker(
        fps=fps, window_s=t.window_s,
        track_activation_threshold=t.track_activation_threshold,
        lost_track_buffer=t.lost_track_buffer,
        minimum_matching_threshold=t.minimum_matching_threshold,
        minimum_consecutive_frames=t.minimum_consecutive_frames,
        prune_after_s=t.prune_after_s,
    )

    # Nama pendek supaya overlay tetap terbaca.
    shown = (
        ("aspect", "bbox_aspect_ratio", "{:5.2f}"),
        ("angle", "trunk_angle_deg", "{:5.1f}"),
        ("trunk_len", "trunk_length_norm", "{:5.3f}"),
        ("hip_knee", "hip_knee_dy_norm", "{:5.3f}"),
        ("hip_ankle", "hip_ankle_dy_norm", "{:5.3f}"),
        ("head_hip", "head_hip_dy_norm", "{:5.3f}"),
        ("spread_y", "keypoint_spread_y_norm", "{:5.3f}"),
        ("hip_vel", "hip_velocity_norm", "{:+5.2f}"),
        ("angle_rate", "trunk_angle_rate", "{:+5.1f}"),
        ("motion", "motion_energy", "{:5.3f}"),
    )
    n_complete = 0
    interrupted = False

    try:
        for frame in frames_source:
            tracked = tracker.update(estimator.estimate(frame.image), frame)
            for person in tracked:
                state = tracker.state(person.track_id)
                static = compute_static(person.pose)
                temporal = compute_temporal(state, max_gap_s=t.max_gap_s)
                values = static.as_dict()
                values.update(temporal.as_dict())
                if static.n_available == 7:
                    n_complete += 1

                draw_pose(frame.image, person.pose, label=person.label)
                for row, (label, key, fmt) in enumerate(shown):
                    value = values.get(key)
                    text = fmt.format(value) if value is not None else "    -"
                    draw_text(
                        frame.image,
                        f"{label:>10s} {text}",
                        (10, 100 + row * 20),
                        (0, 255, 0) if value is not None else (120, 120, 120),
                        scale=0.5,
                    )
                if not temporal.contiguous:
                    draw_text(frame.image, "RIWAYAT BERLUBANG", (10, 320), (0, 165, 255), 0.5)

            _draw_overlay(frame, None, frames_source.declared_fps)
            if flags is not None:
                _draw_ground_truth(frame, flags[frame.index], "ground truth")

            if window:
                cv2.imshow("pfv features", frame.image)
                if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    interrupted = True
                    break
            if max_frames is not None and frames_source.frames_yielded >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        frames_source.release()
        if window:
            cv2.destroyAllWindows()

    total = frames_source.frames_yielded
    typer.echo(
        f"{total} frame | fitur lengkap (7/7): "
        f"{100.0 * n_complete / total:.0f}%" if total else "0 frame"
    )
    if interrupted:
        typer.echo("dihentikan pengguna")


def _draw_ground_truth(frame: Frame, is_fall: bool, activity: str) -> None:
    """Tandai apakah frame ini berada dalam periode tergeletak menurut label."""
    w, h = frame.size
    colour = (0, 0, 255) if is_fall else (0, 200, 0)
    label = "GROUND TRUTH: FALL" if is_fall else "ground truth: nofall"

    cv2.rectangle(frame.image, (0, 0), (w - 1, h - 1), colour, 4)
    origin = (10, h - 16)
    cv2.putText(frame.image, f"{label}  [{activity}]", origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame.image, f"{label}  [{activity}]", origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


if __name__ == "__main__":  # pragma: no cover
    app()
