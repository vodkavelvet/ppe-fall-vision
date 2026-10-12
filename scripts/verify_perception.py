#!/usr/bin/env python
"""Verifikasi stack perception terpasang benar dan MPS bisa dipakai.

Dijalankan sekali setelah pemasangan. Kalau MPS tidak tersedia, inference tetap
jalan di CPU tapi jauh lebih lambat, dan itu harus terlihat sekarang -- bukan
muncul sebagai "kok lambat" beberapa task kemudian.
"""

from __future__ import annotations

import sys


def main() -> int:
    import platform

    print(f"python {sys.version.split()[0]} {platform.machine()}")

    import numpy
    import torch
    import torchvision

    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"numpy {numpy.__version__}")

    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"MPS built={built} available={available}")

    if available:
        x = torch.randn(512, 512, device="mps")
        y = (x @ x).sum().item()
        print(f"matmul di mps OK (sum={y:.1f})")
    else:
        print("PERINGATAN: MPS tidak tersedia, inference akan jalan di CPU")

    import ultralytics
    import supervision

    print(f"ultralytics {ultralytics.__version__} | supervision {supervision.__version__}")

    from ultralytics import YOLO  # noqa: F401
    from supervision import ByteTrack  # noqa: F401

    print("YOLO & ByteTrack importable: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
