#!/bin/bash
set -euo pipefail
# Script untuk menjalankan deteksi jatuh dengan webcam

echo "═══════════════════════════════════════════════════"
echo "🎥 SISTEM DETEKSI JATUH - WEBCAM MODE"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Menyalakan sistem..."
echo ""

cd -- "$(dirname -- "$0")"
.venv/bin/pfv fall --config config/fall_robust.yaml --source 0

echo ""
echo "Sistem dimatikan. Terima kasih!"
