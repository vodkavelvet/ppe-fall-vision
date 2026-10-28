#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec .venv/bin/pfv fall --config config/fall_robust_v3.yaml --clip "Subject.9/Fall sitting"
