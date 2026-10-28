#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec .venv/bin/pfv fall --config config/fall_robust.yaml --clip "Subject.9/Fall sitting"
