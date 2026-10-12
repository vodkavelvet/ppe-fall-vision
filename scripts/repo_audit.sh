#!/bin/sh
# Audit apa yang akan dilacak git sebelum repo dibagikan ke tim.
# Read-only: tidak men-stage, meng-commit, atau mengubah apa pun.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/repo_audit.txt
mkdir -p reports

{
  echo "=== branch dan jumlah commit ==="
  git branch --show-current
  git rev-list --all --count 2>/dev/null || echo 0

  echo
  echo "=== file yang AKAN dilacak (tidak ter-ignore) ==="
  git ls-files --others --exclude-standard | sort
  echo "total: $(git ls-files --others --exclude-standard | wc -l | tr -d ' ') file"

  echo
  echo "=== ukuran total file yang akan dilacak ==="
  git ls-files --others --exclude-standard -z | xargs -0 du -ck 2>/dev/null | tail -1

  echo
  echo "=== 8 file terbesar yang akan dilacak (KB) ==="
  git ls-files --others --exclude-standard -z | xargs -0 du -k 2>/dev/null | sort -rn | head -8

  echo
  echo "=== file besar yang ter-ignore (tidak akan terunggah) ==="
  for p in "Dataset CAUCAFall" data/interim models .venv reports; do
    [ -e "$p" ] && printf "%-22s %s\n" "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
  done

  echo
  echo "=== pola rahasia di file yang akan dilacak ==="
  git ls-files --others --exclude-standard -z | xargs -0 grep -nEi "(api[_-]?key|secret|token|password)[\"' ]*[:=][\"' ]*[A-Za-z0-9_\-]{12,}" 2>/dev/null || echo "tidak ada"

  echo
  echo "=== .env dan kaggle.json ter-ignore? ==="
  git check-ignore -v .env kaggle.json 2>&1
} >"$OUT" 2>&1

cat "$OUT"
