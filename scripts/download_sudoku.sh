#!/bin/bash
# Download PUMA's Sudoku data (Google Drive folder 1VNg6G9faB4FIoFv_9ktspYHURokLKumb from the
# README) into data/sudoku_new and check the array shapes. Files that already check out are skipped.
#   bash scripts/download_sudoku.sh
# Options: DATA_DIR (default data/sudoku_new), PY (default .venv/bin/python; needs gdown and numpy).
# Next step, on a compute node: mkdir -p runs/logs && sbatch scripts/preprocess_sudoku.sbatch
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
DATA_DIR=${DATA_DIR:-data/sudoku_new}
mkdir -p "$DATA_DIR"

check() { # path rows -> success if the file is a [rows, 325] array
    "$PY" -c "import numpy as np, sys; a = np.load(sys.argv[1], mmap_mode='r'); sys.exit(a.shape != (int(sys.argv[2]), 325))" "$1" "$2" 2>/dev/null
}

# name  google-drive-id  rows
while read -r name id rows; do
    out=$DATA_DIR/$name
    if [ -f "$out" ] && check "$out" "$rows"; then
        echo "ok (already present): $out"
        continue
    fi
    "$PY" -m gdown "https://drive.google.com/uc?id=$id" -O "$out.part"
    mv "$out.part" "$out"
    check "$out" "$rows" || { echo "unexpected contents in $out (expected [$rows, 325])" >&2; exit 1; }
    echo "ok (downloaded): $out"
done <<'EOF'
sudoku-test-data.npy 1QbzCnCi06mPn-4CcZYmYt30X7MsG579V 99999
sudoku-train-data.npy 1SqHlSbq2fS0-4eYEPsYSoqt3cTRJpjiC 1804462
EOF
