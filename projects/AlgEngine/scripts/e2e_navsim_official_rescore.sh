#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/e2e_navsim_official_rescore.sh \
  <submission.pkl> [metric_cache_path] [output_dir]

Environment:
  NAVSIM_DEVKIT_ROOT       NAVSIM v1.1 repository root (required)
  NAVSIM_METRIC_CACHE_PATH Full navtest metric-cache root (unless argument 2 is set)
  NAVSIM_EXP_ROOT          Fallback root containing metric_cache_navtest_v1
  PYTHON_BIN               Python executable (default: python)
  OPENBLAS_CORETYPE        OpenBLAS kernel target (default: Prescott)
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMISSION=$1
PYTHON_BIN=${PYTHON_BIN:-python}
RESCORE_SHARDS=8

if [ ! -f "$SUBMISSION" ]; then
    echo "ERROR: submission not found: $SUBMISSION" >&2
    exit 1
fi
SUBMISSION=$(realpath "$SUBMISSION")

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
    echo "ERROR: NAVSIM_DEVKIT_ROOT is not set." >&2
    echo "Point it to the NAVSIM v1.1 repository root." >&2
    exit 1
fi
NAVSIM_OFFICIAL_SCRIPT=${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_from_submission.py
if [ ! -f "$NAVSIM_OFFICIAL_SCRIPT" ]; then
    echo "ERROR: official NAVSIM scorer not found: $NAVSIM_OFFICIAL_SCRIPT" >&2
    exit 1
fi

METRIC_CACHE=${2:-${NAVSIM_METRIC_CACHE_PATH:-}}
if [ -z "$METRIC_CACHE" ] && [ -n "${NAVSIM_EXP_ROOT:-}" ]; then
    for candidate in \
        "${NAVSIM_EXP_ROOT}/metric_cache_navtest_v1" \
        "${NAVSIM_EXP_ROOT}/metric_cache_navtest" \
        "${NAVSIM_EXP_ROOT}/metric_cache"; do
        if [ -d "$candidate" ]; then
            METRIC_CACHE=$candidate
            break
        fi
    done
fi
if [ -z "$METRIC_CACHE" ] || [ ! -d "$METRIC_CACHE" ]; then
    echo "ERROR: navtest metric cache not found: ${METRIC_CACHE:-<unset>}" >&2
    echo "Set NAVSIM_METRIC_CACHE_PATH or pass it as argument 2." >&2
    exit 1
fi
METRIC_CACHE=$(realpath "$METRIC_CACHE")

SUBMISSION_PREFIX=${SUBMISSION%_navsim_submission.pkl}
if [ "$SUBMISSION_PREFIX" = "$SUBMISSION" ]; then
    SUBMISSION_PREFIX=${SUBMISSION%.pkl}
fi
OUTPUT_DIR=${3:-${SUBMISSION_PREFIX}_official_pdms}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
FILTERED_CACHE=${OUTPUT_DIR}/metric_cache_index

echo "Official NAVSIM submission rescoring"
echo "  submission:  $SUBMISSION"
echo "  metric cache: $METRIC_CACHE"
echo "  output:       $OUTPUT_DIR"
echo "  shards:       $RESCORE_SHARDS"

"$PYTHON_BIN" "${SCRIPT_DIR}/prepare_navsim_metric_cache_index.py" \
    --submission "$SUBMISSION" \
    --metric-cache "$METRIC_CACHE" \
    --output-dir "$FILTERED_CACHE" \
    --num-shards "$RESCORE_SHARDS"

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
mapfile -t SHARD_CACHE_DIRS < <(
    find "$FILTERED_CACHE/shards" -mindepth 2 -maxdepth 2 -type d -name cache \
        | sort -V
)
if [ "${#SHARD_CACHE_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no metric-cache shards were generated." >&2
    exit 1
fi

# NAVSIM's PDM scorer is not safe under a large shared thread pool: on full
# navtest it can silently corrupt map-dependent metrics (DAC/progress/comfort).
# Run sequential scorers in separate OS processes and merge their token rows.
PIDS=()
SHARD_OUTPUT_DIRS=()
for SHARD_CACHE in "${SHARD_CACHE_DIRS[@]}"; do
    SHARD_INDEX=$(basename "$(dirname "$SHARD_CACHE")")
    SHARD_OUTPUT=${OUTPUT_DIR}/shards/${SHARD_INDEX}/output
    mkdir -p "$SHARD_OUTPUT"
    SHARD_OUTPUT_DIRS+=("$SHARD_OUTPUT")
    (
        OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-Prescott} \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        "$PYTHON_BIN" "$NAVSIM_OFFICIAL_SCRIPT" \
            train_test_split=navtest \
            worker=sequential \
            submission_file_path="$SUBMISSION" \
            metric_cache_path="$SHARD_CACHE" \
            output_dir="$SHARD_OUTPUT"
    ) >"${OUTPUT_DIR}/shards/${SHARD_INDEX}/stdout.log" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "ERROR: one or more official NAVSIM score shards failed." >&2
    echo "Inspect: ${OUTPUT_DIR}/shards/*/stdout.log" >&2
    exit 1
fi

SHARD_CSVS=()
for SHARD_OUTPUT in "${SHARD_OUTPUT_DIRS[@]}"; do
    LATEST_CSV=$(find "$SHARD_OUTPUT" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-)
    if [ -z "$LATEST_CSV" ]; then
        echo "ERROR: official NAVSIM scorer produced no CSV in $SHARD_OUTPUT" >&2
        exit 1
    fi
    SHARD_CSVS+=("$LATEST_CSV")
done

MERGED_CSV=${OUTPUT_DIR}/pdm_scores_merged.csv
"$PYTHON_BIN" "${SCRIPT_DIR}/merge_navsim_pdm_score_shards.py" \
    --output "$MERGED_CSV" \
    "${SHARD_CSVS[@]}"
echo "Official NAVSIM PDMS result: $MERGED_CSV"
