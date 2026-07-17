#!/usr/bin/env bash

NAVSIM_RESCORE_UTILS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

validate_navsim_official_rescore_mode() {
    NAVSIM_OFFICIAL_RESCORE=${NAVSIM_OFFICIAL_RESCORE:-auto}
    case "$NAVSIM_OFFICIAL_RESCORE" in
        auto|always|never) ;;
        *)
            echo "ERROR: NAVSIM_OFFICIAL_RESCORE must be auto, always, or never." >&2
            return 2
            ;;
    esac
}

maybe_run_navsim_official_rescore() {
    if [ "$#" -ne 2 ]; then
        echo "ERROR: maybe_run_navsim_official_rescore expects <test_dir> <run_marker>." >&2
        return 2
    fi

    local test_dir=$1
    local run_marker=$2
    local submission
    local result_csv
    local should_rescore
    local python_bin=${PYTHON_BIN:-python}

    validate_navsim_official_rescore_mode
    if [ "$NAVSIM_OFFICIAL_RESCORE" = "never" ]; then
        echo "NAVSIM official rescoring disabled (NAVSIM_OFFICIAL_RESCORE=never)."
        return 0
    fi

    submission=$(find "$test_dir" -maxdepth 1 -type f \
        -name '*_navsim_submission.pkl' -newer "$run_marker" -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-)
    if [ -z "$submission" ]; then
        if [ "$NAVSIM_OFFICIAL_RESCORE" = "always" ]; then
            echo "ERROR: evaluation did not export a NAVSIM submission under $test_dir" >&2
            return 1
        fi
        echo "No new NAVSIM submission exported; skipping official rescoring."
        return 0
    fi

    result_csv=${submission%_navsim_submission.pkl}.csv
    should_rescore=$NAVSIM_OFFICIAL_RESCORE
    if [ "$NAVSIM_OFFICIAL_RESCORE" = "auto" ]; then
        if [ ! -f "$result_csv" ]; then
            echo "ERROR: cannot classify evaluation mode; result CSV not found: $result_csv" >&2
            return 1
        fi
        should_rescore=$("$python_bin" - "$result_csv" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="") as file:
    rows = list(csv.DictReader(file))

def missing(value):
    if value is None or not value.strip():
        return True
    try:
        return math.isnan(float(value))
    except ValueError:
        return False

scores = [row.get("score") for row in rows if row.get("token") != "average"]
print("always" if scores and any(missing(value) for value in scores) else "never")
PY
        )
    fi

    if [ "$should_rescore" = "always" ]; then
        echo "Generated/non-selection trajectories detected; starting official NAVSIM rescoring."
        bash "${NAVSIM_OFFICIAL_RESCORE_SCRIPT:-${NAVSIM_RESCORE_UTILS_DIR}/e2e_navsim_official_rescore.sh}" \
            "$submission"
    else
        echo "Selection-model scores detected; official NAVSIM rescoring is not required."
    fi
}
