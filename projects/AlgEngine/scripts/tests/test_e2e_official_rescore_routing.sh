#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALGENGINE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
EVAL_SCRIPT=${ALGENGINE_ROOT}/scripts/e2e_dist_eval.sh
FAILURES_EVAL_SCRIPT=${ALGENGINE_ROOT}/scripts/e2e_dist_eval_navtest_failures.sh

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "${TMP_DIR}/bin" "${TMP_DIR}/checkpoint" "${TMP_DIR}/world"
touch "${TMP_DIR}/checkpoint/model.pth"

cat >"${TMP_DIR}/bin/torchrun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${FAKE_CHECKPOINT_DIR}/test"
prefix="${FAKE_CHECKPOINT_DIR}/test/fake_run_${FAKE_RUN_ID}"
printf 'submission\n' >"${prefix}_navsim_submission.pkl"
{
    printf 'token,score\n'
    printf 'token-a,%s\n' "${FAKE_SCORE}"
    printf 'average,%s\n' "${FAKE_SCORE}"
} >"${prefix}.csv"
EOF
chmod +x "${TMP_DIR}/bin/torchrun"

cat >"${TMP_DIR}/fake_rescore.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$1" >"${FAKE_RESCORE_CAPTURE}"
EOF
chmod +x "${TMP_DIR}/fake_rescore.sh"

export PATH="${TMP_DIR}/bin:${PATH}"
export WORLDENGINE_ROOT="${TMP_DIR}/world"
export FAKE_CHECKPOINT_DIR="${TMP_DIR}/checkpoint"
export FAKE_RESCORE_CAPTURE="${TMP_DIR}/rescore.capture"
export NAVSIM_OFFICIAL_RESCORE_SCRIPT="${TMP_DIR}/fake_rescore.sh"

# Generated/non-selection results carry blank/NaN scores and must be rescored.
export FAKE_RUN_ID=generated
export FAKE_SCORE=
NAVSIM_OFFICIAL_RESCORE=auto bash "$EVAL_SCRIPT" \
    configs/navformer/e2e_diffusiondrive.py \
    "${TMP_DIR}/checkpoint/model.pth" \
    1 >/dev/null
test -f "$FAKE_RESCORE_CAPTURE" || fail "auto mode did not rescore generated results"
grep -F 'fake_run_generated_navsim_submission.pkl' "$FAKE_RESCORE_CAPTURE" >/dev/null || \
    fail "auto mode passed the wrong submission"

# Selection-model results already have numeric scores and must not be rescored.
rm -f "$FAKE_RESCORE_CAPTURE"
export FAKE_RUN_ID=selection
export FAKE_SCORE=0.75
NAVSIM_OFFICIAL_RESCORE=auto bash "$EVAL_SCRIPT" \
    configs/navformer/e2e_vadv2.py \
    "${TMP_DIR}/checkpoint/model.pth" \
    1 >/dev/null
test ! -f "$FAKE_RESCORE_CAPTURE" || fail "auto mode rescored selection results"

# A partially missing result set follows the dataset's any-NaN contract and
# must be officially rescored as a whole.
rm -f "$FAKE_RESCORE_CAPTURE"
export FAKE_RUN_ID=mixed
export FAKE_SCORE=0.75
export FAKE_EXTRA_SCORE=
cat >"${TMP_DIR}/bin/torchrun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${FAKE_CHECKPOINT_DIR}/test"
prefix="${FAKE_CHECKPOINT_DIR}/test/fake_run_${FAKE_RUN_ID}"
printf 'submission\n' >"${prefix}_navsim_submission.pkl"
{
    printf 'token,score\n'
    printf 'token-a,%s\n' "${FAKE_SCORE}"
    printf 'token-b,%s\n' "${FAKE_EXTRA_SCORE}"
    printf 'average,%s\n' "${FAKE_SCORE}"
} >"${prefix}.csv"
EOF
chmod +x "${TMP_DIR}/bin/torchrun"
NAVSIM_OFFICIAL_RESCORE=auto bash "$EVAL_SCRIPT" \
    configs/navformer/e2e_diffusiondrive.py \
    "${TMP_DIR}/checkpoint/model.pth" \
    1 >/dev/null
test -f "$FAKE_RESCORE_CAPTURE" || fail "auto mode ignored a partially missing score set"

# navtest_failures is a navtest subset and must use the same automatic route.
rm -f "$FAKE_RESCORE_CAPTURE"
export FAKE_RUN_ID=navtest_failures
export FAKE_SCORE=
export FAKE_EXTRA_SCORE=
NAVSIM_OFFICIAL_RESCORE=auto bash "$FAILURES_EVAL_SCRIPT" \
    configs/navformer/e2e_diffusiondrive.py \
    "${TMP_DIR}/checkpoint/model.pth" \
    1 >/dev/null
test -f "$FAKE_RESCORE_CAPTURE" || fail "navtest_failures did not invoke official rescoring"
grep -F 'fake_run_navtest_failures_navsim_submission.pkl' "$FAKE_RESCORE_CAPTURE" >/dev/null || \
    fail "navtest_failures passed the wrong submission"

# Explicit inference-only mode must never invoke the scorer.
rm -f "$FAKE_RESCORE_CAPTURE"
export FAKE_RUN_ID=never
export FAKE_SCORE=
NAVSIM_OFFICIAL_RESCORE=never bash "$EVAL_SCRIPT" \
    configs/navformer/e2e_diffusiondrive.py \
    "${TMP_DIR}/checkpoint/model.pth" \
    1 >/dev/null
test ! -f "$FAKE_RESCORE_CAPTURE" || fail "never mode invoked official rescoring"

bash -n "$EVAL_SCRIPT"
bash -n "$FAILURES_EVAL_SCRIPT"
bash -n "${ALGENGINE_ROOT}/scripts/e2e_navsim_rescore_utils.sh"
bash -n "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh"

# The standalone wrapper must prepare an exact cache index and invoke the
# external NAVSIM script, without implementing the scorer in AlgEngine.
FAKE_NAVSIM_ROOT=${TMP_DIR}/navsim
FAKE_CACHE=${TMP_DIR}/metric_cache
FAKE_SUBMISSION=${TMP_DIR}/submission.pkl
FAKE_OFFICIAL_OUTPUT=${TMP_DIR}/official_output
mkdir -p "${FAKE_NAVSIM_ROOT}/navsim/planning/script" "${FAKE_CACHE}/metadata"

python - "$FAKE_SUBMISSION" <<'PY'
import pickle
import sys

with open(sys.argv[1], "wb") as file:
    pickle.dump({"predictions": [{"token-a": None}]}, file)
PY

printf 'file_name\n%s\n' \
    "${FAKE_CACHE}/log/unknown/token-a/metric_cache.pkl" \
    >"${FAKE_CACHE}/metadata/source.csv"

cat >"${FAKE_NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_from_submission.py" <<'PY'
import csv
import os
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    if os.environ.get(variable) != "1":
        raise SystemExit(f"{variable} must be 1 for official NAVSIM scoring")

expected_coretype = os.environ.get("EXPECTED_OPENBLAS_CORETYPE", "Prescott")
if os.environ.get("OPENBLAS_CORETYPE") != expected_coretype:
    raise SystemExit(
        f"OPENBLAS_CORETYPE must be {expected_coretype} for official NAVSIM scoring"
    )

output_dir = next(value.split("=", 1)[1] for value in sys.argv if value.startswith("output_dir="))
cache_dir = next(value.split("=", 1)[1] for value in sys.argv if value.startswith("metric_cache_path="))
Path(output_dir).mkdir(parents=True, exist_ok=True)
metadata = next((Path(cache_dir) / "metadata").glob("*.csv"))
with metadata.open() as file:
    cache_rows = list(csv.DictReader(file))
tokens = [Path(row["file_name"]).parent.name for row in cache_rows]
with (Path(output_dir) / "official.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["token", "valid", "score"])
    writer.writeheader()
    for token in tokens:
        writer.writerow({"token": token, "valid": True, "score": 0.5})
    writer.writerow({"token": "average", "valid": True, "score": 0.5})
PY

# The wrapper has one supported topology: eight isolated scorer processes.
# A legacy environment override must not change that topology.
NAVSIM_RESCORE_SHARDS=1 NAVSIM_DEVKIT_ROOT="$FAKE_NAVSIM_ROOT" \
    PYTHON_BIN=python bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh" \
    "$FAKE_SUBMISSION" "$FAKE_CACHE" "$FAKE_OFFICIAL_OUTPUT" >/dev/null

test -f "${FAKE_OFFICIAL_OUTPUT}/pdm_scores_merged.csv" || \
    fail "standalone wrapper did not invoke the official scorer"
grep -F 'token-a/metric_cache.pkl' \
    "${FAKE_OFFICIAL_OUTPUT}/metric_cache_index/metadata/metric_cache.csv" >/dev/null || \
    fail "standalone wrapper did not prepare the submission-filtered cache index"

EXPECTED_OPENBLAS_CORETYPE=Haswell OPENBLAS_CORETYPE=Haswell \
    NAVSIM_DEVKIT_ROOT="$FAKE_NAVSIM_ROOT" PYTHON_BIN=python \
    bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh" \
    "$FAKE_SUBMISSION" "$FAKE_CACHE" "${FAKE_OFFICIAL_OUTPUT}_override" >/dev/null

echo "PASS: e2e official NAVSIM rescore routing"
