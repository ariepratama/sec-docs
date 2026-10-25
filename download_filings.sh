#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "${SCRIPT_DIR}/sec/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/sec/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Top 20 major US companies by market cap
COMPANY_LIST="${COMPANY_LIST:-AAPL,MSFT,NVDA,AMZN,GOOGL,META,BRK-B,LLY,AVGO,TSLA,JPM,WMT,UNH,V,XOM,MA,PG,COST,JNJ,HD}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data/fillings}"
START_YEAR="${START_YEAR:-2010}"
CURRENT_YEAR="$(date +%Y)"
CURRENT_MONTH="$(date +%-m)"
CURRENT_QUARTER=$(( (CURRENT_MONTH - 1) / 3 + 1 ))

echo "============================================================"
echo "Downloading SEC filings from ${START_YEAR} Q1 to ${CURRENT_YEAR} Q${CURRENT_QUARTER}"
echo "Companies: ${COMPANY_LIST}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python binary: ${PYTHON_BIN}"
echo "============================================================"

for (( year=START_YEAR; year<=CURRENT_YEAR; year++ )); do
    max_quarter=4
    if (( year == CURRENT_YEAR )); then
        max_quarter="${CURRENT_QUARTER}"
    fi

    for (( quarter=1; quarter<=max_quarter; quarter++ )); do
        echo "------------------------------------------------------------"
        echo "Processing Year=${year} Quarter=Q${quarter}"
        echo "------------------------------------------------------------"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/download_filling.py" \
            --company_list "${COMPANY_LIST}" \
            --year "${year}" \
            --quarter "${quarter}" \
            --output_dir "${OUTPUT_DIR}"
    done
done

echo "============================================================"
echo "All requested filings from ${START_YEAR} to ${CURRENT_YEAR} Q${CURRENT_QUARTER} processed."
echo "============================================================"

