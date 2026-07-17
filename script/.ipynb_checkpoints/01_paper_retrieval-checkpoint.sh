#!/bin/bash
# ==============================================================================
# Step 1: Paper Retrieval from Europe PMC
# ==============================================================================
# Downloads full-text PDFs and supplementary materials for given PMC IDs
# using pygetpapers (https://pygetpapers.readthedocs.io/)
#
# Usage:
#   bash 01_paper_retrieval.sh --input <pmc_id_list.txt> --output <download_dir>
#
# Input:  A text file with one PMC ID per line (e.g., PMC12189289)
# Output: Downloaded PDFs in <output_dir>/PMC{ID}/fulltext.pdf + supplementaryfiles/
# ==============================================================================

set -euo pipefail

# === Default paths (override with arguments) ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_FILE="${PROJECT_DIR}/data_demo/pmc_id_list.txt"
OUTPUT_DIR="${PROJECT_DIR}/data_demo/1_download_pmc"
LOG_DIR="${PROJECT_DIR}/logs"

# === Parse arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)  INPUT_FILE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --log)    LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash 01_paper_retrieval.sh [--input <file>] [--output <dir>] [--log <dir>]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# === Setup ===
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
FAILED_LOG="${LOG_DIR}/failed_downloads.txt"
touch "$FAILED_LOG"

echo "======================================"
echo " Paper Retrieval from Europe PMC"
echo "======================================"
echo " Input file:  $INPUT_FILE"
echo " Output dir:  $OUTPUT_DIR"
echo " Log dir:     $LOG_DIR"
echo "======================================"

# === Check dependencies ===
if ! command -v pygetpapers &> /dev/null; then
    echo "[ERROR] pygetpapers not found. Install with: pip install pygetpapers"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] Input file not found: $INPUT_FILE"
    echo "  Create a text file with one PMC ID per line (e.g., PMC12189289)"
    exit 1
fi

# === Download papers ===
TOTAL=$(wc -l < "$INPUT_FILE")
COUNT=0
SUCCESS=0
FAILED=0

while IFS= read -r PMC_ID || [ -n "$PMC_ID" ]; do
    # Skip empty lines and comments
    [[ -z "$PMC_ID" || "$PMC_ID" == \#* ]] && continue
    
    COUNT=$((COUNT + 1))
    
    if [ -d "${OUTPUT_DIR}/${PMC_ID}" ]; then
        echo "[${COUNT}/${TOTAL}] SKIP (exists): ${PMC_ID}"
        continue
    fi
    
    echo "[${COUNT}/${TOTAL}] Downloading: ${PMC_ID}"
    if pygetpapers -q "$PMC_ID" --output "$OUTPUT_DIR" --pdf --supp 2>/dev/null; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✓ Success"
    else
        FAILED=$((FAILED + 1))
        echo "$PMC_ID" >> "$FAILED_LOG"
        echo "  ✗ Failed (logged to $FAILED_LOG)"
    fi
    
    # Avoid rate limiting
    sleep 1
done < "$INPUT_FILE"

echo ""
echo "======================================"
echo " Download Complete"
echo " Total: $COUNT | Success: $SUCCESS | Failed: $FAILED"
echo "======================================"
