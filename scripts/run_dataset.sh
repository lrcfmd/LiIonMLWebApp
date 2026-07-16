#!/usr/bin/env bash
set -euo pipefail

# Run the model in dataset mode with a CSV file (no S3 required)
# Usage: ./scripts/run_dataset.sh [csv_file]
#
# The CSV must have a 'composition' column.
# Examples:
#   ./scripts/run_dataset.sh data/compositions.csv
#   ./scripts/run_dataset.sh path/to/my_compositions.csv

FILE="${1:-data/compositions.csv}"

if [ ! -f "$FILE" ]; then
	echo "Error: File not found: $FILE" >&2
	exit 1
fi

uv run model-run --local \
	--parameters "$(jq -n \
		--arg file "file://${FILE}" \
		'{mode: "dataset", inputs: {file: {uri: $file, mime_type: "text/csv"}}, parameters: {}}')"
