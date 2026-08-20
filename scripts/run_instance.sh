#!/usr/bin/env bash
set -euo pipefail

# Run the model in instance mode locally (no S3 required)
# Usage: ./scripts/run_instance.sh [composition]
#
# Examples:
#   ./scripts/run_instance.sh LiPO3
#   ./scripts/run_instance.sh NaCl

QUERY="${1:-LiPO3}"

uv run model-run --local \
	--parameters "$(jq -n \
		--arg query "${QUERY}" \
		'{mode: "instance", inputs: {query: {value: $query}}, parameters: {}}')"
