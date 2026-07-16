#!/usr/bin/env bash
set -euo pipefail

# Run the model in dataset mode with inline compositions (API-style)
# Usage: ./scripts/run_api.sh "comp1" "comp2" "comp3" ...
#
# Examples:
#   ./scripts/run_api.sh "LiPO3"
#   ./scripts/run_api.sh "LiPO3" "NaCl" "SrTiO3"

COMPS=("$@")
if [ ${#COMPS[@]} -eq 0 ]; then
	COMPS=("LiPO3" "NaCl" "SrTiO3")
fi

JSON_ARRAY=$(printf '%s\n' "${COMPS[@]}" | jq -R . | jq -s '.')

uv run model-run --local \
	--parameters "$(jq -n \
		--argjson comps "${JSON_ARRAY}" \
		'{mode: "dataset", inputs: {compositions: {value: $comps}}, parameters: {}}')"
