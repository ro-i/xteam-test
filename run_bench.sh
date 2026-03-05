#!/usr/bin/env bash
# run_bench.sh — Run xteam benchmarks interleaved across compilers
#
# Usage:
#   ./run_bench.sh xteam_bench_dev xteam_bench_rel [...]
#   ./run_bench.sh -n 5 -o results xteam_bench_dev xteam_bench_rel
#
# Each binary is run ROUNDS times.  Within each round the binaries execute
# in shuffled order so that transient machine load affects all compilers
# equally rather than penalising whichever runs last.
#
# Results are stored in results/<label>_round<N>.txt and a combined summary
# is printed at the end.

set -euo pipefail

ONLY_COLLECT=0
ROUNDS=5
RESULTS_DIR=results

# ── Parse options ───────────────────────────────────────────────────────────
usage() {
  echo "usage: $0 [-n rounds] [-o results_dir] binary1 [binary2 ...]"
  echo "  -c: Only collect results, don't run any tests"
  echo "  -n rounds: Number of rounds to run (default: $ROUNDS)"
  echo "  -o results_dir: Results directory (default: $RESULTS_DIR)"
  echo "  -h: Show this help message"
}

while getopts "cn:o:qh" opt; do
  case "$opt" in
    c) ONLY_COLLECT=1 ;;
    n) ROUNDS="$OPTARG" ;;
    o) RESULTS_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

BINARIES=("$@")
if [[ ${#BINARIES[@]} -eq 0 ]]; then
  usage; exit 1
fi

# Derive labels from binary names (strip xteam_bench_ prefix if present)
LABELS=()
for bin in "${BINARIES[@]}"; do
  label="${bin##*/}"
  label="${label#xteam_bench_}"
  LABELS+=("$label")
done

mkdir -p "$RESULTS_DIR"

echo "═════════════════════════════════════════════════════════════════════"
echo "xteam benchmark — interleaved runner"
echo "═════════════════════════════════════════════════════════════════════"
echo "Binaries:      ${BINARIES[*]}"
echo "Labels:        ${LABELS[*]}"
echo "Rounds:        $ROUNDS"
echo "Only collect:  $ONLY_COLLECT"
echo "Results:       $RESULTS_DIR/"
echo "═════════════════════════════════════════════════════════════════════"
echo

# ── Run rounds ──────────────────────────────────────────────────────────────
if [[ $ONLY_COLLECT -eq 0 ]]; then
  for (( round=1; round<=ROUNDS; round++ )); do
    echo "━━━ Round $round / $ROUNDS ━━━"

    # Create a shuffled index array for this round
    indices=($(seq 0 $(( ${#BINARIES[@]} - 1 )) | shuf))

    for idx in "${indices[@]}"; do
      bin="${BINARIES[$idx]}"
      label="${LABELS[$idx]}"
      outfile="$RESULTS_DIR/${label}_round${round}.txt"

      echo "  Running $label (round $round)..."
      if ! stdbuf -oL -eL "./$bin" 2>&1 | tee "$outfile"; then
        echo "  WARNING: $bin exited with non-zero status" >&2
      fi
    done
    echo
  done
fi

# ── Summarise results ───────────────────────────────────────────────────────
echo "═════════════════════════════════════════════════════════════════════"
echo "Summary: best MB/s and avg MB/s across all rounds (higher is better)"
echo "═════════════════════════════════════════════════════════════════════"
echo

# Header
printf "%-24s %-8s %10s" "test" "type" "N"
for label in "${LABELS[@]}"; do
  printf "  %15s" "$label (best)"
  printf "  %15s" "$label (avg)"
done
echo

# Collect all test lines from round 1 of the first label to get the test list
first_file="$RESULTS_DIR/${LABELS[0]}_round1.txt"
if [[ ! -f "$first_file" ]]; then
  echo "No results found in $first_file" >&2
  exit 1
fi

# Extract data lines (skip headers, blanks, and section markers)
grep -oE '^\s*(red|excl|incl)_\S+\s+\S+\s+[0-9]+' "$first_file" | while read -r test_name type_name n_val; do
  printf "%-24s %-8s %10s" "$test_name" "$type_name" "$n_val"

  for label in "${LABELS[@]}"; do
    unset best_mbps avg_mbps

    mapfile -t file_list < <(printf "${RESULTS_DIR}/${label}_round%d.txt\n" $(seq 1 $ROUNDS))
    mapfile -t best_mbps_list < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$7)}" "${file_list[@]}")
    mapfile -t avg_mbps_list  < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$8)}" "${file_list[@]}")

    if [[ ${#best_mbps_list[@]} -gt 0 ]]; then
      if [[ ! " ${best_mbps_list[@]} " =~ " FAIL " ]]; then
        best_mbps=$(printf "%s\n" "${best_mbps_list[@]}" | sort -rn | head -1)
      else
        best_mbps="FAIL"
      fi
    fi
    if [[ ${#avg_mbps_list[@]} -gt 0 ]]; then
      if [[ ! " ${avg_mbps_list[@]} " =~ " FAIL " ]]; then
        avg_mbps=0
        for mbps in "${avg_mbps_list[@]}"; do
          avg_mbps=$(echo "$avg_mbps + $mbps" | bc -l)
        done
        avg_mbps=$(echo "scale=0; $avg_mbps / ${#avg_mbps_list[@]}" | bc -l)
      else
        avg_mbps="FAIL"
      fi
    fi

    printf "  %15s  %15s" "${best_mbps:-N/A}" "${avg_mbps:-N/A}"
  done
  echo
done

echo
echo "Per-round results saved in $RESULTS_DIR/"
