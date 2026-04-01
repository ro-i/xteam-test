#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# run_bench.sh — Run xteam benchmarks across compilers
#
# Usage:
#   ./run_bench.sh -r aomp_dev aomp trunk [...]
#   ./run_bench.sh -rRq -n5 -o results aomp_dev aomp trunk [...]
#
# The binaries can be run $rounds times in a round-robin way so that changing
# machine load is distributed as evenly as possible. (Only matters for shared
# machines.)
#
# Results are stored in results/<label>_round<N>.txt and a combined summary
# is printed at the end.

set -euo pipefail

# Prefix for binary names
# The binary name is constructed as $binary_prefix<label>
binary_prefix=xteam_bench_

collect_only=0
rounds=1
results_dir=results
# Options passed to binaries
bench_iters_reduction=1000
bench_iters_scan=10
quick_run=0
reduction=0
scan=0
reduction_simulation=0
scan_simulation=0
warmup_iters=2

# ── Parse options ───────────────────────────────────────────────────────────
usage() {
  echo "usage: $0 [-c] [-n rounds] [-o results_dir] [-b N] [-B N] [-q] [-r] [-s] [-R] [-S] [-w N] [-h] compiler_labels..."
  echo "  -c: Only collect results for the given number of rounds and the given labels, don't run any tests"
  echo "  -n rounds: Number of rounds to run for each label (default: $rounds)"
  echo "  -o results_dir: Results directory (default: $results_dir)"
  echo "  -h: Show this help message"
  echo
  echo "Options passed to binaries:"
  echo "  -b N: Benchmark iterations for reduction (default: $bench_iters_reduction)"
  echo "  -B N: Benchmark iterations for scan (default: $bench_iters_scan)"
  echo "  -q: Quick run (test only one array size) (default: $quick_run)"
  echo "  -r: Run reduction tests (default: $reduction)"
  echo "  -s: Run scan tests (default: $scan)"
  echo "  -R: Run reduction simulations (default: $reduction_simulation)"
  echo "  -S: Run scan simulations (default: $scan_simulation)"
  echo "  -w N: Warmup iterations (default: $warmup_iters)"
  echo
  echo "Note that at least one of -r, -s, -R, -S must be specified."
  echo
  echo "Pseudocode of how the benchmark binaries run the tests:"
  echo "  for each data type in alphabetical order (e.g. double, int, long):"
  echo "    for each array size in numerical order:"
  echo "      for each test type in alphabetical order (first all reductions, then all scans):"
  echo "        for each warmup iteration:"
  echo "          run the test and check the result against the gold result"
  echo "        for each timed benchmark iteration:"
  echo "          run the test and check the result against the gold result"
}

while getopts "cn:o:b:B:qrsRSw:h" opt; do
  case "$opt" in
    c) collect_only=1 ;;
    n) rounds="$OPTARG" ;;
    o) results_dir="$OPTARG" ;;
    h) usage; exit 0 ;;
    # Options passed to binaries
    b) bench_iters_reduction="$OPTARG" ;;
    B) bench_iters_scan="$OPTARG" ;;
    q) quick_run=1 ;;
    r) reduction=1 ;;
    s) scan=1 ;;
    R) reduction_simulation=1 ;;
    S) scan_simulation=1 ;;
    w) warmup_iters="$OPTARG" ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ $collect_only -eq 0 && $reduction -eq 0 && $scan -eq 0 && $reduction_simulation -eq 0 && $scan_simulation -eq 0 ]]; then
  echo "Error: at least one of -r, -s, -R, -S must be specified" >&2
  usage; exit 1
fi

labels=("$@")
if [[ ${#labels[@]} -eq 0 ]]; then
  usage; exit 1
fi
# Derive binary names from labels (add xteam_bench_ prefix)
binaries=()
for label in "${labels[@]}"; do
  binaries+=("$binary_prefix$label")
done

# ── Run rounds ──────────────────────────────────────────────────────────────
if [[ $collect_only -eq 0 ]]; then
  # Build the arguments for the binaries
  args=()
  args+=("-b" "$bench_iters_reduction")
  args+=("-B" "$bench_iters_scan")
  [[ $quick_run -eq 1 ]] && args+=("-q")
  [[ $reduction -eq 1 ]] && args+=("-r")
  [[ $scan -eq 1 ]] && args+=("-s")
  [[ $reduction_simulation -eq 1 ]] && args+=("-R")
  [[ $scan_simulation -eq 1 ]] && args+=("-S")
  args+=("-w" "$warmup_iters")

  mkdir -p "$results_dir"

  echo "═════════════════════════════════════════════════════════════════════"
  echo "xteam benchmark — round-robin runner"
  echo "═════════════════════════════════════════════════════════════════════"
  echo "Binaries:      ${binaries[*]}"
  echo "Labels:        ${labels[*]}"
  echo "Arguments:     ${args[*]}"
  echo "Rounds:        $rounds"
  echo "Results:       $results_dir/"
  echo "═════════════════════════════════════════════════════════════════════"
  echo

  for (( round=1; round<=rounds; round++ )); do
    echo "━━━ Round $round / $rounds ━━━"

    for i in $(seq 0 $(( ${#binaries[@]} - 1 ))); do
      bin="${binaries[$i]}"
      label="${labels[$i]}"
      outfile="$results_dir/${label}_round${round}.txt"

      echo "  Running $label (round $round)..."
      if ! stdbuf -oL -eL "./$bin" "${args[@]}" 2>&1 | tee "$outfile"; then
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
for label in "${labels[@]}"; do
  printf "  %15s" "$label (best)"
  printf "  %15s" "$label (avg)"
done
echo

# Collect and form a set of all test lines from round 1 of all labels to get the
# test list.
# (Not all labels might have the same tests, so we need to collect all test
# lines from all labels.)
mapfile -t round1_files < <(printf "${results_dir}/%s_round1.txt\n" "${labels[@]}")
test_spec=$(grep -hEo '^\s*(red|scan)_\S+\s+\S+\s+[0-9]+' "${round1_files[@]}" | sort -b -k2,2 -k3,3n -k1,1V -u)

# Extract data lines (skip headers, blanks, and section markers)
echo "$test_spec" | while read -r test_name type_name n_val; do
  printf "%-24s %-8s %10s" "$test_name" "$type_name" "$n_val"

  for label in "${labels[@]}"; do
    unset best_mbps avg_mbps

    mapfile -t file_list < <(printf "${results_dir}/${label}_round%d.txt\n" $(seq 1 "$rounds"))
    mapfile -t best_mbps_list < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$7)}" "${file_list[@]}")
    mapfile -t avg_mbps_list  < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$8)}" "${file_list[@]}")

    if [[ ${#best_mbps_list[@]} -gt 0 ]]; then
      if [[ ! " ${best_mbps_list[*]} " =~ " FAIL " ]]; then
        best_mbps=$(printf "%s\n" "${best_mbps_list[@]}" | sort -rn | head -1)
      else
        best_mbps="FAIL"
      fi
    fi
    if [[ ${#avg_mbps_list[@]} -gt 0 ]]; then
      if [[ ! " ${avg_mbps_list[*]} " =~ " FAIL " ]]; then
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
echo "Per-round results saved in $results_dir/"
