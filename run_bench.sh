#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# run_bench.sh — Run xteam benchmarks across compilers
#
# Usage:
#   ./run_bench.sh -r red_aomp_208 red_trunk_208 [...]
#   ./run_bench.sh -rsq -n5 -o results red_trunk_208 red_trunk_10400 [...]
#
# The binaries can be run $rounds times in a round-robin way so that changing
# machine load is distributed as evenly as possible. (Matters for shared
# machines.)
#
# Results are stored in $results_dir/<binary name>_round<N>.txt and a combined
# summary is printed at the end. ($results_dir defaults to "results")

set -euo pipefail

collect_only=0
rounds=1
results_dir=results
# Options passed to binaries
bench_iters=-1
evict_cache=0
quick_run=0
run=0
run_sim=0
warmup_iters=-1
custom_quick_array_size=0

# Add locale-independent thousand separators to make visual number parsing easier
format_number() {
  sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
}

usage() {
  echo "usage: $0 [-c] [-n rounds] [-o results_dir] [-b N] [-e] [-q] [-Q N] [-r] [-s] [-w N] [-h] binaries..."
  echo "  -c: Only collect results for the given number of rounds and the given labels, don't run any tests"
  echo "  -n rounds: Number of rounds to run for each label (default: $rounds)"
  echo "  -o results_dir: Results directory (default: $results_dir)"
  echo "  -h: Show this help message"
  echo
  echo "Options passed to binaries:"
  echo "  -b N: Benchmark iterations (default: auto-scaled such that the runtime per test is ~1 second (min 10 iterations))"
  echo "  -e: Evict the GPU L2/MALL cache before each iteration (cold-cache mode)"
  echo "  -q: Quick run (test only one array size)"
  echo "  -Q N: Quick run with custom array size N"
  echo "  -r: Run non-simulation tests"
  echo "  -s: Run simulation tests"
  echo "  -w N: Warmup iterations (default: 2)"
  echo
  echo "Note that at least one of -r or -s must be specified."
  echo
  echo "Pseudocode of how the benchmark binaries run the tests:"
  echo "  for each data type in alphabetical order (e.g. double, int, long):"
  echo "    for each array size in numerical order:"
  echo "      for each test:"
  echo "        for each warmup iteration:"
  echo "          run the test and check the result against the gold result"
  echo "        for each timed benchmark iteration:"
  echo "          run the test and check the result against the gold result"
}

while getopts "cn:o:b:eqQ:rsw:h" opt; do
  case "$opt" in
    c) collect_only=1 ;;
    n) rounds="$OPTARG" ;;
    o) results_dir="$OPTARG" ;;
    h) usage; exit 0 ;;
    # Options passed to binaries
    b) bench_iters="$OPTARG" ;;
    e) evict_cache=1 ;;
    q) quick_run=1 ;;
    Q) quick_run=1; custom_quick_array_size="$OPTARG" ;;
    r) run=1 ;;
    s) run_sim=1 ;;
    w) warmup_iters="$OPTARG" ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ $collect_only -eq 0 && $run -eq 0 && $run_sim -eq 0 ]]; then
  echo "Error: at least one of -r or -s must be specified" >&2
  usage; exit 1
fi

binaries=("$@")
if [[ ${#binaries[@]} -eq 0 ]]; then
  usage; exit 1
fi

# ── Run rounds ──────────────────────────────────────────────────────────────
if [[ $collect_only -eq 0 ]]; then
  # Build the arguments for the binaries
  args=()
  [[ $bench_iters -gt -1 ]] && args+=("-b" "$bench_iters")
  [[ $evict_cache -eq 1 ]] && args+=("-e")
  [[ $quick_run -eq 1 ]] && args+=("-q")
  [[ $custom_quick_array_size -gt 0 ]] && args+=("-Q" "$custom_quick_array_size")
  [[ $run -eq 1 ]] && args+=("-r")
  [[ $run_sim -eq 1 ]] && args+=("-s")
  [[ $warmup_iters -gt -1 ]] && args+=("-w" "$warmup_iters")

  mkdir -p "$results_dir"

  echo "═════════════════════════════════════════════════════════════════════"
  echo "xteam benchmark — round-robin runner"
  echo "═════════════════════════════════════════════════════════════════════"
  echo "Binaries:      ${binaries[*]}"
  echo "Arguments:     ${args[*]}"
  echo "Rounds:        $rounds"
  echo "Results:       $results_dir/"
  echo "═════════════════════════════════════════════════════════════════════"
  echo

  for (( round=1; round<=rounds; round++ )); do
    echo "━━━ Round $round / $rounds ━━━"

    for bin in "${binaries[@]}"; do
      outfile="$results_dir/${bin}_round${round}.txt"

      echo "  Running $bin (round $round)..."
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

# Two-row header: binary names right-aligned over their best/avg sub-columns.
# Each binary's best/avg sub-columns together span 2 + 15 + 2 + 15 = 34 chars.
# If your binary name is longer than 34 chars, this is an issue at the moment :)
# (it will just overflow)
# FIrst, skip the 24 + 1 + 8 + 1 + 15 = 49 chars of the test name, type, and N
# columns.
printf "%49s" ""
for bin in "${binaries[@]}"; do
  printf "%34s" "$bin"
done
echo
printf "%-24s %-8s %15s" "test" "type" "N"
for bin in "${binaries[@]}"; do
  printf "  %15s  %15s" "best" "avg"
done
echo

# Collect and form a set of all test lines from round 1 of all binaries to get
# the test list.
# (Not all binaries might have the same tests, so we need to collect all test
# lines from all binaries.)
mapfile -t round1_files < <(printf "${results_dir}/%s_round1.txt\n" "${binaries[@]}")
test_spec=$(grep -hEo '^\s*(red|scan)_\S+\s+\S+\s+[0-9,]+' "${round1_files[@]}" | sort -b -k2,2 -k3,3n -k1,1V -u)

extract_numeric_data() {
  local is_mbps=$1
  unset best avg

  if [[ $is_mbps -eq 1 ]]; then
    # Get best MB/s and avg MB/s
    mapfile -t best_list < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$7)}" "${file_list[@]}" | sed "s/,//g")
    mapfile -t avg_list  < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$8)}" "${file_list[@]}" | sed "s/,//g")
  else
    # Get min (= best) time (s) and avg time (s)
    mapfile -t best_list < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$4)}" "${file_list[@]}")
    mapfile -t avg_list  < <(awk "/^${test_name}\s+${type_name}\s+${n_val}\s+/ {print (\$4 == \"FAIL\" ? \"FAIL\" : \$6)}" "${file_list[@]}")
  fi

  if [[ ${#best_list[@]} -gt 0 ]]; then
    if [[ ! " ${best_list[*]} " =~ " FAIL " ]]; then
      if [[ $is_mbps -eq 1 ]]; then
        best=$(printf "%s\n" "${best_list[@]}" | sort -rn | head -1 | format_number)
      else
        best=$(printf "%s\n" "${best_list[@]}" | sort -n | head -1)
      fi
    else
      best="FAIL"
    fi
  fi
  if [[ ${#avg_list[@]} -gt 0 ]]; then
    if [[ ! " ${avg_list[*]} " =~ " FAIL " ]]; then
      avg=0
      for val in "${avg_list[@]}"; do
        avg=$(echo "$avg + $val" | bc -l)
      done
      if [[ $is_mbps -eq 1 ]]; then
        avg=$(echo "scale=0; $avg / ${#avg_list[@]}" | bc -l | format_number)
      else
        avg=$(echo "scale=6; $avg / ${#avg_list[@]}" | bc -l | sed 's/^\./0./')
      fi
    else
      avg="FAIL"
    fi
  fi
}

# Extract data lines (skip headers, blanks, and section markers)
echo "$test_spec" | while read -r test_name type_name n_val; do
  printf "%-24s %-8s %15s" "$test_name" "$type_name" "$n_val"

  for bin in "${binaries[@]}"; do
    mapfile -t file_list < <(printf "${results_dir}/${bin}_round%d.txt\n" $(seq 1 "$rounds"))

    extract_numeric_data 1
    # Some tests don't do memory accesses and thus have 0 MB/s. In these cases,
    # we fall back to time.
    if [[ -n ${best:-} && ${best:-0} == 0 || -n ${avg:-} && ${avg:-0} == 0 ]]; then
      extract_numeric_data 0
      printf "  %14s*  %14s*" "${best:-N/A}" "${avg:-N/A}"
    else
      printf "  %15s  %15s" "${best:-N/A}" "${avg:-N/A}"
    fi
  done
  echo
done

echo
echo "* = fallback to time (s) because the test has no memory accesses (this is not an error!)"
echo
echo "Per-round results saved in $results_dir/"
