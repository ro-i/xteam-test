#!/usr/bin/awk -f
#
# Compare the "avg" columns of two build variants in the benchmark summary
# table and report how much better the second variants are.
#
# Usage:
#   awk -f compare_avg.awk ./result_2026-06-01
#
# Only rows whose first field starts with "red_" or "misc_" are processed.
#
# Example of the whitespace-separated column layout (numbers use ',' thousands
# separators):
#   $1 test   $2 type   $3 N
#   $4 best  / $5  avg   -> red_<label1>_208x512
#   $6 best  / $7  avg   -> red_<label2>_208x512
#   $8 best  / $9  avg   -> red_<label1>_10400x512
#   $10 best / $11 avg   -> red_<label2>_10400x512
#
# Throughput rows are MB/s (higher is better). Rows marked with '*' fell back
# to time in ms (lower is better), so the comparison is inverted for them.

# Parse a table cell into a number: strip thousands commas and the '*' marker.
function num(s) {
    gsub(/,/, "", s)
    sub(/\*$/, "", s)
    return s + 0 # idiom for converting string to int
}

# True when a cell carries the '*' ms-fallback marker (lower is better).
function is_ms(s) {
    return (s ~ /\*/)
}

# Percent by which `label2` is better than `label1`.
# Positive => label2 is better. Direction flips when lower_is_better is set.
function pct(base, dev, lower_is_better) {
    if (base == 0)
        return 0
    if (lower_is_better)
        return (base - dev) / base * 100
    return (dev - base) / base * 100
}

/^(red_|misc_)/ {
    lower_is_better = is_ms($5)

    avg_label1 = num($5)
    avg_label2 = num($7)
    if (NF > 7) {
        avg_label1_2 = num($9)
        avg_label2_2 = num($11)
    }

    p1 = pct(avg_label1,   avg_label2,   lower_is_better)
    p2 = pct(avg_label1_2, avg_label2_2, lower_is_better)

    if (NF > 7)
        printf "%-22s %-7s  change for <...>: %+8.2f%%   change for <...>: %+8.2f%%\n", $1, $2, p1, p2
    else
        printf "%-22s %-7s  change for <...>: %+8.2f%%\n", $1, $2, p1
}
