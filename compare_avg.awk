#!/usr/bin/awk -f
#
# Compare the "avg" columns of the dev vs non-dev build variants in the
# benchmark summary table and report how much better the _dev_ variants are.
#
# Usage:
#   awk -f compare_avg.awk ./result_2026-06-01
#
# Only rows whose first field starts with "red_" are processed.
#
# Whitespace-separated column layout (numbers use ',' thousands separators):
#   $1 test   $2 type   $3 N
#   $4 best  / $5  avg   -> red_trunk_208
#   $6 best  / $7  avg   -> red_trunk_dev_208
#   $8 best  / $9  avg   -> red_trunk_10400
#   $10 best / $11 avg   -> red_trunk_dev_10400
#
# Throughput rows are MB/s (higher is better). Rows marked with '*' fell back
# to time in ms (lower is better), so the comparison is inverted for them.

# Parse a table cell into a number: strip thousands commas and the '*' marker.
function num(s) {
    gsub(/,/, "", s)
    sub(/\*$/, "", s)
    return s + 0
}

# True when a cell carries the '*' ms-fallback marker (lower is better).
function is_ms(s) {
    return (s ~ /\*/)
}

# Percent by which `dev` is better than `base`.
# Positive => dev is better. Direction flips when lower_is_better is set.
function pct(base, dev, lower_is_better) {
    if (base == 0)
        return 0
    if (lower_is_better)
        return (base - dev) / base * 100
    return (dev - base) / base * 100
}

/^red_/ {
    lower_is_better = is_ms($5)

    avg_208        = num($5)
    avg_dev_208    = num($7)
    avg_10400      = num($9)
    avg_dev_10400  = num($11)

    p208   = pct(avg_208,   avg_dev_208,   lower_is_better)
    p10400 = pct(avg_10400, avg_dev_10400, lower_is_better)

    printf "%-22s %-7s  dev_208: %+8.2f%%   dev_10400: %+8.2f%%\n", \
           $1, $2, p208, p10400
}
