#!/usr/bin/env bash
# gpu_availability.sh — Show free/used/total GPUs by type across the cluster.
#
# Usage:
#   bash shells/gpu_availability.sh

set -euo pipefail

echo "============================================================"
echo "GPU Availability by Type"
printf "%-12s %6s %6s %6s %6s   %s\n" "GPU TYPE" "FREE" "USED" "TOTAL" "DOWN" "PARTITIONS"
echo "------------------------------------------------------------"

# Parse: partition, gres (e.g. gpu:h100:4), state, nodelist
# States: idle=fully free, mix=partially used, alloc=fully used, down/drain/inval=unavailable
declare -A gpu_free gpu_used gpu_total gpu_down gpu_partitions

while IFS=' ' read -r partition gres state _nodes; do
    # Only lines with actual GPU gres
    [[ "$gres" == gpu:*:* ]] || continue

    # Extract GPU type and count-per-node: gpu:h100:4
    gpu_type=$(echo "$gres" | cut -d: -f2)
    gpus_per_node=$(echo "$gres" | cut -d: -f3 | cut -d'(' -f1)

    # Count nodes in this state by expanding the nodelist
    node_count=$(sinfo -h -o "%D" -p "$partition" --state="$state" -n "$_nodes" 2>/dev/null || echo 0)
    node_count="${node_count:-0}"
    [[ "$node_count" =~ ^[0-9]+$ ]] || node_count=0

    total_gpus=$(( node_count * gpus_per_node ))

    # Accumulate
    gpu_total[$gpu_type]=$(( ${gpu_total[$gpu_type]:-0} + total_gpus ))
    gpu_partitions[$gpu_type]="${gpu_partitions[$gpu_type]:-} $partition"

    case "$state" in
        idle)
            gpu_free[$gpu_type]=$(( ${gpu_free[$gpu_type]:-0} + total_gpus ))
            ;;
        mix)
            # mix means some GPUs on the node are free; count allocated via squeue
            # approximate: add to used but not free (conservative)
            gpu_used[$gpu_type]=$(( ${gpu_used[$gpu_type]:-0} + total_gpus ))
            ;;
        alloc)
            gpu_used[$gpu_type]=$(( ${gpu_used[$gpu_type]:-0} + total_gpus ))
            ;;
        down*|drain*|inval*)
            gpu_down[$gpu_type]=$(( ${gpu_down[$gpu_type]:-0} + total_gpus ))
            ;;
    esac

done < <(sinfo -o "%P %G %t %N" --noheader 2>/dev/null \
         | grep 'gpu:' \
         | awk '!seen[$1$2$3]++')   # deduplicate same partition/gres/state rows

# Print summary, deduplicating partitions per GPU type
for gpu_type in $(echo "${!gpu_total[@]}" | tr ' ' '\n' | sort); do
    free=${gpu_free[$gpu_type]:-0}
    used=${gpu_used[$gpu_type]:-0}
    total=${gpu_total[$gpu_type]:-0}
    down=${gpu_down[$gpu_type]:-0}

    # Deduplicate and shorten partition list
    parts=$(echo "${gpu_partitions[$gpu_type]}" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

    printf "%-12s %6d %6d %6d %6d   %s\n" "$gpu_type" "$free" "$used" "$total" "$down" "$parts"
done

echo "============================================================"
echo ""
echo "Note: 'mix' nodes counted as used (some GPUs may be free)."
echo "For exact free GPUs on mix nodes, run:"
echo "  squeue -o '%.R %.b' | grep gpu | sort | uniq -c"
