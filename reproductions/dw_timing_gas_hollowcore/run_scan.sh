#!/usr/bin/env bash
# Fig. 2 probe scan campaign: 20 shards of the 800-point energy scan.
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  python reproductions/planned/dw_timing_gas_hollowcore/probe_scan.py \
    --shard "$i" 20 \
    > "reproductions/planned/dw_timing_gas_hollowcore/scan_shard_$i.log" 2>&1 &
done
wait
echo SCAN_SHARDS_COMPLETE
