#!/usr/bin/env bash
# Fig. 3 campaign — blocks with tqdm progress shown in the pane.
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1
DIR=reproductions/planned/dw_timing_gas_hollowcore
run_block () {
  for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    python $DIR/probe_scan.py --shard "$i" 20 "$1" "$2" \
      >> "$DIR/scan_shards.log" 2>&1 &
  done
  echo "BLOCK RUNNING: $1 $2"
  wait
  echo "BLOCK_DONE: $1 $2"
}
rem=()
grep -c point $DIR/rdw_scan_0.8bar.jsonl 2>/dev/null | grep -q 800 || rem_block0="--pressure 0.8"
for p in 0.8 1.5 2.1 3.0 4.0; do run_block --pressure "$p"; done
for p in 1.2 2.2 3.2 4.5 6.0; do run_block --gradient "$p"; done
echo ALL_BLOCKS_DONE
