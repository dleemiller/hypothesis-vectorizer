#!/usr/bin/env bash
# Pool-size axis at the smaller encoders (-xs,-s,-m); -l already exists as abl_trec_poolsize.
# Same protocol as that run. Niced/retry for the contended box; then render the crossed figure.
set -u
cd "$(dirname "$0")/../.." || exit 1
export HV_CACHE_DIR="${HV_CACHE_DIR:-/tmp/hv_head_channels_cache}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
LOG=experiments/results/logs/poolsize_x_encoder.log
common="--dataset trec --axis pool_size --pool-json experiments/results/pools/trec_gen256.json \
  --pool-sizes 8 16 32 64 128 192 256 --train-size 20 --seeds 5 --test-size 2000 --test-seed 7"
for size in xs s m; do
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "=== -$size attempt $attempt $(date -u +%H:%M:%S) (free $(free -m|awk '/Mem:/{print $7}')MB load $(cut -d' ' -f1 /proc/loadavg)) ===" >> "$LOG"
    nice -n 19 uv run python experiments/scripts/run_ablation.py $common \
      --encoder "dleemiller/finecat-nli-$size" --run-id "abl_trec_poolsize_$size" >> "$LOG" 2>&1
    if [ $? -eq 0 ]; then echo "=== -$size SUCCESS ===" >> "$LOG"; ok=1; break; fi
    echo "=== -$size attempt $attempt FAILED, backing off ===" >> "$LOG"; sleep $((attempt*120))
  done
  [ $ok -eq 0 ] && echo "=== -$size GAVE UP ===" >> "$LOG"
done
echo "=== rendering figure ===" >> "$LOG"
nice -n 19 uv run python experiments/scripts/fig_poolsize_x_encoder.py >> "$LOG" 2>&1
echo "=== ALL DONE ===" >> "$LOG"
