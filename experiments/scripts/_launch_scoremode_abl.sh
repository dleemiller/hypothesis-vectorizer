#!/usr/bin/env bash
# Resilient, low-priority launcher for the sst2 + ag_news score_mode ablation (neutral-axis test).
# Same niced/retry pattern as _launch_head_channels.sh — the box runs other GPU/CPU jobs. Reuses the
# probe's warm NLI cache so the fixed test-set pairs don't re-score.
set -u
cd "$(dirname "$0")/../.." || exit 1
export HV_CACHE_DIR="${HV_CACHE_DIR:-/tmp/hv_head_channels_cache}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
LOG=experiments/results/logs/abl_scoremode_sst2_agnews.log
common="--encoder dleemiller/finecat-nli-l --axis score_mode --train-size 100 --seeds 5 --test-size 2000 --test-seed 7"
for ds in sst2 ag_news; do
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "=== $ds attempt $attempt $(date -u +%H:%M:%S) (free $(free -m|awk '/Mem:/{print $7}')MB, load $(cut -d' ' -f1 /proc/loadavg)) ===" >> "$LOG"
    nice -n 19 uv run python experiments/scripts/run_ablation.py --dataset "$ds" $common \
      --run-id "abl_${ds}_scoremode" >> "$LOG" 2>&1
    if [ $? -eq 0 ]; then echo "=== $ds SUCCESS ===" >> "$LOG"; ok=1; break; fi
    echo "=== $ds attempt $attempt FAILED, backing off ===" >> "$LOG"; sleep $((attempt * 120))
  done
  [ $ok -eq 0 ] && { echo "=== $ds GAVE UP ===" >> "$LOG"; }
done
echo "=== ALL DONE ===" >> "$LOG"
