#!/usr/bin/env bash
# The whole night's GPU work, in dependency order, unattended.
#
# Stage 1 runs the base-site rows, which need only the two midsem sites and
# so can start immediately. Stage 2 waits for the remaining downloads,
# prepares them, then runs the multi-biome rows. Stage 3 is the beta-NLL
# ablation, last because it is the only part that carries no primary claim.
#
# run_night.py is resumable: it skips any (run, seed) whose metrics are
# already in results_final.json. So killing and relaunching this script
# loses at most the run in flight, and relaunching is always safe.

set -u
cd "$(dirname "$0")" || exit 1
LOG=../results/night.log

say() { echo "[night $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== stage 1: base-site rows (R4 uncertainty, R3 consistency, R1a aug) ==="
python run_night.py --runs R4 R3 R1a --seeds 0 1 2 >> "$LOG" 2>&1
say "stage 1 exit $?"

# ---- wait for the two remaining site downloads -------------------------
say "=== stage 2: waiting for ANJI + NARYN ==="
for i in $(seq 1 240); do          # up to 2 hours
  if [ -f ../data/raw/ANJI.zip ] && [ -f ../data/raw/NARYN.zip ] \
     && [ ! -f ../data/raw/ANJI.zip.part ] && [ ! -f ../data/raw/NARYN.zip.part ]; then
    say "both archives present"
    break
  fi
  sleep 30
done

for s in ANJI NARYN; do
  if [ -f "../data/raw/$s.zip" ]; then
    if [ -f "../data/cache/${s}_lr.npy" ]; then
      say "$s already prepared"
    else
      # Cap at 1500 patches: bounds peak RAM (only ~3 GB free while the
      # training job runs) and keeps the biome mix balanced, since NARYN
      # uncapped is ~4000 patches against 900-1400 for every other site.
      say "preparing $s (capped at 1500 patches)"
      python prepare_data.py "$s" --limit 1500 >> "$LOG" 2>&1
      say "$s prepare exit $?"
    fi
  else
    say "$s never arrived; continuing without it"
  fi
done

say "=== stage 2: multi-biome rows (R1m, R1ma, R4m) ==="
python run_night.py --runs R1m R1ma R4m --seeds 0 1 2 >> "$LOG" 2>&1
say "stage 2 exit $?"

say "=== stage 3: beta-NLL ablation (1 seed, no primary claim) ==="
python run_night.py --runs R4b0 R4b05 --seeds 0 >> "$LOG" 2>&1
say "stage 3 exit $?"

say "=== ALL STAGES DONE ==="
