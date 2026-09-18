#!/usr/bin/env bash
# Full ReflexRL pipeline, resumable at every stage. Runs under WSL Linux.
#   bash scripts/pipeline.sh   (from repo root)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/opt/rrl/bin/python
export HF_HOME=/mnt/d/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
log() { echo "$(date '+%F %T') $*" | tee -a runs/logs/pipeline.log; }

log "stage 1: Qwen teacher gate (dtc hg dc hgs)"
$PY -u scripts/teacher_gate.py --policy qwen --scenarios dtc hg dc hgs --episodes 30 \
  >> runs/logs/gate.log 2>&1 || { log "gate FAILED"; exit 1; }

log "stage 2: teacher proxies / BC"
for s in dtc hg dc hgs; do
  name=$($PY -c "import sys;sys.path.insert(0,'.');from reflexrl.env.scenarios import get_scenario;print(get_scenario('$s').name)")
  if [ -f runs/teachers/$name/teacher.json ]; then log "  $s proxy exists"; continue; fi
  $PY -u scripts/build_teacher.py --scenario $s --labels runs/phase0/Qwen3-VL-2B-Instruct/labels \
    >> runs/logs/teachers.log 2>&1 || { log "  build_teacher $s FAILED"; exit 1; }
  log "  $(tail -1 runs/logs/teachers.log)"
done

log "stage 3: main runs"
$PY -u scripts/run_queue.py experiments/configs/plan_main.json --n-envs 12 >> runs/logs/queue_main.log 2>&1
log "stage 4: held-out adaptation"
$PY -u scripts/run_queue.py experiments/configs/plan_heldout.json --n-envs 12 >> runs/logs/queue_heldout.log 2>&1
log "stage 5: analysis"
$PY -u scripts/analyze.py --out results >> runs/logs/analyze.log 2>&1
log "pipeline complete"
