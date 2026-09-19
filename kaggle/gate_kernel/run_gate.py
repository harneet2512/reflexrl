"""Kaggle job: fp32 calibrated Qwen3-VL-2B teacher gate + labels (T4, Linux).

Uses both T4s when present: scenarios are split into two lanes that run in
parallel, one per GPU, each resumable per scenario and restarted on a crash.
Results + labels land in /kaggle/working/runs/phase0/lane*/ for download.
"""
import glob
import os
import shutil
import subprocess
import sys

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 transformers==5.17.0 "
   "accelerate opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
assert src, "code dataset not mounted"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
lanes = [["dtc", "dc"], ["hg", "hgs"]] if n_gpu >= 2 else [["dtc", "dc", "hg", "hgs"]]
print(f"GPUs: {n_gpu}; lanes: {lanes}", flush=True)

def lane_cmd(i, scen):
    gate = (f"{sys.executable} -u scripts/teacher_gate.py --policy qwen --variant calibrated "
            f"--scenarios {' '.join(scen)} --episodes 30 --n-envs 8 --out /kaggle/working/runs/phase0/lane{i}")
    # up to 4 attempts; the gate resumes at the first unfinished scenario
    return (f"export CUDA_VISIBLE_DEVICES={i if n_gpu >= 2 else 0}; "
            f"for a in 1 2 3 4; do {gate} >> /kaggle/working/lane{i}.log 2>&1 && break; done")

procs = [subprocess.Popen(lane_cmd(i, scen), shell=True) for i, scen in enumerate(lanes)]
rcs = [p.wait() for p in procs]
print("lane return codes:", rcs, flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
