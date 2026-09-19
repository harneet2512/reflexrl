"""Kaggle job: Qwen3-VL-2B teacher gate + label collection on a T4 (Linux).

Installs pinned deps, copies the code from the private dataset, runs the
gate for all four scenarios (resumable per scenario, restarted on a crash),
and leaves results + labels in /kaggle/working/runs/phase0 for download.
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
root = os.path.dirname(os.path.dirname(src[0]))
shutil.copytree(root, "/kaggle/working/reflexrl_repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/reflexrl_repo")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
for attempt in range(4):
    rc = sh(f"{sys.executable} -u scripts/teacher_gate.py --policy qwen "
            "--scenarios dtc hg dc hgs --episodes 30 --n-envs 8 --out /kaggle/working/runs/phase0")
    print(f"gate attempt {attempt} rc={rc}", flush=True)
    if rc == 0:
        break
shutil.rmtree("/kaggle/working/reflexrl_repo", ignore_errors=True)
