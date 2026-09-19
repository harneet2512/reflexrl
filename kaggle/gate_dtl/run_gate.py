"""Kaggle T4: held-out gate on defend_the_line (random baseline + fp32 calibrated Qwen-2B)."""
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
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
out = "/kaggle/working/runs/phase0"
rc = sh(f"{sys.executable} -u scripts/teacher_gate.py --policy random --scenarios dtl --episodes 30 --n-envs 8 --out {out}")
for attempt in range(4):
    rc = sh(f"{sys.executable} -u scripts/teacher_gate.py --policy qwen --variant calibrated "
            f"--scenarios dtl --episodes 30 --n-envs 8 --out {out}")
    if rc == 0:
        break
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
sys.exit(rc)
