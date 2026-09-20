"""Kaggle T4: held-out transfer to defend_the_line (lane a).

Scratch vs PPO fine-tuned vs the ReflexRL-trained policy fine-tuned, all
without any teacher (no DTL teacher exists), plus the random baseline.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

START = time.time()

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), repo, dirs_exist_ok=True)
os.chdir(repo)
os.makedirs("runs/train/defend_the_center", exist_ok=True)
for d in glob.glob("/kaggle/input/**/train/defend_the_center", recursive=True):
    shutil.copytree(d, "runs/train/defend_the_center", dirs_exist_ok=True)
print("DTC policies available:", sorted(os.listdir("runs/train/defend_the_center")), flush=True)
if "a" == "a":
    sh(f"{sys.executable} -u scripts/teacher_gate.py --policy random --scenarios dtl "
       "--episodes 30 --n-envs 8 --out /kaggle/working/runs/phase0")
os.environ["REFLEXRL_DEADLINE"] = str(START + 11 * 3600)
rc = sh(f"{sys.executable} -u scripts/run_queue.py experiments/configs/plan_heldout_a.json --n-envs 12")
shutil.copytree(os.path.join(repo, "runs"), "/kaggle/working/runs", dirs_exist_ok=True)
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
