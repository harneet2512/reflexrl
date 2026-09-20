"""Kaggle T4: GUIDED transfer to defend_the_line - the DTC Qwen+Jev teacher, reused as is.

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
for f in glob.glob("/kaggle/input/**/teachers/*/*.pt", recursive=True):
    shutil.copytree(os.path.dirname(os.path.dirname(f)), "runs/teachers", dirs_exist_ok=True)
# the DTC teacher is reused unchanged: same perception question, same Jev table
os.makedirs("runs/teachers/defend_the_line", exist_ok=True)
for f in glob.glob("runs/teachers/defend_the_center/*"):
    shutil.copy2(f, "runs/teachers/defend_the_line/")
os.environ["REFLEXRL_DEADLINE"] = str(START + 11 * 3600)
rc = sh(f"{sys.executable} -u scripts/run_queue.py experiments/configs/plan_heldout_guided.json --n-envs 12")
shutil.copytree(os.path.join(repo, "runs"), "/kaggle/working/runs", dirs_exist_ok=True)
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
