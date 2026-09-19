"""Kaggle T4 pilot: ReflexRL + BC->PPO with the Qwen-8B-sees / Jev-decides teacher: resumable queue with an 11 h session deadline.

Previous sessions' runs (checkpoints, resume.pt, metrics) come in through the
private dataset harneetb/reflexrl-runs-pilot when it exists; finished runs are
skipped and paused ones resume. Everything under runs/ is saved as output.
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
os.makedirs("runs", exist_ok=True)
t = glob.glob("/kaggle/input/**/teachers/*/proxy.pt", recursive=True)
assert t, "teacher proxies not mounted"
shutil.copytree(os.path.dirname(os.path.dirname(t[0])), "runs/teachers", dirs_exist_ok=True)
prev = [p for p in glob.glob("/kaggle/input/**/train", recursive=True) if "reflexrl-runs" in p]
if prev:
    shutil.copytree(prev[0], "runs/train", dirs_exist_ok=True)
    print("resumed from", prev[0], flush=True)
os.environ["REFLEXRL_DEADLINE"] = str(START + 11 * 3600)
rc = sh(f"{sys.executable} -u scripts/run_queue.py experiments/configs/plan_kaggle_pilot.json --n-envs 12")
shutil.move(os.path.join(repo, "runs"), "/kaggle/working/runs")
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
