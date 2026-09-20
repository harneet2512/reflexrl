"""Kaggle T4: final report on fresh, never-used episodes (seeds 7,000,000+)."""
import glob
import json
import os
import shutil
import subprocess
import sys

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
for f in glob.glob("/kaggle/input/**/teachers/*/*.pt", recursive=True):
    shutil.copytree(os.path.dirname(os.path.dirname(f)), "runs/teachers", dirs_exist_ok=True)
print("policies:", sorted(os.listdir("runs/train/defend_the_center")), flush=True)
sh(f"{sys.executable} -m pytest -q tests")
rc = sh(f"{sys.executable} -u scripts/final_eval.py --runs runs/train/defend_the_center "
        f"--episodes 50 --out /kaggle/working/results/final_eval.json")
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
