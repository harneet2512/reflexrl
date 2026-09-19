"""Kaggle T4: deployment benchmark + real-time eval + demo video for the trained reflex policy."""
import glob
import json
import os
import shutil
import subprocess
import sys

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 transformers==5.17.0 "
   "accelerate opencv-python-headless matplotlib")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), repo, dirs_exist_ok=True)
os.chdir(repo)
runs = [p for p in glob.glob("/kaggle/input/**/train/defend_the_center", recursive=True)]
assert runs, "trained runs not mounted"
shutil.copytree(runs[0], "runs/train/defend_the_center", dirs_exist_ok=True)
done = [(json.load(open(f"{r}/done.json"))["final_eval"], r)
        for r in glob.glob("runs/train/defend_the_center/ppo_s*") if os.path.exists(f"{r}/done.json")]
best = max(done)[1]
print("best run:", best, done, flush=True)
out = "/kaggle/working/results"
sh(f"{sys.executable} -u scripts/benchmark_deploy.py --scenario dtc --ckpt {best}/ckpt_final.pt --out {out}/benchmark")
sh(f"{sys.executable} -u scripts/make_demo.py --runs runs/train/defend_the_center --episodes 10 --out {out}/demo")
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
