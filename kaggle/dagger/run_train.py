"""Kaggle T4: ReflexRL with the live teacher in the loop (Qwen sees, Jev decides).

Every dagger-every steps the current student plays in a full-resolution env,
Qwen3-VL-8B is queried live on those states, Jev decides, and the perception
student is refit on the grown label set.
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

try:  # live Jev needs the key; without it Jev's cached decision table is used
    from kaggle_secrets import UserSecretsClient
    os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
    print("OpenRouter key loaded from Kaggle Secrets: Jev will be called live", flush=True)
except Exception as e:
    print(f"no Kaggle Secret ({type(e).__name__}); Jev decisions come from the cached table",
          flush=True)

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 transformers==5.17.0 "
   "accelerate bitsandbytes opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), repo, dirs_exist_ok=True)
os.chdir(repo)
os.makedirs("runs", exist_ok=True)
for f in glob.glob("/kaggle/input/**/teachers/*/*.pt", recursive=True):
    shutil.copytree(os.path.dirname(os.path.dirname(f)), "runs/teachers", dirs_exist_ok=True)
labels = "/kaggle/working/labels"
os.makedirs(labels, exist_ok=True)
import numpy as np
lane_dirs = sorted(glob.glob("/kaggle/input/**/Qwen3-VL-8B-Instruct_pjev/labels/*", recursive=True))
for lane_i, d in enumerate(lane_dirs):
    dst = os.path.join(labels, os.path.basename(d))
    os.makedirs(dst, exist_ok=True)
    for shard in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = dict(np.load(shard))
        z["episode"] = z["episode"] + 10_000 * lane_i
        np.savez_compressed(os.path.join(dst, f"lane{lane_i}_{os.path.basename(shard)}"), **z)
print("label dirs:", lane_dirs, flush=True)

os.environ["REFLEXRL_DEADLINE"] = str(START + 11 * 3600)
rc = 0
for seed in (0, 1):
    rc |= sh(f"{sys.executable} -u scripts/train.py --method reflexrl --scenario dtc "
             f"--steps 1500000 --seed {seed} --n-envs 12 --tag reflexrl_live --dagger "
             f"--labels {labels} --dagger-every 100000 --dagger-steps 300 --dagger-rounds 4")
shutil.copytree(os.path.join(repo, "runs"), "/kaggle/working/runs", dirs_exist_ok=True)
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
shutil.rmtree(labels, ignore_errors=True)
sys.exit(rc)
