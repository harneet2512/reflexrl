"""Kaggle T4: Health-Gathering perception probe for Qwen3-VL-8B (NF4 weights, fp32 compute).

Can the model see medkits well enough to teach navigation? Position and distance
questions, scored against the eval-only labels buffer.
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
   "accelerate bitsandbytes opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), repo, dirs_exist_ok=True)
os.chdir(repo)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
rc = sh(f"{sys.executable} -u scripts/hg_probe.py --model Qwen/Qwen3-VL-8B-Instruct --nf4 "
        "--frames 150 --perms 2 --out /kaggle/working/hg_probe")
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
