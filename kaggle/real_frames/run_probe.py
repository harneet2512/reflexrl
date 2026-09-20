"""Kaggle T4: the sees/decides pipeline on real Valorant frames with ground-truth boxes."""
import glob
import os
import shutil
import subprocess
import sys

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q transformers==5.17.0 accelerate bitsandbytes "
   "vizdoom==1.3.0 gymnasium==1.3.0 opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), repo, dirs_exist_ok=True)
os.chdir(repo)
root = "/kaggle/input"  # the mount name varies; the probe discovers the dataset
print("mounted:", os.listdir(root), flush=True)
for f in glob.glob(f"{root}/**/README*.txt", recursive=True):  # record the stated licence
    print(f"--- {f}\n{open(f).read()[:600]}", flush=True)
rc = sh(f"{sys.executable} -u scripts/real_frames_probe.py --root {root} --frames 150 --perms 2 "
        "--out /kaggle/working/real_frames")
os.chdir("/kaggle/working")
shutil.rmtree(repo, ignore_errors=True)
sys.exit(rc)
