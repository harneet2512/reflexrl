"""Kaggle job: pre-registered debias probe for the Qwen3-VL-2B teacher (T4, Linux)."""
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
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
rc = 0
for k in (1,):
    rc |= sh(f"{sys.executable} -u scripts/debias_probe.py --frames 150 --upscale {k} "
             "--out /kaggle/working/probe")
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
sys.exit(rc)
