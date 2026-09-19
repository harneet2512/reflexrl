"""Kaggle 2xT4: stage-1 small check for Qwen3-VL-8B, NF4 weights + fp32 compute, one T4."""
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
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} -u scripts/teacher_size_check.py "
   "--model Qwen/Qwen3-VL-8B-Instruct --dtype fp32 --nf4 --frames 150 --out /kaggle/working/size_check")
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
