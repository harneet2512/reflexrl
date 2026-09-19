"""Kaggle 2xT4: stage-1 small check for Qwen3-VL-4B (fp32) and Qwen3-VL-8B (fp16)."""
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
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
for model, dtype in (("Qwen/Qwen3-VL-4B-Instruct", "fp32"), ("Qwen/Qwen3-VL-8B-Instruct", "fp16")):
    sh(f"{sys.executable} -u scripts/teacher_size_check.py --model {model} --dtype {dtype} "
       "--out /kaggle/working/size_check")
    sh("rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct")  # free disk before 8B
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
