"""Kaggle 2xT4: DTC gate for the Qwen3-VL-8B-sees / Jev-decides teacher, 15+15 episodes."""
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
sh("huggingface-cli download Qwen/Qwen3-VL-8B-Instruct > /dev/null 2>&1")  # one download for both lanes

def lane(i):
    gate = (f"{sys.executable} -u scripts/teacher_gate.py --policy qwen --model Qwen/Qwen3-VL-8B-Instruct "
            f"--nf4 --variant perception_jev --scenarios dtc --episodes 15 --n-envs 8 "
            f"--seed-offset {100 * i} --out /kaggle/working/runs/phase0/lane{i}")
    return subprocess.Popen(f"export CUDA_VISIBLE_DEVICES={i}; for a in 1 2 3; do {gate} "
                            f">> /kaggle/working/lane{i}.log 2>&1 && break; done", shell=True)

procs = [lane(0), lane(1)]
print("lane rcs:", [p.wait() for p in procs], flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
