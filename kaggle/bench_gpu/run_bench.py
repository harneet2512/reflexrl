"""Kaggle T4 session: PPO throughput benchmark."""
import glob
import os
import shutil
import subprocess
import sys
import time

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
sh("nproc; free -g | head -2; python -c 'import torch;print(torch.__version__, torch.get_num_threads())'")
for n_envs in (8, 12):
    t0 = time.time()
    sh(f"{sys.executable} -u scripts/train.py --method ppo --scenario dtc --steps 60000 --seed 0 "
       f"--n-envs {n_envs} --device cuda --out /kaggle/working/bench_n{n_envs}")
    print(f"BENCH n_envs={n_envs}: {60000 / (time.time() - t0):.0f} steps/s incl. startup+evals", flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
