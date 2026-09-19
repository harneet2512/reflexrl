"""Kaggle T4: fit the BC network / teacher proxy on the calibrated Qwen labels, per scenario."""
import glob
import os
import shutil
import subprocess
import sys

def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh(f"{sys.executable} -m pip install -q vizdoom==1.3.0 gymnasium==1.3.0 opencv-python-headless")
src = glob.glob("/kaggle/input/**/reflexrl/__init__.py", recursive=True)
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
labels = "/kaggle/working/all_labels"
os.makedirs(labels, exist_ok=True)
import numpy as np
# Both GPU lanes wrote labels for the same scenario: merge every shard, and
# offset episode ids per lane so the by-episode validation split stays clean.
lane_dirs = sorted(glob.glob("/kaggle/input/**/Qwen3-VL-8B-Instruct_pjev/labels/*", recursive=True))
print("lane label dirs:", lane_dirs, flush=True)
for lane_i, d in enumerate(lane_dirs):
    dst = os.path.join(labels, os.path.basename(d))
    os.makedirs(dst, exist_ok=True)
    for shard in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = dict(np.load(shard))
        z["episode"] = z["episode"] + 10_000 * lane_i
        np.savez_compressed(os.path.join(dst, f"shard_lane{lane_i}_{os.path.basename(shard)}"), **z)
print("label dirs:", sorted(os.listdir(labels)), flush=True)
rc = 0
for s in ("dtc",):
    rc |= sh(f"{sys.executable} -u scripts/build_teacher.py --scenario {s} --labels {labels} "
             "--out /kaggle/working/runs/teachers")
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
shutil.rmtree(labels, ignore_errors=True)
sys.exit(rc)
