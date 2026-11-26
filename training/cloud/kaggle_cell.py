# ===== Paste this whole cell into a Kaggle notebook (GPU T4, Internet ON), then Save Version -> Run All =====
import os, subprocess, sys, urllib.request

REPO = "https://github.com/AnshulMohanty/Agent_Yodha.git"
WORK = "/kaggle/working"
CODE = f"{WORK}/Agent_Yodha"
CKPT_URL = "https://github.com/AnshulMohanty/Agent_Yodha/releases/download/ckpt-A/latest.pt"

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

if not os.path.exists(CODE):
    sh(f"git clone --depth 1 {REPO} {CODE}")
sh("pip install -q uv")
sh("uv venv -q -p 3.13 /tmp/venv")
sh(f"uv pip install -q --python /tmp/venv/bin/python vizdoom==1.3.3 gymnasium numpy torch==2.15.0 pyyaml tensorboard -e {CODE}/env")
try:
    urllib.request.urlretrieve(CKPT_URL, f"{WORK}/ckpt_A.pt")
    print("downloaded Mac checkpoint ckpt-A")
    extra = ""
except Exception as e:
    print("no ckpt-A release yet, starting from scratch:", e)
    extra = "--set init_ckpt=null"
sh(f"cd {CODE} && PYTHONPATH=. /tmp/venv/bin/python -m training.algo.kickstart_ppo --config training/configs/kaggle.yaml {extra} 2>&1 | grep -v Warning")
sh(f"mkdir -p {WORK}/kaggle_export && cp {CODE}/runs/kaggle_b/export/* {WORK}/kaggle_export/ && cp {CODE}/runs/kaggle_b/latest.pt {WORK}/kaggle_export/")
print("DONE. Download kaggle_export/ from the Output tab.")
