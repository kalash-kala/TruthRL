
import subprocess
import os
import sys

def check_gpu(index):
    code = f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{index}"
import torch
try:
    if torch.cuda.is_available():
        print(f"Index {index}: SUCCESS - {{torch.cuda.get_device_name(0)}}")
    else:
        print(f"Index {index}: FAILED - torch.cuda.is_available() returned False")
except Exception as e:
    print(f"Index {index}: FAILED - {{e}}")
"""
    # Use the specific python environment
    # python_bin = "/root/anaconda3/envs/truthrl-verl/bin/python3"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error output for {index}:\n{result.stderr}")

print("Checking GPU accessibility individually...")
for i in range(8):
    check_gpu(i)
