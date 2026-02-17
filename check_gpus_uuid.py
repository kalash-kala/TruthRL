
import subprocess
import os
import re

def get_gpu_uuids():
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
        # Extract UUIDs like GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        uuids = re.findall(r'UUID: (GPU-[a-z0-9-]+)', result.stdout)
        return uuids
    except Exception as e:
        print(f"Error getting UUIDs: {e}")
        return []

def check_gpu_via_uuid(uuid):
    code = f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{uuid}"
import torch
try:
    if torch.cuda.is_available():
        print(f"UUID {uuid}: SUCCESS - {{torch.cuda.get_device_name(0)}}")
    else:
        print(f"UUID {uuid}: FAILED - torch.cuda.is_available() returned False")
except Exception as e:
    print(f"UUID {uuid}: FAILED - {{e}}")
"""
    python_bin = "/root/anaconda3/envs/truthrl-verl/bin/python3"
    result = subprocess.run([python_bin, "-c", code], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.stderr:
        # Check if there are common CUDA initialization errors in stderr
        if "CUDA error" in result.stderr or "driver" in result.stderr:
             print(f"Stderr for {uuid}:\n{result.stderr.strip()}")

uuids = get_gpu_uuids()
if not uuids:
    print("No GPUs found via nvidia-smi -L")
else:
    print(f"Checking {len(uuids)} GPUs via UUID addressing...")
    for uuid in uuids:
        check_gpu_via_uuid(uuid)
