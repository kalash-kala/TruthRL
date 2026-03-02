# Debug Summary: Ray GCS Startup Timeout (gRPC Conflict)

## Case: Ray Node Timed Out During Startup
**Date:** 2026-02-27
**Target Model:** Qwen2.5-VL-3B-Instruct
**Environment:** `truthrl-verl` Conda environment

### 1. Symptoms
The training script fails immediately at the `ray.init()` stage with the following error:
```text
Exception: The current node timed out during startup. 
This could happen because some of the raylet failed to startup or the GCS has become overloaded.
```
Inspecting the hidden logs in `/tmp/ray/session_latest/logs/dashboard_agent.log` reveals:
```text
ERROR agent.py:520 -- Agent is working abnormally. It will exit immediately.
grpc._cython.cygrpc._RequestCallError: Failed "grpc_server_request_call": None
```

### 2. Root Cause
**Dependency Conflict:** The environment had `grpcio==1.78.0` installed. 
Ray (version 2.54.0 and others) relies on `grpcio` for communication between the Global Control Service (GCS) and various agents. Versions of `grpcio >= 1.60.0` introduced breaking changes in the C-core layer that cause the Ray dashboard agent to crash silently during startup. This crash prevents the GCS from completing its "handshake," leading to a timeout.

### 3. Resolution
**Downgrade gRPC:**
Forced a downgrade of `grpcio` to a stable version compatible with Ray.
```bash
pip install "grpcio<1.60.0"
```

### 4. Verification
After downgrading to `grpcio 1.59.5`, Ray initialization (`ray.init()`) succeeded instantly on all allocated GPUs.

### 5. Prevention
Updated `requirements.txt` to pin `grpcio<1.60.0` to ensure future environment setup avoids this conflict.
