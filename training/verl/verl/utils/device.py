# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


# NOTE: Do NOT cache torch.cuda.is_available() at module level!
# Ray workers may import this module before CUDA_VISIBLE_DEVICES is properly set,
# causing a stale False value. We use a function that checks live instead.
def _check_cuda_available():
    import os
    available = torch.cuda.is_available()
    if not available:
        print(f"[DEBUG DEVICE] CUDA not available! PID={os.getpid()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    return available


# Keep module-level variables for backward compatibility with code that imports them,
# but make them properties via a wrapper that checks live.
is_npu_available = is_torch_npu_available()


class _CudaAvailableProxy:
    """A proxy object that behaves like a bool but checks torch.cuda.is_available() live."""
    def __bool__(self):
        return _check_cuda_available()
    def __repr__(self):
        return repr(bool(self))
    def __eq__(self, other):
        return bool(self) == other
    def __hash__(self):
        return hash(bool(self))


is_cuda_available = _CudaAvailableProxy()


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    return "CUDA_VISIBLE_DEVICES" if _check_cuda_available() else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if _check_cuda_available():
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if _check_cuda_available():
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")
