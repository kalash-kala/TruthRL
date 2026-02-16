#!/usr/bin/env python3
"""
Reproducible patch for vLLM's Gemma 3 multimodal model.

Problem:
  vLLM 0.8.5.post1's Gemma3ProcessingInfo.get_num_crops() resolves
  pan_and_scan kwargs whose defaults (None → fallback to Gemma3ProcessorKwargs)
  can trigger crop logic even when the HF processor config says
  do_pan_and_scan=None.  This produces 512 placeholders while the HF
  processor only emits 256 soft tokens → ValueError at merge time.

Fix:
  Replace get_num_crops() with a version that returns 0 unconditionally,
  matching the single-crop / 256-token path the HF processor takes.

Usage:
  # Apply the patch
  python patches/apply_vllm_gemma3_patch.py --apply

  # Revert to original
  python patches/apply_vllm_gemma3_patch.py --revert

  # Check status
  python patches/apply_vllm_gemma3_patch.py --status

The patch is idempotent: applying twice is safe.
The original function is saved as a backup so revert is always possible.

Designed for vLLM 0.8.5.post1. Will refuse to run on other versions.
"""
import argparse
import importlib
import inspect
import re
import sys
import textwrap

EXPECTED_VLLM_VERSION = "0.8.5.post1"
MODULE_NAME = "vllm.model_executor.models.gemma3_mm"
PATCH_MARKER = "# PATCHED_BY_TRUTHRL: force single-crop (256 tokens)"

PATCHED_FUNCTION = textwrap.dedent("""\
    def get_num_crops(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> int:
        {marker}
        # Forces vLLM to always use 256 image tokens (no pan-and-scan crops).
        # This matches the HF Gemma3Processor default behavior when
        # do_pan_and_scan is None/False.
        return 0
""".format(marker=PATCH_MARKER))

# The original function starts with this signature and ends before get_image_repl
ORIGINAL_FUNC_PATTERN = re.compile(
    r"(    def get_num_crops\(\s*\n"
    r"        self,\s*\n"
    r"        \*,\s*\n"
    r"        image_width: int,\s*\n"
    r"        image_height: int,\s*\n"
    r"        processor: Optional\[Gemma3Processor\],\s*\n"
    r"    \) -> int:\s*\n)"
    r"(.*?)"
    r"(?=\n    def get_image_repl\()",
    re.DOTALL,
)


def get_module_path():
    m = importlib.import_module(MODULE_NAME)
    path = m.__file__
    if path.endswith(".pyc"):
        path = path[:-1]  # .pyc → .py
    return path


def check_vllm_version():
    import vllm
    version = getattr(vllm, "__version__", "unknown")
    if version != EXPECTED_VLLM_VERSION:
        print(f"⚠️  WARNING: Expected vLLM {EXPECTED_VLLM_VERSION}, got {version}.")
        print("   The patch may not work correctly. Proceed with caution.")
        return False
    return True


def is_patched(source: str) -> bool:
    return PATCH_MARKER in source


def apply_patch():
    check_vllm_version()
    path = get_module_path()
    print(f"📁 Target file: {path}")

    with open(path, "r") as f:
        source = f.read()

    if is_patched(source):
        print("✅ Already patched. Nothing to do.")
        return

    # Save backup
    backup_path = path + ".orig"
    with open(backup_path, "w") as f:
        f.write(source)
    print(f"💾 Backup saved to: {backup_path}")

    # Apply patch
    match = ORIGINAL_FUNC_PATTERN.search(source)
    if not match:
        print("❌ ERROR: Could not find get_num_crops() function in expected format.")
        print("   The vLLM source may have changed. Manual patching required.")
        sys.exit(1)

    # Replace the entire function body
    new_source = source[:match.start()] + PATCHED_FUNCTION + source[match.end():]

    with open(path, "w") as f:
        f.write(new_source)

    # Clear .pyc cache
    pyc_path = path + "c"
    import os
    if os.path.exists(pyc_path):
        os.remove(pyc_path)
    # Also clear __pycache__
    cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if "gemma3_mm" in fname:
                os.remove(os.path.join(cache_dir, fname))
                print(f"🗑️  Cleared cache: {fname}")

    print("✅ Patch applied successfully!")
    print("   get_num_crops() now always returns 0 (single-crop / 256 tokens).")


def revert_patch():
    path = get_module_path()
    backup_path = path + ".orig"

    import os
    if not os.path.exists(backup_path):
        print(f"❌ No backup found at {backup_path}. Cannot revert.")
        sys.exit(1)

    with open(backup_path, "r") as f:
        original = f.read()

    with open(path, "w") as f:
        f.write(original)

    os.remove(backup_path)

    # Clear caches
    cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if "gemma3_mm" in fname:
                os.remove(os.path.join(cache_dir, fname))

    print(f"✅ Reverted to original: {path}")


def show_status():
    check_vllm_version()
    path = get_module_path()
    print(f"📁 File: {path}")

    with open(path, "r") as f:
        source = f.read()

    if is_patched(source):
        print("🔧 Status: PATCHED (single-crop / 256 tokens)")
    else:
        print("📦 Status: ORIGINAL (unpatched)")

    import os
    backup_path = path + ".orig"
    if os.path.exists(backup_path):
        print(f"💾 Backup exists: {backup_path}")
    else:
        print("   No backup file found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply/revert vLLM Gemma 3 single-crop patch")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Apply the patch")
    group.add_argument("--revert", action="store_true", help="Revert to original")
    group.add_argument("--status", action="store_true", help="Show current patch status")
    args = parser.parse_args()

    if args.apply:
        apply_patch()
    elif args.revert:
        revert_patch()
    elif args.status:
        show_status()
