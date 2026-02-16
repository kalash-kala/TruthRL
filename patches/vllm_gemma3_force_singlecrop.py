import os
import importlib
import logging

log = logging.getLogger("vllm_gemma3_patch")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

ENV_FORCE = "VLLM_GEMMA3_FORCE_SINGLE_CROP"
ENV_FIX_PH = "VLLM_GEMMA3_FIX_PLACEHOLDERS"   # new


def _on(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def apply_patch() -> None:
    # Patch module
    mod_name = "vllm.model_executor.models.gemma3_mm"
    m = importlib.import_module(mod_name)

    try:
        import vllm
        log.info("[Gemma3Patch] vLLM version: %s", getattr(vllm, "__version__", "unknown"))
    except Exception:
        pass
    log.info("[Gemma3Patch] Module file: %s", getattr(m, "__file__", "unknown"))

    # ------------------------------------------------------------
    # (A) Crop patch (already working)
    # ------------------------------------------------------------
    if _on(ENV_FORCE):
        if hasattr(m, "Gemma3ProcessingInfo") and hasattr(m.Gemma3ProcessingInfo, "get_num_crops"):
            orig = m.Gemma3ProcessingInfo.get_num_crops

            def forced_get_num_crops(self, *args, **kwargs):
                return 0

            m.Gemma3ProcessingInfo.get_num_crops = forced_get_num_crops
            m.Gemma3ProcessingInfo._orig_get_num_crops = orig
            log.info("[Gemma3Patch] Patched Gemma3ProcessingInfo.get_num_crops -> 0 (single-crop).")
        else:
            log.warning("[Gemma3Patch] Could not patch get_num_crops (symbol not found).")

    # ------------------------------------------------------------
    # (B) Placeholder patch (this targets your actual failure)
    # ------------------------------------------------------------
    if _on(ENV_FIX_PH):
        if not hasattr(m, "find_mm_placeholders"):
            raise RuntimeError("[Gemma3Patch] find_mm_placeholders not found; vLLM internals changed.")

        orig_find = m.find_mm_placeholders

        def patched_find_mm_placeholders(*args, **kwargs):
            """
            Wrap vLLM's placeholder finder and force it to align with the
            actual <image_soft_token> positions.

            Goal: prevent 512 placeholders when prompt contains only 256 soft tokens.
            """
            out = orig_find(*args, **kwargs)

            # Try to access prompt token ids:
            # In vLLM, find_mm_placeholders is typically called with token ids as first arg.
            token_ids = None
            if len(args) >= 1 and isinstance(args[0], (list, tuple)):
                token_ids = args[0]

            # If we can't see token ids, fall back to original output.
            if token_ids is None:
                return out

            IMAGE_SOFT_TOKEN_ID = 262144  # from your debug print
            soft_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_SOFT_TOKEN_ID]

            # If soft tokens are 256 but out implies 512 placeholders, trim placeholders to soft_positions.
            # We handle a few possible "out" shapes (dataclass-like / dict-like).
            def _set_placeholders(obj):
                # common attribute names
                for name in ("placeholder_positions", "placeholder_indices", "positions", "indices"):
                    if hasattr(obj, name):
                        setattr(obj, name, soft_positions)
                        return True
                return False

            def _set_in_dict(d):
                for name in ("placeholder_positions", "placeholder_indices", "positions", "indices"):
                    if name in d:
                        d[name] = soft_positions
                        return True
                return False

            changed = False
            if isinstance(out, dict):
                changed = _set_in_dict(out)
            else:
                changed = _set_placeholders(out)

            if changed:
                log.info("[Gemma3Patch] Forced placeholders to %d <image_soft_token> positions (expected 256).",
                         len(soft_positions))
                return out

            # If structure unknown, we can still hard-guard by returning original out
            # (so we don't silently break). But log it for next patch refinement.
            log.warning("[Gemma3Patch] Could not rewrite placeholder structure (unknown type: %s).", type(out))
            return out

        m.find_mm_placeholders = patched_find_mm_placeholders
        m._orig_find_mm_placeholders = orig_find
        log.info("[Gemma3Patch] Patched gemma3_mm.find_mm_placeholders to align with <image_soft_token> positions.")


# Apply on import
apply_patch()