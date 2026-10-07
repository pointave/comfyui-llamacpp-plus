from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ComfyUI looks for JS files in WEB_DIRECTORY
# Some versions expect web/js/, others just web/ — we put the file in both
WEB_DIRECTORY = "./web"

# ---------------------------------------------------------------------------
# Run LlamaCPPUnloadModel nodes first in every prompt
#
# Execution roots (is_output_node=True nodes) run in the order of the
# "outputs_to_execute" list, which ComfyUI builds from a set — arbitrary
# order. To make the unload node deterministically the first node to run,
# reorder that list when the prompt is queued: unload node ids move to the
# front. The node's execute() blocks until the model is actually freed, so
# by the time any other node (e.g. a diffusion model loader) starts, the LLM
# is already out of memory.
# ---------------------------------------------------------------------------

import execution as _execution

_UNLOAD_CLASS_TYPE = "LlamaCPPUnloadModel"


def _put_unload_first(self, item):
    try:
        number, prompt_id, prompt, extra_data, outputs = item[:5]
        if isinstance(outputs, (list, tuple)):
            unload_ids = [
                oid for oid in outputs
                if isinstance(prompt.get(oid), dict)
                and prompt[oid].get("class_type") == _UNLOAD_CLASS_TYPE
            ]
            if unload_ids:
                rest = [oid for oid in outputs if oid not in unload_ids]
                item = (number, prompt_id, prompt, extra_data,
                        unload_ids + rest, *item[5:])
    except Exception:
        pass
    return _put_unload_first.orig(self, item)


if not getattr(_execution.PromptQueue.put, "_unload_first_patched", False):
    _put_unload_first.orig = _execution.PromptQueue.put
    _put_unload_first._unload_first_patched = True
    _execution.PromptQueue.put = _put_unload_first

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
