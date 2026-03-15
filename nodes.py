from __future__ import annotations
import copy
import re
import random
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field
import asyncio
import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""

CHAT_SESSIONS: dict[str, ChatSession] = {}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_HTTP_ADAPTER = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=Retry(total=0))
_HTTP = _requests.Session()
_HTTP.mount("http://", _HTTP_ADAPTER)
_HTTP.mount("https://", _HTTP_ADAPTER)

def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    r = _HTTP.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _get(url: str, timeout: int = 10) -> dict:
    r = _HTTP.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Server routes
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.post("/llamacpp/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url", "").rstrip("/") + "/v1/models"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get, url)
        models = [m["id"] for m in result.get("data", [])]
        return web.json_response(models)
    except Exception:
        return web.json_response([], status=500)


@PromptServer.instance.routes.post("/llamacpp/unload_model")
async def unload_model_endpoint(request):
    data = await request.json()
    base_url = data.get("url", "").rstrip("/")
    model = data.get("model", "")
    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(None, lambda: _do_unload_sync(base_url, model))
        return web.json_response({"status": "ok", "method": msg})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_unload_sync(base_url: str, model: str) -> str:
    candidates = [
        (f"{base_url}/models/unload",        {}),
        (f"{base_url}/models/unload",        {"model": model}),
        (f"{base_url}/api/v1/models/unload", {"identifier": model}),
    ]
    for unload_url, body in candidates:
        try:
            r = _HTTP.post(unload_url, json=body, timeout=5)
            if r.status_code in (200, 204):
                return f"unloaded via {unload_url}"
        except Exception:
            pass
    return "unload failed"


def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not options:
        return None
    enablers = [
        "enable_mirostat", "enable_mirostat_eta", "enable_mirostat_tau",
        "enable_num_ctx", "enable_repeat_last_n", "enable_repeat_penalty",
        "enable_temperature", "enable_seed", "enable_stop", "enable_tfs_z",
        "enable_num_predict", "enable_top_k", "enable_top_p", "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None


def _images_to_b64(images: list) -> list[str]:
    import numpy as np
    result: list[str] = []
    for image in images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(i.clip(0, 255).astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _blank_tensor(width: int = 800, height: int = 600):
    import torch, numpy as np
    arr = np.full((height, width, 3), 30, dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _pil_to_tensor(img: Image.Image):
    import torch, numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _render_html(text: str, width: int = 800, height: int = 600, js_delay: int = 500):
    """
    Render HTML string to PIL Image via playwright (Chromium).
    Runs sync_playwright in a thread to avoid asyncio conflicts in ComfyUI.
    Returns (Image, err).
    pip install playwright && playwright install chromium
    """
    m = re.search(r"<!DOCTYPE[\s\S]*?</html>", text, re.IGNORECASE)
    if not m:
        m = re.search(r"<html[\s\S]*?</html>", text, re.IGNORECASE)
    if m:
        html = m.group(0)
    else:
        stripped = re.sub(r"```[a-zA-Z]*\n", "", text)
        stripped = re.sub(r"```", "", stripped).strip()
        html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<style>body{{margin:0;padding:0;}}</style></head>"
                f"<body>{stripped}</body></html>")

    def _do_render(html, width, height, js_delay):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="domcontentloaded")
            if js_delay > 0:
                page.wait_for_timeout(js_delay)
            png = page.screenshot(full_page=False)
            browser.close()
        return png

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            png = pool.submit(_do_render, html, width, height, js_delay).result(timeout=60)
        img = Image.open(BytesIO(png)).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img, None
    except ImportError:
        return None, "playwright not installed: pip install playwright && playwright install chromium"
    except Exception as e:
        return None, f"playwright error: {e}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class LlamaCPPOptions:
    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {"required": {
            "enable_seed":           ("BOOLEAN", {"default": False}),
            "seed":                  ("INT",     {"default": seed, "min": 0,   "max": 2**31, "step": 1}),
            "enable_num_ctx":        ("BOOLEAN", {"default": False}),
            "num_ctx":               ("INT",     {"default": 2048, "min": 0,   "max": 2**31, "step": 1}),
            "enable_repeat_last_n":  ("BOOLEAN", {"default": False}),
            "repeat_last_n":         ("INT",     {"default": 64,   "min": -1,  "max": 64,    "step": 1}),
            "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
            "repeat_penalty":        ("FLOAT",   {"default": 1.1,  "min": 0,   "max": 2,     "step": 0.05}),
            "enable_temperature":    ("BOOLEAN", {"default": False}),
            "temperature":           ("FLOAT",   {"default": 0.8,  "min": -10, "max": 10,    "step": 0.05}),
            "enable_stop":           ("BOOLEAN", {"default": False}),
            "stop":                  ("STRING",  {"default": "",   "multiline": False}),
            "enable_top_k":          ("BOOLEAN", {"default": False}),
            "top_k":                 ("INT",     {"default": 40,   "min": 0,   "max": 100,   "step": 1}),
            "enable_top_p":          ("BOOLEAN", {"default": False}),
            "top_p":                 ("FLOAT",   {"default": 0.9,  "min": 0,   "max": 1,     "step": 0.05}),
            "enable_min_p":          ("BOOLEAN", {"default": False}),
            "min_p":                 ("FLOAT",   {"default": 0.0,  "min": 0,   "max": 1,     "step": 0.05}),
            "enable_main_gpu":       ("BOOLEAN", {"default": False}),
            "main_gpu":              ("INT",     {"default": 0,    "min": 0,   "max": 100,   "step": 1}),
            "debug":                 ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("LLAMACPP_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"

    def run(self, **kargs):
        if kargs.get("debug"):
            pprint(kargs)
        return (kargs,)


class LlamaCPPConnectivity:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "url":             ("STRING", {"multiline": False, "default": "http://127.0.0.1:8081"}),
            "model":           ((), {}),
            "keep_alive":      ("INT",    {"default": 0, "min": -1, "max": 0, "step": 1}),
            "keep_alive_unit": (["minutes", "hours"],),
        }}

    @classmethod
    def VALIDATE_INPUTS(s, url, model, keep_alive, keep_alive_unit):
        return True

    RETURN_TYPES = ("LLAMACPP_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"

    def run(self, url, model, keep_alive, keep_alive_unit):
        return ({"url": url, "model": model,
                 "keep_alive": keep_alive, "keep_alive_unit": keep_alive_unit},)


class LlamaCPPVisualizerHTML:
    """
    Takes the html_output from LlamaCPPChat and renders it to a ComfyUI IMAGE.
    Connect image output to a Preview Image node to see the chart.
    pip install playwright && playwright install chromium
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "html":        ("STRING", {"forceInput": True}),
            "width":       ("INT",    {"default": 800, "min": 64, "max": 4096, "step": 8}),
            "height":      ("INT",    {"default": 600, "min": 64, "max": 4096, "step": 8}),
            "js_delay_ms": ("INT",    {"default": 500, "min": 0,  "max": 5000, "step": 100,
                                       "tooltip": "ms to wait for JS to execute before screenshot"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"

    def run(self, html: str, width: int = 800, height: int = 600, js_delay_ms: int = 500):
        if not html or not html.strip():
            print("[LlamaCPP HTML] empty input — nothing to render")
            return (_blank_tensor(width, height),)
        img, err = _render_html(html, width, height, js_delay_ms)
        if err:
            print(f"[LlamaCPP HTML] ERROR: {err}")
            return (_blank_tensor(width, height),)
        print(f"[LlamaCPP HTML] OK — rendered {img.size}")
        return (_pil_to_tensor(img),)


class LlamaCPPChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system":        ("STRING",  {"multiline": True, "default": "You are an AI assistant."}),
                "prompt":        ("STRING",  {"multiline": True, "default": "Hello!"}),
                "think":         ("BOOLEAN", {"default": False}),
                "format":        (["text", "json"],),
                "visualization": (["disabled", "html"], {"default": "disabled",
                    "tooltip": "html → injects HTML chart instructions, extract via html_output"}),
                "reset_session": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "connectivity":  ("LLAMACPP_CONNECTIVITY", {"forceInput": False}),
                "options":       ("LLAMACPP_OPTIONS",      {"forceInput": False}),
                "images":        ("IMAGE",                 {"forceInput": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result", "thinking", "html_output")
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"
    DESCRIPTION = (
        "Chat with llama.cpp. "
        "html_output extracts any HTML doc — wire to LlamaCPP Visualizer HTML then Preview Image."
    )

    async def run(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        visualization: str = "disabled",
        reset_session: bool = False,
        options: dict | None = None,
        connectivity: dict | None = None,
        images=None,
    ):
        if connectivity is None:
            raise ValueError("Connect a LlamaCPP Connectivity node.")

        conn            = connectivity
        url             = conn["url"].rstrip("/")
        model           = conn["model"]
        api_url         = f"{url}/v1/chat/completions"
        keep_alive_raw  = conn.get("keep_alive", 5)
        keep_alive_unit = conn.get("keep_alive_unit", "minutes")
        debug           = bool(options and options.get("debug", False))
        request_options = _filter_enabled_options(options)

        loop = asyncio.get_event_loop()

        images_b64: list[str] | None = None
        if images is not None:
            images_b64 = await loop.run_in_executor(None, _images_to_b64, images)

        # Session keyed on node's unique_id
        if reset_session:
            CHAT_SESSIONS.pop(unique_id, None)
        if unique_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[unique_id] = ChatSession()
        chat_session = CHAT_SESSIONS[unique_id]

        effective_system = system
        user_suffix = ""
        if visualization == "html":
            effective_system = (
                "You are a data visualization assistant. "
                "Your response must be a single complete HTML document and nothing else. "
                "No markdown, no code fences, no explanation, no prose. "
                "Use inline JavaScript for charts (Chart.js via CDN is fine). "
                "Start with <!DOCTYPE html> and end with </html>."
            )
            user_suffix = (
                "\n\nOutput ONLY a complete HTML document. "
                "Begin your response with <!DOCTYPE html> and end with </html>. "
                "No other text."
            )

        if effective_system:
            if chat_session.messages and chat_session.messages[0].get("role") == "system":
                chat_session.messages[0]["content"] = effective_system
            else:
                chat_session.messages.insert(0, {"role": "system", "content": effective_system})

        user_content = prompt + user_suffix if user_suffix else prompt
        chat_session.messages.append({"role": "user", "content": user_content})
        messages_for_api = copy.deepcopy(chat_session.messages)

        if images_b64:
            last = messages_for_api[-1]
            last["content"] = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
                for b in images_b64
            ]

        if keep_alive_raw == -1:
            keep_alive_val = -1
        elif keep_alive_raw == 0:
            keep_alive_val = 0
        else:
            suffix = "m" if keep_alive_unit == "minutes" else "h"
            keep_alive_val = f"{keep_alive_raw}{suffix}"

        payload: dict[str, Any] = {
            "model":      model,
            "messages":   messages_for_api,
            "keep_alive": keep_alive_val,
        }

        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        payload["chat_template_kwargs"] = {"enable_thinking": think}
        payload["reasoning_format"]     = "deepseek" if think else "none"
        print(f"[LlamaCPP] think={think} enable_thinking={think} reasoning_format={payload['reasoning_format']}")

        if request_options:
            for src, dst in {
                "temperature":    "temperature",
                "top_p":          "top_p",
                "top_k":          "top_k",
                "repeat_penalty": "frequency_penalty",
                "num_predict":    "max_tokens",
                "seed":           "seed",
                "stop":           "stop",
                "num_ctx":        "n_ctx",
                "main_gpu":       "main_gpu",
            }.items():
                if src in request_options:
                    payload[dst] = request_options[src]

        if debug:
            print(f"[LlamaCPP] POST {api_url}")
            pprint(payload)

        response_data = await loop.run_in_executor(None, lambda: _post(api_url, payload, 300))

        if debug:
            pprint(response_data)

        choices = response_data.get("choices", [])
        if not choices:
            return ("Error: empty response", "", "")

        message       = choices[0].get("message", {})
        result_text   = message.get("content", "") or ""
        raw_thinking  = message.get("reasoning_content", "") or ""
        thinking_text = raw_thinking if think else ""

        if think and not raw_thinking:
            print("[LlamaCPP] WARNING: think=True but reasoning_content empty. "
                  "Ensure llama-server started with --jinja")

        chat_session.messages.append({"role": "assistant", "content": result_text})

        # --- Extract HTML ---
        html_output = ""
        hm = re.search(r"<!DOCTYPE[\s\S]*?</html>", result_text, re.IGNORECASE)
        if not hm:
            hm = re.search(r"<html[\s\S]*?</html>", result_text, re.IGNORECASE)
        if hm:
            html_output = hm.group(0)
            print(f"[LlamaCPP] html_output: {len(html_output)} chars")
        else:
            stripped = re.sub(r"```[a-zA-Z]*\n", "", result_text)
            stripped = re.sub(r"```", "", stripped).strip()
            if stripped.startswith("<!") or stripped.lower().startswith("<html"):
                html_output = stripped
                print(f"[LlamaCPP] html_output from stripped fences: {len(html_output)} chars")
            else:
                print(f"[LlamaCPP] html_output: no HTML found in response")

        if keep_alive_raw == 0:
            try:
                await loop.run_in_executor(None, lambda: _do_unload_sync(url, model))
            except Exception:
                pass

        return (result_text, thinking_text, html_output)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LlamaCPPOptions":        LlamaCPPOptions,
    "LlamaCPPConnectivity":   LlamaCPPConnectivity,
    "LlamaCPPChat":           LlamaCPPChat,
    "LlamaCPPVisualizerHTML": LlamaCPPVisualizerHTML,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPPOptions":        "LlamaCPP Options",
    "LlamaCPPConnectivity":   "LlamaCPP Connectivity",
    "LlamaCPPChat":           "LlamaCPP Chat",
    "LlamaCPPVisualizerHTML": "LlamaCPP Visualizer HTML",
}