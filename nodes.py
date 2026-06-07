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
        "enable_thinking_budget",
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


def _sample_video_frames(images, max_frames: int = 60, frame_step: int = 1) -> list:
    """
    Sample frames from a video batch tensor.
    frame_step=1 â†’ every frame; frame_step=2 â†’ every 2nd frame, etc.
    Result is hard-capped at max_frame.
    """
    total = len(images)
    if total == 0:
        return []
    indices = list(range(0, total, max(1, frame_step)))
    if len(indices) > max_frames:
        factor = len(indices) / max_frames
        indices = [indices[int(i * factor)] for i in range(max_frames)]
    return [images[i] for i in indices]


def _audio_tensor_to_b64(waveform, sample_rate: int, max_seconds: float = 30.0) -> tuple[str, str]:
    import io, wave, struct
    import numpy as np
    import base64

    max_samples = int(max_seconds * sample_rate)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]
        print(f"[LlamaCPP] Audio trimmed to {max_seconds}s")

    # Convert to 16-bit PCM numpy â€” no torchaudio needed
    audio_np = waveform.cpu().numpy()                       # [channels, samples]
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]                  # ensure [channels, samples]
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767).astype(np.int16)               # float32 â†’ int16

    n_channels, n_samples = pcm.shape
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)                                   # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        # wave expects interleaved samples: [L0,R0,L1,R1,...] â†’ transpose then flatten
        wf.writeframes(pcm.T.flatten().tobytes())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return data, "audio/wav"


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
    Render HTML string to PIL Image via playwright (Chromium headless).
    Runs sync_playwright in a thread to avoid asyncio conflicts in ComfyUI.
    Returns (Image, err_string_or_None).
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
            # â”€â”€ Sampling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "enable_seed":            ("BOOLEAN", {"default": False}),
            "seed":                   ("INT",     {"default": seed, "min": 0,   "max": 2**31, "step": 1}),
            "enable_num_ctx":         ("BOOLEAN", {"default": False}),
            "num_ctx":                ("INT",     {"default": 2048, "min": 0,   "max": 2**31, "step": 1}),
            "enable_repeat_last_n":   ("BOOLEAN", {"default": False}),
            "repeat_last_n":          ("INT",     {"default": 64,   "min": -1,  "max": 64,    "step": 1}),
            "enable_repeat_penalty":  ("BOOLEAN", {"default": False}),
            "repeat_penalty":         ("FLOAT",   {"default": 1.1,  "min": 0,   "max": 2,     "step": 0.05}),
            "enable_temperature":     ("BOOLEAN", {"default": False}),
            "temperature":            ("FLOAT",   {"default": 0.8,  "min": -10, "max": 10,    "step": 0.05}),
            "enable_stop":            ("BOOLEAN", {"default": False}),
            "stop":                   ("STRING",  {"default": "",   "multiline": False}),
            "enable_top_k":           ("BOOLEAN", {"default": False}),
            "top_k":                  ("INT",     {"default": 40,   "min": 0,   "max": 100,   "step": 1}),
            "enable_top_p":           ("BOOLEAN", {"default": False}),
            "top_p":                  ("FLOAT",   {"default": 0.9,  "min": 0,   "max": 1,     "step": 0.05}),
            "enable_min_p":           ("BOOLEAN", {"default": False}),
            "min_p":                  ("FLOAT",   {"default": 0.0,  "min": 0,   "max": 1,     "step": 0.05}),
            "enable_main_gpu":        ("BOOLEAN", {"default": False}),
            "main_gpu":               ("INT",     {"default": 0,    "min": 0,   "max": 100,   "step": 1}),
            # â”€â”€ Thinking budget (Gemma4 / QwQ style) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "enable_thinking_budget": ("BOOLEAN", {"default": False,
                                        "tooltip": "Cap reasoning tokens (Gemma4, QwQ). 0 disables thinking entirely."}),
            "thinking_budget":        ("INT",     {"default": 1024, "min": 0,   "max": 32768, "step": 128}),
            # â”€â”€ Video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "video_frame_step":       ("INT",     {"default": 1,  "min": 1, "max": 60, "step": 1,
                                        "tooltip": "Sample every Nth source frame from the video input. "
                                                   "1 = every frame."}),
            "video_max_frames":       ("INT",     {"default": 60, "min": 1, "max": 60, "step": 1,
                                        "tooltip": "Hard cap on frames sent to model. "
                                                   "Video"}),
            # â”€â”€ Audio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "audio_max_seconds":      ("FLOAT",   {"default": 30.0, "min": 1.0, "max": 30.0, "step": 1.0,
                                        "tooltip": "Trim audio to this many seconds before sending. "
                                                   "Gemma4 E2B/E4B hard limit is 30 s."}),
            # â”€â”€ Debug â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "debug":                  ("BOOLEAN", {"default": False}),
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
    HTML render-settings node. Wire its output into LlamaCPPChat -> viz_settings.
    Controls the Playwright viewport used when LlamaCPPChat renders html_image.
    pip install playwright && playwright install chromium
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width":       ("INT", {"default": 800, "min": 64, "max": 4096, "step": 8}),
            "height":      ("INT", {"default": 600, "min": 64, "max": 4096, "step": 8}),
            "js_delay_ms": ("INT", {"default": 500, "min": 0,  "max": 5000, "step": 100,
                                    "tooltip": "ms to wait for JS to execute before screenshot"}),
        }}

    RETURN_TYPES = ("LLAMACPP_VIZ_SETTINGS",)
    RETURN_NAMES = ("viz_settings",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"

    def run(self, width: int = 800, height: int = 600, js_delay_ms: int = 500):
        return ({"width": width, "height": height, "js_delay_ms": js_delay_ms},)

class LlamaCPPChat:
    """
    Chat node for llama.cpp.  Supports text, images, video (frame batch), and audio.

    media_mode selects which optional socket is active:
      none  â€” text only; media inputs are ignored even if wired
      image â€” wire IMAGE for still frames
      video â€” wire IMAGE batch treated as temporal video; sampling in Options
      audio â€” wire AUDIO (Gemma4 E2B/E4B, 30 s max)

    visualization injects HTML generation instructions into the system prompt:
      disabled â€” system prompt used as-is
      html     â€” overrides system prompt to force a full HTML document output;
                 html_image carries a Playwright-rendered IMAGE of it

    html_render_width / html_render_height / html_render_delay_ms
      Control the viewport used when auto-rendering html_image.
      Right-click any of these â†’ Convert to Input to wire a Primitive.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system":               ("STRING",  {"multiline": True,  "default": "You are an AI assistant."}),
                "prompt":               ("STRING",  {"multiline": True,  "default": "Hello!"}),
                "think":                ("BOOLEAN", {"default": False}),
                "format":               (["text", "json"],),
                "reset_session":        ("BOOLEAN", {"default": True}),
                "media_mode":           (["none", "image", "video", "audio"], {"default": "none",
                                          "tooltip": "Which optional media socket to process. "
                                                     "none=text only (inputs ignored even if wired), "
                                                     "image=still frames, video=temporal batch, "
                                                     "audio=AUDIO input (Gemma4 E2B/E4B only)"}),
                "visualization":        (["disabled", "html"], {"default": "disabled",
                                          "tooltip": "html: forces full HTML document output and "
                                                     "renders the result to html_image via Playwright."}),
            },
            "optional": {
                "connectivity":         ("LLAMACPP_CONNECTIVITY", {"forceInput": False}),
                "options":              ("LLAMACPP_OPTIONS",      {"forceInput": False}),
                # â”€â”€ Media sockets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                "images":               ("IMAGE", {"forceInput": False,
                                          "tooltip": "Active when media_mode=image"}),
                "video":                ("IMAGE", {"forceInput": False,
                                          "tooltip": "Active when media_mode=video. "
                                                     "Batch treated as temporal frames. "
                                                     "Max 60 frames."}),
                "audio":                ("AUDIO", {"forceInput": False,
                                          "tooltip": "Active when media_mode=audio. "
                                                     "Gemma4 E2B/E4B only. Max 30 s."}),
                # ─── Visualization settings ──────────────────────────────────────
                "viz_settings":         ("LLAMACPP_VIZ_SETTINGS",
                                          {"tooltip": "Wire a LlamaCPP Visualizer HTML node to "
                                                       "control render width/height/delay. "
                                                       "Optional — defaults to 800x600."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # Outputs: result, thinking, html_image  (html_output removed)
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("result", "thinking", "html_image")
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"
    DESCRIPTION = (
        "Chat with llama.cpp. "
        "Use media_mode to select none / image / video / audio input. "
        "Set visualization=html to force HTML output and render it to html_image. "
        "Wire a LlamaCPP Visualizer HTML node to viz_settings to control render dimensions "
        "(optional, defaults to 800x600)."
    )

    async def run(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        reset_session: bool = True,
        media_mode: str = "none",
        visualization: str = "disabled",
        options: dict | None = None,
        connectivity: dict | None = None,
        images=None,
        video=None,
        audio=None,
        viz_settings: dict | None = None,
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

        # ── viz_settings: connecting this node enables html mode ──────────
        viz = viz_settings or {}
        # viz_settings provides render dimensions; visualization param controls mode
        html_render_width  = viz.get("width",       800)
        html_render_height = viz.get("height",      600)
        html_render_delay_ms = viz.get("js_delay_ms", 500)
        request_options = _filter_enabled_options(options)

        video_frame_step  = int(options.get("video_frame_step",  1))      if options else 1
        video_max_frames  = int(options.get("video_max_frames",  60))     if options else 60
        audio_max_seconds = float(options.get("audio_max_seconds", 30.0)) if options else 30.0

        loop = asyncio.get_event_loop()

        # â”€â”€ Media: only the branch matching media_mode is used â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        images_b64: list[str] | None = None
        video_b64:  list[str] | None = None
        audio_b64:  str | None = None

        if media_mode == "image" and images is not None:
            images_b64 = await loop.run_in_executor(None, _images_to_b64, images)
            print(f"[LlamaCPP] image mode: {len(images_b64)} frame(s)")

        elif media_mode == "video" and video is not None:
            def _proc_video(v, step, max_f):
                sampled = _sample_video_frames(v, max_frames=max_f, frame_step=step)
                print(f"[LlamaCPP] video mode: {len(v)} source â†’ {len(sampled)} frames")
                return _images_to_b64(sampled)
            video_b64 = await loop.run_in_executor(
                None, _proc_video, video, video_frame_step, video_max_frames
            )

        elif media_mode == "audio" and audio is not None:
            def _proc_audio(a, max_sec):
                waveform    = a["waveform"]
                sample_rate = a["sample_rate"]
                # ComfyUI waveform shape: [batch, channels, samples] â€” squeeze batch
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                return _audio_tensor_to_b64(waveform, sample_rate, max_sec)
            audio_b64, _ = await loop.run_in_executor(
                None, _proc_audio, audio, audio_max_seconds
            )
            print(f"[LlamaCPP] audio mode: encoded")

        # media_mode == "none" â†’ no media processing, text only

        # â”€â”€ Session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                f"\n\nOutput ONLY a complete HTML document. "
                f"Begin your response with <!DOCTYPE html> and end with </html>. "
                f"Ensure all content is visible within a {html_render_width}x{html_render_height} viewport. "
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

        # â”€â”€ Build multimodal content block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        has_multimodal = bool(images_b64 or video_b64 or audio_b64)
        if has_multimodal:
            content: list[dict] = [{"type": "text", "text": user_content}]

            if images_b64:
                for b in images_b64:
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b}"}})

            if video_b64:
                for b in video_b64:
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b}"}})

            if audio_b64:
                content.append({"type": "input_audio",
                                 "input_audio": {"data": audio_b64, "format": "wav"}})

            messages_for_api[-1]["content"] = content

        # â”€â”€ keep_alive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # reasoning_format=deepseek when think=True: server extracts <think> into
        # reasoning_content rather than leaving raw tags in content.
        # When think=False, reasoning_format=none so normal output is unaffected.
        payload["chat_template_kwargs"] = {"enable_thinking": think}
        payload["reasoning_format"]     = "deepseek" if think else "none"

        # json_object grammar conflicts with reasoning_format=deepseek when
        # thinking is enabled — skip response_format in that case.
        if format == "json":
            if think:
                print("[LlamaCPP] WARNING: json format skipped while think=True "
                      "(grammar conflicts with reasoning_format=deepseek)")
            else:
                payload["response_format"] = {"type": "json_object"}
        print(f"[LlamaCPP] think={think} format={format} reasoning_format=deepseek")

        if request_options:
            for src, dst in {
                "temperature":     "temperature",
                "top_p":           "top_p",
                "top_k":           "top_k",
                "repeat_penalty":  "frequency_penalty",
                "num_predict":     "max_tokens",
                "seed":            "seed",
                "stop":            "stop",
                "num_ctx":         "n_ctx",
                "main_gpu":        "main_gpu",
                "thinking_budget": "thinking_budget",
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
            return ("Error: empty response", "",
                    _blank_tensor(html_render_width, html_render_height))

        message       = choices[0].get("message", {})
        result_text   = message.get("content", "") or ""
        raw_thinking  = message.get("reasoning_content", "") or ""
        # reasoning_content is always extracted by the server when present.
        # When think=False, enable_thinking=False means the model won't produce
        # thinking tokens, so raw_thinking will naturally be empty.
        thinking_text = raw_thinking

        if think and not raw_thinking:
            print("[LlamaCPP] WARNING: think=True but reasoning_content empty. "
                  "Ensure llama-server started with --jinja")

        chat_session.messages.append({"role": "assistant", "content": result_text})

        # ── Extract and auto-render HTML ──────────────────────────────────────
        print(f"[LlamaCPP] DEBUG: visualization={repr(visualization)} result_text[:200]={repr(result_text[:200])}")
        html_tensor = _blank_tensor(html_render_width, html_render_height)
        if visualization == "html":
            print("[LlamaCPP] DEBUG: entering html branch")
            html_output = ""
            hm = re.search(r"<!DOCTYPE[\s\S]*?</html>", result_text, re.IGNORECASE)
            if not hm:
                hm = re.search(r"<html[\s\S]*?</html>", result_text, re.IGNORECASE)
            if hm:
                html_output = hm.group(0)
                print(f"[LlamaCPP] html extracted: {len(html_output)} chars")
            else:
                stripped = re.sub(r"```[a-zA-Z]*\n", "", result_text)
                stripped = re.sub(r"```", "", stripped).strip()
                if stripped.startswith("<!") or stripped.lower().startswith("<html>"):
                    html_output = stripped
                    print(f"[LlamaCPP] html extracted from stripped fences: {len(html_output)} chars")
                else:
                    print("[LlamaCPP] no HTML found in response")
                    print(f"[LlamaCPP] DEBUG: stripped[:300]={repr(stripped[:300])}")

            if html_output:
                print(f"[LlamaCPP] DEBUG: calling _render_html({html_render_width}x{html_render_height}, delay={html_render_delay_ms})")
                img, err = _render_html(html_output, html_render_width,
                                        html_render_height, html_render_delay_ms)
                if err:
                    print(f"[LlamaCPP] html_image render error: {err}")
                else:
                    html_tensor = _pil_to_tensor(img)
                    print(f"[LlamaCPP] html_image rendered: {img.size}")
        else:
            print("[LlamaCPP] DEBUG: visualization is NOT html, skipping render")

        if keep_alive_raw == 0:
            try:
                await loop.run_in_executor(None, lambda: _do_unload_sync(url, model))
            except Exception:
                pass

        return (result_text, thinking_text, html_tensor)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OpenAI TTS Nodes
# ---------------------------------------------------------------------------

import tempfile as _tempfile
import wave as _wave
import struct as _struct
import numpy as _np
import os as _os
from openai import OpenAI as _OpenAI


@PromptServer.instance.routes.post("/openai_tts/get_models")
async def openai_tts_get_models(request):
    data = await request.json()
    url = data.get("url", "").rstrip("/") + "/v1/models"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get, url)
        models = [m["id"] for m in result.get("data", [])]
        return web.json_response(models)
    except Exception as e:
        print(f"[OpenAI TTS] Error fetching models: {e}")
        return web.json_response([], status=500)


@PromptServer.instance.routes.post("/openai_tts/unload_model")
async def openai_tts_unload_model(request):
    data = await request.json()
    base_url = data.get("url", "").rstrip("/")
    model = data.get("model", "")
    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(
            None, lambda: _do_tts_unload_sync(base_url, model)
        )
        return web.json_response({"status": "ok", "method": msg})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/openai_tts/reload_model")
async def openai_tts_reload_model(request):
    data = await request.json()
    base_url = data.get("url", "").rstrip("/")
    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(
            None, lambda: _do_tts_reload_sync(base_url)
        )
        return web.json_response({"status": "ok", "method": msg})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/openai_tts/generate_speech")
async def openai_tts_generate(request):
    data = await request.json()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _do_tts_generate(
                data.get("base_url", ""),
                data.get("model", ""),
                data.get("voice", ""),
                data.get("text", ""),
                data.get("speed", 1.0),
            ),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def _do_tts_unload_sync(base_url: str, model: str) -> str:
    """Try multiple unload endpoints for TTS servers."""
    candidates = [
        (f"{base_url}/v1/models/unload", {}),              # OmniVoice
        (f"{base_url}/v1/audio/unload", {}),               # tts_webui_extension
        (f"{base_url}/v1/models/unload", {"model": model}),
        (f"{base_url}/models/unload", {}),
        (f"{base_url}/models/unload", {"model": model}),
        (f"{base_url}/api/v1/models/unload", {"identifier": model}),
    ]
    for unload_url, body in candidates:
        try:
            r = _HTTP.post(unload_url, json=body, timeout=10)
            if r.status_code in (200, 204):
                return f"unloaded via {unload_url}"
        except Exception:
            pass
    return "unload failed – no endpoint responded"


def _do_tts_reload_sync(base_url: str) -> str:
    """Try to reload the TTS model."""
    candidates = [
        (f"{base_url}/v1/models/reload", {}),              # OmniVoice
        (f"{base_url}/v1/audio/reload", {}),               # tts_webui_extension
    ]
    for reload_url, body in candidates:
        try:
            r = _HTTP.post(reload_url, json=body, timeout=60)
            if r.status_code in (200, 204):
                return f"reloaded via {reload_url}"
        except Exception:
            pass
    return "reload failed – no endpoint responded"


def _save_audio_tensor_to_wav(waveform, sample_rate: int) -> str:
    """Save a ComfyUI audio tensor to a temporary WAV file."""
    import torch
    audio_np = waveform.cpu().numpy()
    if audio_np.ndim == 3:
        audio_np = audio_np[0]  # squeeze batch dim
    if audio_np.ndim == 1:
        audio_np = audio_np[_np.newaxis, :]  # ensure [channels, samples]
    audio_np = _np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767).astype(_np.int16)
    n_channels, n_samples = pcm.shape
    fd, path = _tempfile.mkstemp(suffix=".wav")
    _os.close(fd)
    with _wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.T.flatten().tobytes())
    return path


def _do_tts_generate(base_url: str, model: str, voice: str, text: str, speed: float) -> dict:
    """Generate speech via an OpenAI-compatible TTS server and return WAV bytes."""
    client = _OpenAI(base_url=base_url, api_key="dummy")
    speed = float(speed)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        extra_body={"speed": speed},
    ) as response:
        fd, path = _tempfile.mkstemp(suffix=".wav")
        _os.close(fd)
        response.stream_to_file(path)
    with open(path, "rb") as f:
        wav_bytes = f.read()
    _os.unlink(path)
    return {"wav_bytes": wav_bytes}


class OpenAITTSConnectivity:
    """Connectivity node for OpenAI-compatible TTS servers (e.g., Omnivoice)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "url":   ("STRING", {"multiline": False, "default": "http://127.0.0.1:7778"}),
            "model": ((), {}),
        }}

    @classmethod
    def VALIDATE_INPUTS(s, url, model):
        return True

    RETURN_TYPES = ("OPENAI_TTS_CONN",)
    RETURN_NAMES = ("tts_connection",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"

    def run(self, url, model):
        return ({"url": url.rstrip("/"), "model": model},)


class OpenAITTSSpeech:
    """
    Text-to-speech node for OpenAI-compatible TTS servers (Omnivoice, etc.).

    voice_path  — absolute path to a reference audio file (.wav/.mp3/.ogg)
    voice_audio  — alternatively, wire an AUDIO node; it will be saved
                   to a temp file and used as the voice reference.
                   voice_audio takes priority over voice_path.
    speed       — playback speed multiplier (0.5 — 2.0)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text":       ("STRING", {"multiline": True, "default": "Hello, world!"}),
                "voice_path": ("STRING", {"multiline": False,
                                           "default": "",
                                           "tooltip": "Path to a reference audio file (.wav/.mp3/.ogg) for voice cloning."}),
                "speed":      ("FLOAT",  {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                           "tooltip": "Speech speed multiplier (0.5 = slow, 2.0 = fast)."}),
            },
            "optional": {
                "tts_connection": ("OPENAI_TTS_CONN", {"forceInput": False,
                                                       "tooltip": "Wire an OpenAI TTS Connectivity node."}),
                "voice_audio":    ("AUDIO", {"forceInput": False,
                                              "tooltip": "Alternatively, wire an AUDIO clip as the voice reference."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "LlamaCPP API"
    DESCRIPTION = (
        "Generate speech using an OpenAI-compatible TTS server (e.g., Omnivoice). "
        "Provide a voice reference via voice_path or by wiring an AUDIO input."
    )

    def run(self, text, voice_path, speed, tts_connection=None, voice_audio=None):
        if tts_connection is None:
            raise ValueError("Connect an OpenAI TTS Connectivity node.")

        url = tts_connection["url"].rstrip("/")
        model = tts_connection["model"]
        base_url = url + "/v1"

        # Resolve voice reference
        voice_ref = voice_path
        temp_voices = []

        if voice_audio is not None:
            # AUDIO input takes priority — save to temp WAV
            waveform = voice_audio["waveform"]
            sample_rate = voice_audio["sample_rate"]
            voice_ref = _save_audio_tensor_to_wav(waveform, sample_rate)
            temp_voices.append(voice_ref)
        elif not voice_ref or not _os.path.isfile(voice_ref):
            raise ValueError(
                "No voice reference provided. Either set voice_path to an audio file "
                "or wire an AUDIO input to voice_audio."
            )

        try:
            client = _OpenAI(base_url=base_url, api_key="dummy")
            speed_clamped = max(0.5, min(2.0, float(speed)))

            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice_ref,
                input=text,
                extra_body={"speed": speed_clamped},
            ) as response:
                fd, out_path = _tempfile.mkstemp(suffix=".wav")
                _os.close(fd)
                response.stream_to_file(out_path)

            # Read the WAV and convert to ComfyUI AUDIO tensor
            with _wave.open(out_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                samples = _struct.unpack(
                    f"<{n_frames * n_channels}h", raw
                )
                samples = _np.array(samples, dtype=_np.float32) / 32767.0
                samples = samples.reshape(n_channels, n_frames)

            _os.unlink(out_path)

            import torch
            waveform_tensor = torch.from_numpy(samples).unsqueeze(0)  # [1, ch, samples]
            return ({"waveform": waveform_tensor, "sample_rate": sample_rate},)

        finally:
            for v in temp_voices:
                try:
                    _os.unlink(v)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LlamaCPPOptions":        LlamaCPPOptions,
    "LlamaCPPConnectivity":   LlamaCPPConnectivity,
    "LlamaCPPChat":           LlamaCPPChat,
    "LlamaCPPVisualizerHTML": LlamaCPPVisualizerHTML,
    "OpenAITTSConnectivity":  OpenAITTSConnectivity,
    "OpenAITTSSpeech":        OpenAITTSSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPPOptions":        "LlamaCPP Options",
    "LlamaCPPConnectivity":   "LlamaCPP Connectivity",
    "LlamaCPPChat":           "LlamaCPP Chat",
    "LlamaCPPVisualizerHTML": "LlamaCPP Visualizer HTML",
    "OpenAITTSConnectivity":  "OpenAI TTS Connectivity",
    "OpenAITTSSpeech":        "OpenAI TTS Speech",
}
