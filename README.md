# comfyui-llamacpp-plus

A set of nodes to connect llama.cpp models to ComfyUI. Supports text, image, video, and audio inputs, multi-turn chat, HTML/chart visualization, model unloading, and reasoning mode.

> If you want a UI to update and load your llama.cpp server and are on Windows, check out [Togglellama](https://github.com/pointave/Togglellama) to have a taskbar system tray icon that unloads model and has 
8 custom flag presets.
---

## Nodes

- **LlamaCPP Connectivity** — Set your server URL, model, and keep-alive behavior
- **LlamaCPP Chat** — Main inference node. Supports multi-modal input, session memory, and HTML visualization
- **LlamaCPP Options** — Optional sampling overrides (temperature, top-k, seed, context length, etc.)
- **LlamaCPP Visualizer HTML** — Renders any HTML string to a ComfyUI IMAGE via Playwright
- **OpenAI TTS Connectivity** -- Refresh will populate dropdown with model deployed
- **OpenAI TTS Speech** -- Outputs audio using either reference from audio input or filepath

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/comfyui-llamacpp-plus
```

For HTML visualization:
```bash
pip install playwright && playwright install chromium
```

---

## Notes

- Set `keep_alive=0` to unload the model from VRAM after each run
- `media_mode=none` lets you leave inputs wired without sending them

---

## License

MIT
