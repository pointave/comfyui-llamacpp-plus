# comfyui-llamacpp-plus

A set of nodes to connect llama.cpp models to ComfyUI. Supports text, image, video, and audio inputs (waiting for llama.cpp implementation), multi-turn chat, HTML/chart visualization, model unloading, and thinking mode.

> If you want a UI to update and load your llama.cpp server, check out [Togglellama](https://github.com/pointave/Togglellama)

---

## Nodes

- **LlamaCPP Connectivity** — Set your server URL, model, and keep-alive behavior
- **LlamaCPP Chat** — Main inference node. Supports multi-modal input, session memory, and HTML visualization
- **LlamaCPP Options** — Optional sampling overrides (temperature, top-k, seed, context length, etc.)
- **LlamaCPP Visualizer HTML** — Renders any HTML string to a ComfyUI IMAGE via Playwright

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
