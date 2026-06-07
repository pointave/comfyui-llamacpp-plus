import { app } from "../../scripts/app.js";

// LlamaCPPVisualizerHTML is a settings/config node — it has no visual output
// of its own. This extension just ensures ComfyUI doesn't apply any stale
// widget behaviour from an old version of the node.
app.registerExtension({
    name: "LlamaCPP.VisualizerHTML",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LlamaCPPVisualizerHTML") return;
        // No custom widgets needed — width / height / js_delay_ms are plain INT
        // widgets registered automatically by ComfyUI from INPUT_TYPES.
    },
});
