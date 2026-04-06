import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LlamaCPP.Visualizer",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LlamaCPPVisualizer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);

            // Default dimensions — overridden by viz_settings when the node executes
            this._vizWidth  = 800;
            this._vizHeight = 500;

            const iframe = document.createElement("iframe");
            iframe.style.cssText = `width:${this._vizWidth}px;height:${this._vizHeight}px;border:none;display:block;background:#fff;`;
            iframe.src = "about:blank";

            this.addDOMWidget("viz_iframe", "viz_iframe", iframe, {
                serialize: false,
                getValue() { return ""; },
                setValue() {},
                getMinHeight() { return this._vizHeight ?? 500; },
                getMaxHeight() { return this._vizHeight ?? 500; },
            });

            this._updateNodeSize();
            this._vizIframe = iframe;
        };

        nodeType.prototype._updateNodeSize = function() {
            const w = (this._vizWidth  ?? 800) + 24;
            const h = (this._vizHeight ?? 500) + 120;
            this.setSize([w, h]);
            if (this._vizIframe) {
                this._vizIframe.style.width  = `${this._vizWidth  ?? 800}px`;
                this._vizIframe.style.height = `${this._vizHeight ?? 500}px`;
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);

            // The server can pass viz_settings dimensions in the execution output
            // as message.viz_settings = { width, height }
            if (message?.viz_settings) {
                const { width, height } = message.viz_settings;
                if (width  && width  !== this._vizWidth)  this._vizWidth  = width;
                if (height && height !== this._vizHeight) this._vizHeight = height;
                this._updateNodeSize();
            }

            // Load the server-side page — real URL, real origin, CDN scripts work fine
            if (this._vizIframe) {
                this._vizIframe.src = `/llamacpp/viz_page/${this.id}?t=${Date.now()}`;
            }
        };
    },
});
