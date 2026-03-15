import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LlamaCPP.Visualizer",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LlamaCPPVisualizer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);

            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:800px;height:500px;border:none;display:block;background:#fff;";
            iframe.src = "about:blank";

            this.addDOMWidget("viz_iframe", "viz_iframe", iframe, {
                serialize: false,
                getValue() { return ""; },
                setValue() {},
                getMinHeight() { return 500; },
                getMaxHeight() { return 500; },
            });

            this.setSize([824, 620]);
            this._vizIframe = iframe;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);
            // Load the server-side page — real URL, real origin, CDN scripts work fine
            if (this._vizIframe) {
                this._vizIframe.src = `/llamacpp/viz_page/${this.id}?t=${Date.now()}`;
            }
        };
    },
});