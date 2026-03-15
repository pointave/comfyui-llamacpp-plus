import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.LlamaCPPNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "LlamaCPPConnectivity") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget   = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        let refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");
        let unloadButtonWidget  = this.addWidget("button", "⏏️ Unload Model");

        // ----------------------------------------------------------------
        // Fetch model list from server
        // ----------------------------------------------------------------
        const fetchModels = async (url) => {
          const response = await fetch("/llamacpp/get_models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          });
          if (response.ok) {
            const models = await response.json();
            console.debug("[LlamaCPP] Fetched models:", models);
            return models;
          } else {
            throw new Error(`HTTP ${response.status}`);
          }
        };

        // ----------------------------------------------------------------
        // Reconnect — refresh model dropdown
        // ----------------------------------------------------------------
        const updateModels = async () => {
          refreshButtonWidget.name = "⏳ Fetching...";
          this.setDirtyCanvas(true);

          const url = urlWidget.value;
          let models = [];
          try {
            models = await fetchModels(url);
          } catch (error) {
            console.error("[LlamaCPP] Error fetching models:", error);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "LlamaCPP connection error",
              detail: "Make sure llama.cpp server is running at " + url,
              life: 5000,
            });
            refreshButtonWidget.name = "🔄 Reconnect";
            this.setDirtyCanvas(true);
            return;
          }

          const prevValue = modelWidget.value;
          modelWidget.options.values = models;
          if (models.includes(prevValue)) {
            modelWidget.value = prevValue;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          }

          refreshButtonWidget.name = "🔄 Reconnect";
          this.setDirtyCanvas(true);
          console.debug("[LlamaCPP] Model set to:", modelWidget.value);
        };

        // ----------------------------------------------------------------
        // Unload model from VRAM
        // ----------------------------------------------------------------
        const unloadModel = async () => {
          const url   = urlWidget.value;
          const model = modelWidget.value;

          if (!url || !model) {
            app.extensionManager.toast.add({
              severity: "warn",
              summary: "LlamaCPP",
              detail: "Set a URL and select a model before unloading.",
              life: 4000,
            });
            return;
          }

          unloadButtonWidget.name = "⏳ Unloading...";
          this.setDirtyCanvas(true);

          try {
            const response = await fetch("/llamacpp/unload_model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ url, model }),
            });
            const data = await response.json();

            if (data.error) {
              console.error("[LlamaCPP] Unload error:", data.error);
              app.extensionManager.toast.add({
                severity: "error",
                summary: "LlamaCPP unload failed",
                detail: data.error,
                life: 5000,
              });
              unloadButtonWidget.name = "✗ Unload failed";
            } else {
              console.log("[LlamaCPP] Unloaded via method:", data.method);
              app.extensionManager.toast.add({
                severity: "success",
                summary: "LlamaCPP",
                detail: `Model "${model}" unloaded from VRAM.`,
                life: 3000,
              });
              unloadButtonWidget.name = "✓ Unloaded";
            }
          } catch (err) {
            console.error("[LlamaCPP] Unload request failed:", err);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "LlamaCPP unload error",
              detail: String(err),
              life: 5000,
            });
            unloadButtonWidget.name = "✗ Unload failed";
          }

          this.setDirtyCanvas(true);
          setTimeout(() => {
            unloadButtonWidget.name = "⏏️ Unload Model";
            this.setDirtyCanvas(true);
          }, 3000);
        };

        // ----------------------------------------------------------------
        // Wire up callbacks
        // ----------------------------------------------------------------
        urlWidget.callback           = updateModels;
        refreshButtonWidget.callback = updateModels;
        unloadButtonWidget.callback  = unloadModel;

        refreshButtonWidget.serialize = false;
        unloadButtonWidget.serialize  = false;

        // Initial model fetch
        const dummy = async () => {};
        await dummy();
        await updateModels();
      };
    }
  },
});