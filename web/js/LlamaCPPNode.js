import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.LlamaCPPNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // ------------------------------------------------------------------
    // OpenAI TTS Connectivity — refresh & unload buttons
    // ------------------------------------------------------------------
    if (nodeData.name === "OpenAITTSConnectivity") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) originalNodeCreated.apply(this, arguments);

        const urlWidget   = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const fetchModels = async (url) => {
          const response = await fetch("/openai_tts/get_models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          });
          if (response.ok) {
            return await response.json();
          }
          throw new Error(`HTTP ${response.status}`);
        };

        const updateModels = async () => {
          refreshBtn.name = "⏳ Fetching...";
          this.setDirtyCanvas(true);
          const url = urlWidget.value;
          let models = [];
          try {
            models = await fetchModels(url);
          } catch (error) {
            console.error("[OpenAI TTS] Error fetching models:", error);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "OpenAI TTS connection error",
              detail: "Make sure the TTS server is running at " + url,
              life: 5000,
            });
            refreshBtn.name = "🔄 Refresh";
            this.setDirtyCanvas(true);
            return;
          }
          const prev = modelWidget.value;
          modelWidget.options.values = models;
          if (models.includes(prev)) {
            modelWidget.value = prev;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          }
          refreshBtn.name = "🔄 Refresh";
          this.setDirtyCanvas(true);
        };

        const unloadModel = async () => {
          const url   = urlWidget.value;
          const model = modelWidget.value;
          if (!url || !model) {
            app.extensionManager.toast.add({
              severity: "warn",
              summary: "OpenAI TTS",
              detail: "Set a URL and select a model before unloading.",
              life: 4000,
            });
            return;
          }
          unloadBtn.name = "⏳ Unloading...";
          this.setDirtyCanvas(true);
          try {
            const response = await fetch("/openai_tts/unload_model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ url, model }),
            });
            const data = await response.json();
            if (data.error) {
              app.extensionManager.toast.add({
                severity: "error",
                summary: "OpenAI TTS unload failed",
                detail: data.error,
                life: 5000,
              });
              unloadBtn.name = "✗ Failed";
            } else {
              app.extensionManager.toast.add({
                severity: "success",
                summary: "OpenAI TTS",
                detail: `Model unloaded (${data.method}).`,
                life: 3000,
              });
              unloadBtn.name = "✓ Unloaded";
            }
          } catch (err) {
            app.extensionManager.toast.add({
              severity: "error",
              summary: "OpenAI TTS unload error",
              detail: String(err),
              life: 5000,
            });
            unloadBtn.name = "✗ Failed";
          }
          this.setDirtyCanvas(true);
          setTimeout(() => {
            unloadBtn.name = "⏏️ Unload";
            this.setDirtyCanvas(true);
          }, 3000);
        };

        const reloadModel = async () => {
          const url = urlWidget.value;
          if (!url) {
            app.extensionManager.toast.add({
              severity: "warn",
              summary: "OpenAI TTS",
              detail: "Set a URL before reloading.",
              life: 4000,
            });
            return;
          }
          reloadBtn.name = "⏳ Reloading...";
          this.setDirtyCanvas(true);
          try {
            const response = await fetch("/openai_tts/reload_model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ url }),
            });
            const data = await response.json();
            if (data.error) {
              app.extensionManager.toast.add({
                severity: "error",
                summary: "OpenAI TTS reload failed",
                detail: data.error,
                life: 5000,
              });
              reloadBtn.name = "✗ Failed";
            } else {
              app.extensionManager.toast.add({
                severity: "success",
                summary: "OpenAI TTS",
                detail: `Model reloaded (${data.method}).`,
                life: 3000,
              });
              reloadBtn.name = "✓ Loaded";
            }
          } catch (err) {
            app.extensionManager.toast.add({
              severity: "error",
              summary: "OpenAI TTS reload error",
              detail: String(err),
              life: 5000,
            });
            reloadBtn.name = "✗ Failed";
          }
          this.setDirtyCanvas(true);
          setTimeout(() => {
            reloadBtn.name = "🔁 Reload";
            this.setDirtyCanvas(true);
          }, 3000);
        };

        const refreshBtn = this.addWidget("button", "🔄 Refresh", "", updateModels, { serialize: false });
        const unloadBtn  = this.addWidget("button", "⏏️ Unload", "", unloadModel, { serialize: false });
        const reloadBtn  = this.addWidget("button", "🔁 Reload", "", reloadModel, { serialize: false });

        urlWidget.callback = updateModels;
        await updateModels();
      };
    }

    // ------------------------------------------------------------------
    // LlamaCPP Connectivity — refresh & unload buttons
    // ------------------------------------------------------------------
    if (nodeData.name === "LlamaCPPConnectivity") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget   = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

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
        // Add buttons with callback as 4th argument (required by newer ComfyUI)
        // Signature: addWidget(type, name, value, callback, options)
        // ----------------------------------------------------------------
        const refreshButtonWidget = this.addWidget(
          "button",
          "🔄 Reconnect",
          "",
          updateModels,
          { serialize: false }
        );

        const unloadButtonWidget = this.addWidget(
          "button",
          "⏏️ Unload Model",
          "",
          unloadModel,
          { serialize: false }
        );

        // ----------------------------------------------------------------
        // Wire URL change → auto-refresh
        // ----------------------------------------------------------------
        urlWidget.callback = updateModels;

        // Initial model fetch on node creation
        await updateModels();
      };
    }
  },
});
