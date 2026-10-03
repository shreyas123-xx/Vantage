import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL, timeout: 120000 });

export const api = {
  health: () => client.get("/api/health").then((r) => r.data),
  modelStatus: () => client.get("/api/model-status").then((r) => r.data),

  generateSynthetic: (payload) =>
    client.post("/api/datasets/synthetic", payload).then((r) => r.data),

  uploadDataset: (file, labelColumn) => {
    const form = new FormData();
    form.append("file", file);
    if (labelColumn) form.append("label_column", labelColumn);
    return client
      .post("/api/datasets/upload", form, { headers: { "Content-Type": "multipart/form-data" } })
      .then((r) => r.data);
  },

  analyzeDrift: (referenceId, currentId) =>
    client
      .post("/api/drift/analyze", { reference_id: referenceId, current_id: currentId })
      .then((r) => r.data),

  predictImpact: ({ analysisId, driftFeatureVector, metric }) =>
    client
      .post("/api/impact/predict", {
        analysis_id: analysisId,
        drift_feature_vector: driftFeatureVector,
        metric,
      })
      .then((r) => r.data),

  recommendSchedule: (payload) =>
    client.post("/api/schedule/recommend", payload).then((r) => r.data),

  runLongitudinal: (payload) =>
    client.post("/api/experiments/longitudinal", payload).then((r) => r.data),
};

export default api;
