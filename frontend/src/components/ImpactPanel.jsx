import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import api from "../api/client";
import { LoadingStrip, ErrorBox } from "./ui";

const METRICS = [
  { id: "accuracy", label: "Accuracy" },
  { id: "f1", label: "F1 score" },
  { id: "roc_auc", label: "ROC AUC" },
];

export default function ImpactPanel({ analysis }) {
  const [metric, setMetric] = useState("accuracy");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!analysis?.analysis_id) return;
    setLoading(true);
    setError(null);
    api
      .predictImpact({ analysisId: analysis.analysis_id, metric })
      .then(setResult)
      .catch((e) => setError(e?.response?.data?.detail || e.message))
      .finally(() => setLoading(false));
  }, [analysis?.analysis_id, metric]);

  if (!analysis) {
    return (
      <div className="panel">
        <div className="empty-state">Run a drift analysis on the Drift tab first &mdash; the impact model reads its drift signature.</div>
      </div>
    );
  }

  const deg = result?.predicted_degradation ?? 0;
  const tone = deg >= 0.08 ? "red" : deg >= 0.03 ? "amber" : "teal";
  const [lo, hi] = result?.confidence_interval ?? [0, 0];

  const importanceData = (result?.feature_importance ?? []).map((f) => ({ name: f.feature.replace(/_/g, " "), importance: f.importance }));

  return (
    <div className="stack">
      <div className="panel raised">
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h2 style={{ marginBottom: 10 }}>Predicted performance degradation</h2>
            <div className="hero-stat">
              <span className={`value ${tone}`}>{loading ? "\u2026" : `${(deg * 100).toFixed(1)}%`}</span>
              <span className="label">drop in {METRICS.find((m) => m.id === metric)?.label.toLowerCase()}</span>
            </div>
            <div className="hero-sub">
              90% interval: {(lo * 100).toFixed(1)}% &ndash; {(hi * 100).toFixed(1)}%
            </div>
          </div>
          <div className="field" style={{ minWidth: 160 }}>
            <label>Metric</label>
            <div className="segmented">
              {METRICS.map((m) => (
                <button key={m.id} className={metric === m.id ? "active" : ""} onClick={() => setMetric(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <LoadingStrip show={loading} />
        {error && (
          <div style={{ marginTop: 10 }}>
            <ErrorBox message={error} />
          </div>
        )}
      </div>

      {result && (
        <>
          <div className="panel">
            <h2>What&rsquo;s driving the prediction</h2>
            <p className="panel-hint">Feature importances from the trained impact-model regressor for this metric.</p>
            <ResponsiveContainer width="100%" height={Math.max(160, importanceData.length * 28)}>
              <BarChart data={importanceData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" horizontal={false} />
                <XAxis type="number" stroke="var(--ink-faint)" fontSize={11} />
                <YAxis type="category" dataKey="name" stroke="var(--ink-faint)" fontSize={11} width={110} />
                <Tooltip contentStyle={{ background: "var(--panel-raised)", border: "1px solid var(--line)", fontSize: 12 }} />
                <Bar dataKey="importance" fill="var(--accent-teal)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="panel">
            <h2>Model validation</h2>
            <p className="panel-hint">
              Measured on held-out simulated scenarios (real classifiers trained and evaluated under controlled drift) &mdash;
              not invented numbers.
            </p>
            <div className="row" style={{ gap: 28 }}>
              <div className="kv-list">
                <div className="kv">
                  <span className="k">R&sup2;</span>
                  <span className="v">{result.model_validation.r2}</span>
                </div>
                <div className="kv">
                  <span className="k">MAE</span>
                  <span className="v">{result.model_validation.mae}</span>
                </div>
                <div className="kv">
                  <span className="k">Test scenarios</span>
                  <span className="v">{result.model_validation.n_test}</span>
                </div>
              </div>
              <div className="kv-list">
                <div className="kv" style={{ fontSize: 12, color: "var(--ink-faint)" }}>
                  <span className="k">MAE by drift type</span>
                </div>
                {Object.entries(result.model_validation.mae_by_drift_type || {}).map(([k, v]) => (
                  <div className="kv" key={k}>
                    <span className="k">{k}</span>
                    <span className="v">{v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="note-box" style={{ marginTop: 14 }}>
              Concept-drift scenarios carry the least predictable degradation here on purpose: the drift metrics feeding
              this model only see input distributions, and pure concept drift doesn&rsquo;t move those. Wider confidence
              intervals at low drift scores are the model correctly expressing that uncertainty rather than a defect.
            </div>
          </div>
        </>
      )}
    </div>
  );
}
