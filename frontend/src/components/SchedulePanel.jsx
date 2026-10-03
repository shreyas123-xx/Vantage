import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import api from "../api/client";
import { LoadingStrip, ErrorBox } from "./ui";

const DEFAULT_HISTORY = [
  { day: 0, drift_score: 0.03 },
  { day: 7, drift_score: 0.08 },
  { day: 14, drift_score: 0.16 },
  { day: 21, drift_score: 0.24 },
];

export default function SchedulePanel({ analysis }) {
  const [history, setHistory] = useState(DEFAULT_HISTORY);
  const [retrainCost, setRetrainCost] = useState(500);
  const [degradationCost, setDegradationCost] = useState(50);
  const [threshold, setThreshold] = useState(0.15); // clears the impact model's typical zero-drift floor (~0.12)
  const [horizon, setHorizon] = useState(60);
  const [metric, setMetric] = useState("accuracy");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function updateHistoryPoint(i, field, value) {
    setHistory((h) => h.map((p, idx) => (idx === i ? { ...p, [field]: parseFloat(value) || 0 } : p)));
  }
  function addPoint() {
    const last = history[history.length - 1];
    setHistory((h) => [...h, { day: last.day + 7, drift_score: last.drift_score }]);
  }
  function removePoint(i) {
    setHistory((h) => h.filter((_, idx) => idx !== i));
  }

  async function handleRecommend() {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        drift_history: history,
        analysis_id: analysis?.analysis_id,
        metric,
        retrain_cost: retrainCost,
        degradation_cost_per_point_per_day: degradationCost,
        horizon_days: horizon,
        degradation_threshold: threshold,
      };
      const res = await api.recommendSchedule(payload);
      setResult(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid-2">
      <div className="panel">
        <h2>Drift history &amp; costs</h2>
        <p className="panel-hint">
          {analysis
            ? "Using the drift signature from your last analysis as the reference scale."
            : "No analysis run yet \u2014 using a generic reference drift signature. Run one on the Drift tab for a calibrated result."}
        </p>

        <div className="field">
          <label>Observed drift score by day</label>
          <div className="stack" style={{ gap: 8 }}>
            {history.map((p, i) => (
              <div className="row" key={i} style={{ gap: 6, flexWrap: "nowrap" }}>
                <input type="number" value={p.day} onChange={(e) => updateHistoryPoint(i, "day", e.target.value)} style={{ width: 70 }} />
                <input
                  type="number"
                  step="0.01"
                  value={p.drift_score}
                  onChange={(e) => updateHistoryPoint(i, "drift_score", e.target.value)}
                  style={{ width: 90 }}
                />
                <button className="btn secondary" style={{ padding: "6px 10px" }} onClick={() => removePoint(i)}>
                  &times;
                </button>
              </div>
            ))}
          </div>
          <button className="btn secondary" style={{ marginTop: 8 }} onClick={addPoint}>
            + add point
          </button>
        </div>

        <div className="field">
          <label>
            Retrain cost <span className="num">${retrainCost}</span>
          </label>
          <input type="range" min="50" max="5000" step="50" value={retrainCost} onChange={(e) => setRetrainCost(parseFloat(e.target.value))} />
        </div>
        <div className="field">
          <label>
            Degradation cost / point / day <span className="num">${degradationCost}</span>
          </label>
          <input type="range" min="0" max="500" step="10" value={degradationCost} onChange={(e) => setDegradationCost(parseFloat(e.target.value))} />
        </div>
        <div className="field">
          <label>
            Degradation threshold <span className="num">{(threshold * 100).toFixed(1)}%</span>
          </label>
          <input type="range" min="0.005" max="0.2" step="0.005" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} />
        </div>
        <div className="field">
          <label>
            Horizon (days) <span className="num">{horizon}</span>
          </label>
          <input type="range" min="14" max="180" step="1" value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} />
        </div>

        <button className="btn" style={{ width: "100%" }} onClick={handleRecommend} disabled={loading}>
          {loading ? "Computing\u2026" : "Recommend schedule"}
        </button>
        <div style={{ marginTop: 10 }}>
          <LoadingStrip show={loading} />
        </div>
        {error && (
          <div style={{ marginTop: 10 }}>
            <ErrorBox message={error} />
          </div>
        )}
      </div>

      <div className="stack">
        {!result && !loading && (
          <div className="panel">
            <div className="empty-state">Set a drift history and costs, then recommend a schedule.</div>
          </div>
        )}

        {result?.floor_warning && (
          <div className="note-box">{result.floor_warning}</div>
        )}

        {result && (
          <>
            <div className="panel">
              <h2>Projected trajectory</h2>
              <p className="panel-hint">Forecast if nothing is retrained, extrapolated from your drift history (slope: {result.trend_slope_per_day}/day).</p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={result.projected_trajectory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                  <XAxis dataKey="day" stroke="var(--ink-faint)" fontSize={11} />
                  <YAxis stroke="var(--ink-faint)" fontSize={11} />
                  <Tooltip contentStyle={{ background: "var(--panel-raised)", border: "1px solid var(--line)", fontSize: 12 }} />
                  <ReferenceLine y={threshold} stroke="var(--accent-red)" strokeDasharray="4 4" label={{ value: "threshold", fontSize: 10, fill: "var(--accent-red)" }} />
                  <Line type="monotone" dataKey="projected_degradation" stroke="var(--accent-amber)" strokeWidth={2} dot={false} name="predicted degradation" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h2>Strategy comparison</h2>
              <p className="panel-hint">Total projected cost = retrains &times; retrain cost + degradation cost over the horizon.</p>
              <div className="stack">
                {Object.entries(result.strategy_costs).map(([key, s]) => {
                  const isRecommended =
                    result.recommended_strategy === key ||
                    (result.recommended_strategy.startsWith("fixed_every") && key === "best_fixed_periodic");
                  return (
                    <div key={key} className={`strategy-card ${isRecommended ? "recommended" : ""}`}>
                      <h3>
                        {key === "never_retrain" && "Never retrain"}
                        {key === "best_fixed_periodic" && `Fixed periodic (every ${s.interval_days} days)`}
                        {key === "adaptive_threshold" && "Adaptive (minimal necessary)"}
                        {isRecommended && <span style={{ color: "var(--accent-teal)", marginLeft: 8, fontWeight: 500 }}>&larr; recommended</span>}
                      </h3>
                      <div className="row" style={{ gap: 24 }}>
                        <span className="kv">
                          <span className="k">Total cost&nbsp;</span>
                          <span className="v">${s.total_cost.toLocaleString()}</span>
                        </span>
                        <span className="kv">
                          <span className="k">Retrains&nbsp;</span>
                          <span className="v">{s.n_retrains}</span>
                        </span>
                        {key === "adaptive_threshold" && s.trigger_after_days != null && (
                          <span className="kv">
                            <span className="k">First in&nbsp;</span>
                            <span className="v">{s.trigger_after_days}d</span>
                          </span>
                        )}
                        {key === "adaptive_threshold" && s.repeat_every_days != null && (
                          <span className="kv">
                            <span className="k">then every&nbsp;</span>
                            <span className="v">{s.repeat_every_days}d</span>
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
