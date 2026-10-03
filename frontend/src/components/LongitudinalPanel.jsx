import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceDot,
} from "recharts";
import api from "../api/client";
import { LoadingStrip, ErrorBox } from "./ui";

export default function LongitudinalPanel() {
  const [nWeeks, setNWeeks] = useState(26);
  const [driftPattern, setDriftPattern] = useState("gradual");
  const [peakMagnitude, setPeakMagnitude] = useState(0.7);
  const [driftType, setDriftType] = useState("covariate");
  const [retrainStrategy, setRetrainStrategy] = useState("adaptive");
  const [threshold, setThreshold] = useState(0.15); // clears the impact model's typical zero-drift floor (~0.12)
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.runLongitudinal({
        n_weeks: nWeeks,
        drift_pattern: driftPattern,
        peak_magnitude: peakMagnitude,
        drift_type: driftType,
        retrain_strategy: retrainStrategy,
        degradation_threshold: threshold,
        metric: "accuracy",
      });
      setResult(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  const retrainWeeks = (result?.weeks ?? []).filter((w) => w.retrained);

  return (
    <div className="grid-2">
      <div className="panel">
        <h2>Simulation setup</h2>
        <p className="panel-hint">
          Deploys a real RandomForest classifier over {nWeeks} weeks of drifting production data and compares the impact
          model&rsquo;s predictions to the model&rsquo;s <em>true</em> measured performance each week.
        </p>

        <div className="field">
          <label>
            Weeks <span className="num">{nWeeks}</span>
          </label>
          <input type="range" min="6" max="52" value={nWeeks} onChange={(e) => setNWeeks(parseInt(e.target.value))} />
        </div>

        <div className="field">
          <label>Drift pattern</label>
          <div className="segmented">
            {["gradual", "sudden", "recurring"].map((p) => (
              <button key={p} className={driftPattern === p ? "active" : ""} onClick={() => setDriftPattern(p)}>
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>
            Peak magnitude <span className="num">{peakMagnitude.toFixed(2)}</span>
          </label>
          <input type="range" min="0.1" max="1" step="0.02" value={peakMagnitude} onChange={(e) => setPeakMagnitude(parseFloat(e.target.value))} />
        </div>

        <div className="field">
          <label>Drift mechanism</label>
          <div className="segmented">
            {["covariate", "concept", "mixed"].map((t) => (
              <button key={t} className={driftType === t ? "active" : ""} onClick={() => setDriftType(t)}>
                {t}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Retrain strategy</label>
          <div className="segmented">
            {["none", "fixed", "adaptive"].map((s) => (
              <button key={s} className={retrainStrategy === s ? "active" : ""} onClick={() => setRetrainStrategy(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {retrainStrategy === "adaptive" && (
          <div className="field">
            <label>
              Degradation threshold <span className="num">{(threshold * 100).toFixed(1)}%</span>
            </label>
            <input type="range" min="0.01" max="0.2" step="0.005" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} />
          </div>
        )}

        <button className="btn" style={{ width: "100%" }} onClick={handleRun} disabled={loading}>
          {loading ? "Simulating\u2026" : "Run experiment"}
        </button>
        <div style={{ marginTop: 10 }}>
          <LoadingStrip show={loading} />
        </div>
        {error && (
          <div style={{ marginTop: 10 }}>
            <ErrorBox message={error} />
          </div>
        )}

        {driftType === "concept" && (
          <div className="note-box" style={{ marginTop: 14 }}>
            With concept drift, expect predicted and true degradation to diverge &mdash; that&rsquo;s the honest, expected
            result, not a bug.
          </div>
        )}
      </div>

      <div className="stack">
        {!result && !loading && (
          <div className="panel">
            <div className="empty-state">Run a longitudinal experiment to validate predicted vs. true performance over time.</div>
          </div>
        )}

        {result && (
          <>
            <div className="panel raised">
              <h2>Predicted vs. true degradation</h2>
              <p className="panel-hint">
                Predicted comes from drift metrics alone (no labels). True comes from evaluating the live model on that
                week&rsquo;s labelled data &mdash; only possible here because this is a simulation with ground truth.
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={result.weeks}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                  <XAxis dataKey="week" stroke="var(--ink-faint)" fontSize={11} />
                  <YAxis stroke="var(--ink-faint)" fontSize={11} />
                  <Tooltip contentStyle={{ background: "var(--panel-raised)", border: "1px solid var(--line)", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="predicted_degradation" stroke="var(--accent-amber)" strokeWidth={2} dot={false} name="predicted" />
                  <Line type="monotone" dataKey="true_degradation" stroke="var(--ink)" strokeWidth={2} dot={false} name="true" />
                  {retrainWeeks.map((w) => (
                    <ReferenceDot key={w.week} x={w.week} y={w.true_degradation} r={4} fill="var(--accent-teal)" stroke="none" />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div className="hero-sub">
                <span className="legend-dot" style={{ background: "var(--accent-teal)" }} /> marks a retrain event ({result.summary.n_retrains} total)
              </div>
            </div>

            <div className="panel">
              <h2>Validation summary</h2>
              <div className="row" style={{ gap: 32 }}>
                <div className="kv-list">
                  <div className="kv">
                    <span className="k">MAE (predicted vs true)</span>
                    <span className="v">{result.summary.mae_predicted_vs_true}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Correlation</span>
                    <span className="v">{result.summary.correlation_predicted_vs_true ?? "n/a"}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Mean true performance</span>
                    <span className="v">{result.summary.mean_true_performance}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Final true performance</span>
                    <span className="v">{result.summary.final_true_performance}</span>
                  </div>
                </div>
              </div>
              <div className="note-box" style={{ marginTop: 14 }}>{result.summary.note}</div>
            </div>

            <div className="panel">
              <h2>Retraining strategy comparison</h2>
              <p className="panel-hint">Same drift trajectory, re-run under each strategy end to end.</p>
              <div className="overflow-x">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th className="num">Retrains</th>
                      <th className="num">Mean performance</th>
                      <th className="num">Min performance</th>
                      <th className="num">Final performance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.strategy_comparison).map(([name, s]) => (
                      <tr key={name}>
                        <td style={{ textTransform: "capitalize" }}>{name}</td>
                        <td className="num">{s.n_retrains}</td>
                        <td className="num">{s.mean_true_performance}</td>
                        <td className="num">{s.min_true_performance}</td>
                        <td className="num">{s.final_true_performance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
