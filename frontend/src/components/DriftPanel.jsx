import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import api from "../api/client";
import { SeverityBadge, LoadingStrip, ErrorBox, ArcGauge } from "./ui";

const severityColor = (s) => (s === "major" ? "var(--accent-red)" : s === "moderate" ? "var(--accent-amber)" : "var(--accent-teal)");

export default function DriftPanel({ analysis, onAnalysis }) {
  const [magnitude, setMagnitude] = useState(0.4);
  const [driftType, setDriftType] = useState("covariate");
  const [nReference, setNReference] = useState(1500);
  const [nCurrent, setNCurrent] = useState(800);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState(null);

  const report = analysis?.report;

  async function handleGenerateAndAnalyze() {
    setLoading(true);
    setError(null);
    try {
      const ds = await api.generateSynthetic({
        n_reference: nReference,
        n_current: nCurrent,
        magnitude,
        drift_type: driftType,
        seed,
      });
      setDatasetPreview(ds);
      const result = await api.analyzeDrift(ds.reference_id, ds.current_id);
      onAnalysis({
        analysis_id: result.analysis_id,
        feature_vector: result.drift_feature_vector,
        report: result,
      });
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const chartData = report?.features?.map((f) => ({ name: f.feature, psi: f.psi, severity: f.severity })) ?? [];

  return (
    <div className="grid-2">
      <div className="panel">
        <h2>Dataset</h2>
        <p className="panel-hint">
          Generate a synthetic production batch with a known amount of injected drift, or bring your own reference/current
          CSVs via the API&rsquo;s upload endpoint.
        </p>

        <div className="field">
          <label>
            Drift magnitude <span className="num">{magnitude.toFixed(2)}</span>
          </label>
          <input type="range" min="0" max="1" step="0.02" value={magnitude} onChange={(e) => setMagnitude(parseFloat(e.target.value))} />
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

        <div className="row">
          <div className="field" style={{ flex: 1, minWidth: 120 }}>
            <label>Reference rows</label>
            <input type="number" value={nReference} min={200} max={5000} onChange={(e) => setNReference(parseInt(e.target.value) || 0)} />
          </div>
          <div className="field" style={{ flex: 1, minWidth: 120 }}>
            <label>Current rows</label>
            <input type="number" value={nCurrent} min={200} max={5000} onChange={(e) => setNCurrent(parseInt(e.target.value) || 0)} />
          </div>
        </div>

        <button className="btn" onClick={handleGenerateAndAnalyze} disabled={loading} style={{ width: "100%" }}>
          {loading ? "Running\u2026" : "Generate & analyze"}
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
            Concept drift changes the label relationship without moving the input distribution &mdash; the drift metrics
            below will likely stay low even at high magnitude. That is expected, not a bug; see the Impact tab.
          </div>
        )}
      </div>

      <div className="stack">
        {!report && !loading && (
          <div className="panel">
            <div className="empty-state">Generate a dataset pair to see drift metrics here.</div>
          </div>
        )}

        {report && (
          <>
            <div className="panel raised row" style={{ justifyContent: "space-between" }}>
              <ArcGauge
                value={report.overall_drift_score}
                max={1.5}
                label="Overall drift score"
                sublabel={`${report.n_reference} reference rows vs ${report.n_current} current rows`}
                color={severityColor(report.severity)}
              />
              <div className="kv-list" style={{ minWidth: 220 }}>
                <div className="kv">
                  <span className="k">Severity</span>
                  <SeverityBadge severity={report.severity} />
                </div>
                <div className="kv">
                  <span className="k">C2ST AUC (multivariate)</span>
                  <span className="v">{report.c2st_auc.toFixed(3)}</span>
                </div>
                <div className="kv">
                  <span className="k">Drifted features</span>
                  <span className="v">
                    {report.n_drifted_features} / {report.n_features}
                  </span>
                </div>
              </div>
            </div>

            <div className="panel">
              <h2>Per-feature PSI</h2>
              <p className="panel-hint">Population Stability Index by feature. Above ~0.1 is moderate drift, above ~0.25 is major.</p>
              <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 30)}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" horizontal={false} />
                  <XAxis type="number" stroke="var(--ink-faint)" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="var(--ink-faint)" fontSize={11} width={70} />
                  <Tooltip contentStyle={{ background: "var(--panel-raised)", border: "1px solid var(--line)", fontSize: 12 }} />
                  <Bar dataKey="psi" radius={[0, 3, 3, 0]}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={severityColor(d.severity)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h2>Feature detail</h2>
              <div className="overflow-x">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Type</th>
                      <th className="num">PSI</th>
                      <th className="num">JS div.</th>
                      <th>Test</th>
                      <th className="num">p-value</th>
                      <th>Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.features.map((f) => (
                      <tr key={f.feature}>
                        <td>{f.feature}</td>
                        <td>{f.dtype}</td>
                        <td className="num">{f.psi.toFixed(3)}</td>
                        <td className="num">{f.js_divergence.toFixed(3)}</td>
                        <td>{f.test_name}</td>
                        <td className="num">{f.p_value?.toFixed(4)}</td>
                        <td>
                          <SeverityBadge severity={f.severity} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {datasetPreview && (
          <div className="panel">
            <h2>Current batch preview</h2>
            <p className="panel-hint">First rows of the generated current (production) batch.</p>
            <div className="overflow-x">
              <table className="data-table">
                <thead>
                  <tr>
                    {Object.keys(datasetPreview.current_preview[0] || {}).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetPreview.current_preview.slice(0, 8).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j} className={typeof v === "number" ? "num" : ""}>
                          {typeof v === "number" ? v.toFixed(3) : String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
