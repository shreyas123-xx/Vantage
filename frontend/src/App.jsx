import { useEffect, useState } from "react";
import "./App.css";
import api from "./api/client";
import DriftPanel from "./components/DriftPanel";
import ImpactPanel from "./components/ImpactPanel";
import SchedulePanel from "./components/SchedulePanel";
import LongitudinalPanel from "./components/LongitudinalPanel";

const TABS = [
  { id: "drift", label: "Drift" },
  { id: "impact", label: "Impact" },
  { id: "schedule", label: "Schedule" },
  { id: "longitudinal", label: "Longitudinal" },
];

export default function App() {
  const [tab, setTab] = useState("drift");
  const [modelStatus, setModelStatus] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);
  const [analysis, setAnalysis] = useState(null); // { analysis_id, feature_vector, report... }

  useEffect(() => {
    api
      .health()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
    api
      .modelStatus()
      .then(setModelStatus)
      .catch(() => setModelStatus({ trained: false }));
  }, []);

  const statusTone = apiOnline === false ? "bad" : modelStatus && !modelStatus.trained ? "warn" : "ok";
  const statusText =
    apiOnline === false
      ? "API unreachable"
      : modelStatus === null
      ? "Checking backend\u2026"
      : modelStatus.trained
      ? "Impact model ready"
      : "Impact model not trained yet";

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title-block">
          <h1>Drift Impact Estimator</h1>
          <p>Measures dataset drift, predicts the performance hit it will cause, and recommends the smallest retraining schedule that keeps it in check.</p>
        </div>
        <span className="status-chip">
          <span className={`status-dot ${statusTone}`} />
          {statusText}
        </span>
      </header>

      <nav className="tab-rail">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "drift" && <DriftPanel analysis={analysis} onAnalysis={setAnalysis} />}
      {tab === "impact" && <ImpactPanel analysis={analysis} />}
      {tab === "schedule" && <SchedulePanel analysis={analysis} />}
      {tab === "longitudinal" && <LongitudinalPanel />}

      <footer className="footer-note">
        Drift metrics (PSI, KS test, Jensen-Shannon divergence, classifier two-sample test) are computed from live input
        distributions only. The impact model is trained on simulated production scenarios and predicts degradation from
        those drift signatures &mdash; treat it as decision support, not ground truth: pure concept drift, where the
        relationship between inputs and labels changes without the inputs themselves shifting, cannot be detected this
        way at all and needs a delayed-label feedback loop instead.
      </footer>
    </div>
  );
}
