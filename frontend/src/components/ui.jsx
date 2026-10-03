export function SeverityBadge({ severity }) {
  const label = severity === "major" ? "Major drift" : severity === "moderate" ? "Moderate drift" : "No significant drift";
  return <span className={`severity-label ${severity}`}>{label}</span>;
}

export function LoadingStrip({ show }) {
  if (!show) return null;
  return <div className="loading-strip" />;
}

export function ErrorBox({ message }) {
  if (!message) return null;
  return <div className="error-box">{message}</div>;
}

/** Simple arc gauge, 0 -> max, drawn as an SVG stroke-dasharray arc. */
export function ArcGauge({ value, max = 1.5, label, sublabel, color = "var(--ink)" }) {
  const clamped = Math.max(0, Math.min(value, max));
  const pct = clamped / max;
  const radius = 74;
  const circumference = Math.PI * radius; // half circle
  const dash = circumference * pct;

  return (
    <div style={{ textAlign: "center" }}>
      <svg width="200" height="120" viewBox="0 0 200 120">
        <path
          d="M 20 110 A 80 80 0 0 1 180 110"
          fill="none"
          stroke="var(--line)"
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M 20 110 A 80 80 0 0 1 180 110"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          style={{ transition: "stroke-dasharray 0.4s ease" }}
        />
        <text x="100" y="92" textAnchor="middle" fontSize="30" fontFamily="var(--font-display)" fill="var(--ink)">
          {value.toFixed(2)}
        </text>
      </svg>
      {label && <div style={{ fontSize: 13, color: "var(--ink-soft)", marginTop: -6 }}>{label}</div>}
      {sublabel && <div style={{ fontSize: 12, color: "var(--ink-faint)", marginTop: 2 }}>{sublabel}</div>}
    </div>
  );
}
