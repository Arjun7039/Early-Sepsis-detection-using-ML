import { RISK_COLORS } from "../types";

interface RiskBadgeProps {
  riskLevel: "Low" | "Moderate" | "High" | "Critical";
  recommendation: string;
}

export default function RiskBadge({
  riskLevel,
  recommendation,
}: RiskBadgeProps) {
  const colors = RISK_COLORS[riskLevel];

  const RISK_ICONS = {
    Low: "✅",
    Moderate: "⚡",
    High: "🔶",
    Critical: "🚨",
  };

  return (
    <div
      className="risk-badge-card"
      style={{
        background: colors.bg,
        borderColor: colors.border,
      }}
    >
      <div className="risk-badge-header">
        <span className="risk-badge-icon">{RISK_ICONS[riskLevel]}</span>
        <span
          className="risk-badge-pill"
          style={{
            backgroundColor: colors.primary,
            color: "#fff",
          }}
        >
          {riskLevel} Risk
        </span>
      </div>
      <p className="risk-badge-recommendation">{recommendation}</p>
    </div>
  );
}
