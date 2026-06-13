import { RISK_COLORS, CRITERIA_LABELS } from "../types";

interface JustificationCardProps {
  justification: string;
  riskLevel: "Low" | "Moderate" | "High" | "Critical";
  criteriaMet: string[];
}

export default function JustificationCard({
  justification,
  riskLevel,
  criteriaMet,
}: JustificationCardProps) {
  const colors = RISK_COLORS[riskLevel];

  // Highlight numeric values in the justification text
  const highlightedText = justification.replace(
    /(\d+\.?\d*)\s*(°C|mmHg|bpm|mmol\/L|×10³\/µL|%)/g,
    '<mark class="vital-highlight">$1 $2</mark>'
  );

  // Highlight risk level mentions
  const finalText = highlightedText.replace(
    /(Critical Risk|High Risk|Moderate Risk|Low Risk)/g,
    `<span class="risk-mention" style="color: ${colors.primary}; font-weight: 700;">$1</span>`
  );

  return (
    <div
      className="justification-card"
      style={{ borderLeftColor: colors.primary }}
    >
      <div className="justification-header">
        <span className="justification-icon">🧠</span>
        <h3>Clinical Justification</h3>
      </div>

      <p
        className="justification-text"
        dangerouslySetInnerHTML={{ __html: finalText }}
      />

      {criteriaMet.length > 0 && (
        <div className="criteria-pills">
          <span className="criteria-label">Sepsis-3 Criteria Met:</span>
          <div className="criteria-list">
            {criteriaMet.map((criterion) => (
              <span
                key={criterion}
                className="criteria-pill"
                style={{
                  backgroundColor: colors.bg,
                  borderColor: colors.border,
                  color: colors.primary,
                }}
              >
                🔴 {CRITERIA_LABELS[criterion] || criterion}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
