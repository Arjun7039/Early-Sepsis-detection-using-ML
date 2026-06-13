import type { HistoryEntry } from "../types";
import { RISK_COLORS } from "../types";

interface HistoryTableProps {
  history: HistoryEntry[];
  onClear: () => void;
}

export default function HistoryTable({ history, onClear }: HistoryTableProps) {
  if (history.length === 0) {
    return (
      <div className="history-empty">
        <span className="history-empty-icon">📋</span>
        <p>No predictions yet. Submit patient vitals to get started.</p>
      </div>
    );
  }

  return (
    <div className="history-table-container">
      <div className="history-header">
        <h2>
          <span className="history-icon">📋</span> Prediction History
        </h2>
        <div className="history-actions">
          <span className="history-count">{history.length} predictions</span>
          <button className="btn-clear" onClick={onClear}>
            🗑️ Clear
          </button>
        </div>
      </div>

      <div className="table-scroll">
        <table className="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Temp (°C)</th>
              <th>HR (bpm)</th>
              <th>Lactate</th>
              <th>Ward</th>
              <th>Risk Score</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {history.map((entry) => {
              const colors = RISK_COLORS[entry.prediction.risk_level];
              const time = new Date(entry.timestamp).toLocaleTimeString();
              return (
                <tr key={entry.id}>
                  <td className="td-time">{time}</td>
                  <td>{entry.vitals.temperature_c.toFixed(1)}</td>
                  <td>{entry.vitals.heart_rate}</td>
                  <td>{entry.vitals.lactate.toFixed(1)}</td>
                  <td>
                    <span className="ward-chip">{entry.vitals.ward}</span>
                  </td>
                  <td>
                    <div className="score-bar-cell">
                      <div
                        className="score-bar-fill"
                        style={{
                          width: `${Math.round(entry.prediction.risk_score * 100)}%`,
                          backgroundColor: colors.primary,
                        }}
                      />
                      <span className="score-bar-text">
                        {Math.round(entry.prediction.risk_score * 100)}%
                      </span>
                    </div>
                  </td>
                  <td>
                    <span
                      className="risk-level-chip"
                      style={{
                        backgroundColor: colors.bg,
                        color: colors.primary,
                        borderColor: colors.border,
                      }}
                    >
                      {entry.prediction.risk_level}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
