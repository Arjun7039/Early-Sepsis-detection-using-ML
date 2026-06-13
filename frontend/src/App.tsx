import { useState, useCallback } from "react";
import PatientForm from "./components/PatientForm";
import RiskGauge from "./components/RiskGauge";
import RiskBadge from "./components/RiskBadge";
import JustificationCard from "./components/JustificationCard";
import ShapChart from "./components/ShapChart";
import HistoryTable from "./components/HistoryTable";
import { predictSepsis } from "./api/client";
import type { PatientVitals, PredictionResponse, HistoryEntry } from "./types";

type View = "predictor" | "history";

export default function App() {
  const [view, setView] = useState<View>("predictor");
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [lastVitals, setLastVitals] = useState<PatientVitals | null>(null);

  const handlePredict = useCallback(async (vitals: PatientVitals) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await predictSepsis(vitals);
      setPrediction(result);
      setLastVitals(vitals);

      // Add to history (max 20)
      const entry: HistoryEntry = {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        vitals,
        prediction: result,
      };
      setHistory((prev) => [entry, ...prev].slice(0, 20));
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to connect to API";
      setError(
        `Prediction failed: ${message}. Make sure the backend is running.`
      );
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearHistory = () => setHistory([]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <span className="header-logo">🩺</span>
            <div>
              <h1>Early Sepsis Detection</h1>
              <p className="header-subtitle">
                AI-Powered Clinical Risk Assessment System
              </p>
            </div>
          </div>
          <nav className="header-nav">
            <button
              className={`nav-btn ${view === "predictor" ? "nav-btn-active" : ""}`}
              onClick={() => setView("predictor")}
            >
              🔬 Risk Predictor
            </button>
            <button
              className={`nav-btn ${view === "history" ? "nav-btn-active" : ""}`}
              onClick={() => setView("history")}
            >
              📋 History
              {history.length > 0 && (
                <span className="nav-badge">{history.length}</span>
              )}
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {view === "predictor" ? (
          <div className="predictor-layout">
            {/* Left Column — Input Form */}
            <div className="predictor-left">
              <PatientForm onSubmit={handlePredict} isLoading={isLoading} />
            </div>

            {/* Right Column — Results */}
            <div className="predictor-right">
              {error && (
                <div className="error-banner">
                  <span className="error-icon">⚠️</span>
                  <p>{error}</p>
                </div>
              )}

              {prediction ? (
                <div className="results-panel">
                  <div className="results-top-row">
                    <RiskGauge
                      riskScore={prediction.risk_score}
                      riskLevel={prediction.risk_level}
                    />
                    <RiskBadge
                      riskLevel={prediction.risk_level}
                      recommendation={prediction.recommendation}
                    />
                  </div>

                  <JustificationCard
                    justification={prediction.justification}
                    riskLevel={prediction.risk_level}
                    criteriaMet={prediction.criteria_met}
                  />

                  <ShapChart shapValues={prediction.shap_values} />
                </div>
              ) : (
                <div className="results-placeholder">
                  <div className="placeholder-icon">🔬</div>
                  <h3>Enter Patient Vitals</h3>
                  <p>
                    Submit patient data to receive a sepsis risk assessment with
                    AI-powered clinical justification.
                  </p>
                  <div className="placeholder-features">
                    <div className="feature-item">
                      <span>📊</span> SHAP Explainability
                    </div>
                    <div className="feature-item">
                      <span>🧠</span> Clinical Justification
                    </div>
                    <div className="feature-item">
                      <span>⚡</span> Real-time Scoring
                    </div>
                    <div className="feature-item">
                      <span>🏥</span> Sepsis-3 Criteria
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <HistoryTable history={history} onClear={clearHistory} />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Early Sepsis Detection System v1.0 — Ensemble ML (RF + XGBoost +
          LightGBM + GBM) with SHAP Explainability
        </p>
      </footer>
    </div>
  );
}
