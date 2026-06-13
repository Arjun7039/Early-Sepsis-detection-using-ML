/**
 * TypeScript interfaces for the Sepsis Detection System
 */

export interface PatientVitals {
  temperature_c: number | "";
  bp_systolic: number | "";
  bp_diastolic: number | "";
  heart_rate: number | "";
  wbc_count: number | "";
  lactate: number | "";
  ward: "ICU-A" | "ICU-B" | "Ward-X" | "ER" | "";
}

export interface PredictionResponse {
  risk_score: number;
  risk_level: "Low" | "Moderate" | "High" | "Critical";
  sepsis_predicted: boolean;
  shap_values: Record<string, number>;
  recommendation: string;
  justification: string;
  criteria_met: string[];
  criteria_count: number;
}

export interface HistoryEntry {
  id: string;
  timestamp: Date;
  vitals: PatientVitals;
  prediction: PredictionResponse;
}

export const WARD_OPTIONS: ("ICU-A" | "ICU-B" | "Ward-X" | "ER")[] = [
  "ICU-A",
  "ICU-B",
  "Ward-X",
  "ER",
];

export const VITALS_RANGES = {
  temperature_c: { min: 33.0, max: 43.0, step: 0.1, label: "Temperature (°C)", unit: "°C" },
  bp_systolic: { min: 50, max: 200, step: 1, label: "Systolic BP (mmHg)", unit: "mmHg" },
  bp_diastolic: { min: 30, max: 120, step: 1, label: "Diastolic BP (mmHg)", unit: "mmHg" },
  heart_rate: { min: 20, max: 200, step: 1, label: "Heart Rate (bpm)", unit: "bpm" },
  wbc_count: { min: 0.0, max: 20.0, step: 0.1, label: "WBC Count (×10³/µL)", unit: "×10³/µL" },
  lactate: { min: 0.0, max: 10.0, step: 0.1, label: "Lactate (mmol/L)", unit: "mmol/L" },
} as const;

export const RISK_COLORS = {
  Low: { primary: "#10b981", bg: "rgba(16, 185, 129, 0.1)", border: "rgba(16, 185, 129, 0.3)" },
  Moderate: { primary: "#f59e0b", bg: "rgba(245, 158, 11, 0.1)", border: "rgba(245, 158, 11, 0.3)" },
  High: { primary: "#f97316", bg: "rgba(249, 115, 22, 0.1)", border: "rgba(249, 115, 22, 0.3)" },
  Critical: { primary: "#ef4444", bg: "rgba(239, 68, 68, 0.1)", border: "rgba(239, 68, 68, 0.3)" },
} as const;

export const CRITERIA_LABELS: Record<string, string> = {
  fever_hypothermia: "Abnormal Temperature",
  tachycardia: "Tachycardia",
  wbc_abnormal: "Abnormal WBC",
  high_lactate: "High Lactate",
  hypotension: "Hypotension",
};

export const DEFAULT_VITALS: PatientVitals = {
  temperature_c: "",
  bp_systolic: "",
  bp_diastolic: "",
  heart_rate: "",
  wbc_count: "",
  lactate: "",
  ward: "",
};

export const SAMPLE_HIGH_RISK: PatientVitals = {
  temperature_c: 39.5,
  bp_systolic: 84,
  bp_diastolic: 52,
  heart_rate: 118,
  wbc_count: 14.2,
  lactate: 3.1,
  ward: "ICU-A",
};

export const SAMPLE_LOW_RISK: PatientVitals = {
  temperature_c: 37.1,
  bp_systolic: 122,
  bp_diastolic: 78,
  heart_rate: 72,
  wbc_count: 7.5,
  lactate: 1.2,
  ward: "Ward-X",
};
