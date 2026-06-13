import React, { useState, useCallback } from "react";
import type { PatientVitals } from "../types";
import {
  WARD_OPTIONS,
  VITALS_RANGES,
  DEFAULT_VITALS,
  SAMPLE_HIGH_RISK,
  SAMPLE_LOW_RISK,
} from "../types";

interface PatientFormProps {
  onSubmit: (vitals: PatientVitals) => void;
  isLoading: boolean;
}

export default function PatientForm({ onSubmit, isLoading }: PatientFormProps) {
  const [vitals, setVitals] = useState<PatientVitals>(DEFAULT_VITALS);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateField = useCallback(
    (name: keyof typeof VITALS_RANGES, value: number | ""): string | null => {
      if (value === "" || isNaN(value)) {
        return "Field is required";
      }
      const range = VITALS_RANGES[name];
      if (value < range.min || value > range.max) {
        return `Must be between ${range.min} and ${range.max}`;
      }
      return null;
    },
    []
  );

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;

    if (name === "ward") {
      setVitals((prev) => ({ ...prev, ward: value as PatientVitals["ward"] }));
      setErrors((prev) => {
        const next = { ...prev };
        if (value === "") next["ward"] = "Please select a ward";
        else delete next["ward"];
        return next;
      });
      return;
    }

    const numValue = value === "" ? "" : parseFloat(value);
    setVitals((prev) => ({ ...prev, [name]: numValue }));

    if (name in VITALS_RANGES) {
      const error = validateField(
        name as keyof typeof VITALS_RANGES,
        numValue
      );
      setErrors((prev) => {
        const next = { ...prev };
        if (error) next[name] = error;
        else delete next[name];
        return next;
      });
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate all fields
    const newErrors: Record<string, string> = {};
    (Object.keys(VITALS_RANGES) as Array<keyof typeof VITALS_RANGES>).forEach(
      (key) => {
        const error = validateField(key, vitals[key]);
        if (error) newErrors[key] = error;
      }
    );

    if (vitals.ward === "") {
      newErrors["ward"] = "Please select a ward";
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    onSubmit(vitals as PatientVitals);
  };

  const loadSample = (sample: PatientVitals) => {
    setVitals(sample);
    setErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className="patient-form">
      <div className="form-header">
        <h2>
          <span className="form-icon">🩺</span> Patient Vitals
        </h2>
        <div className="sample-buttons">
          <button
            type="button"
            className="btn-sample btn-sample-high"
            onClick={() => loadSample(SAMPLE_HIGH_RISK)}
          >
            ⚠️ Load High Risk
          </button>
          <button
            type="button"
            className="btn-sample btn-sample-low"
            onClick={() => loadSample(SAMPLE_LOW_RISK)}
          >
            ✅ Load Low Risk
          </button>
        </div>
      </div>

      <div className="form-grid">
        {(
          Object.entries(VITALS_RANGES) as [
            keyof typeof VITALS_RANGES,
            (typeof VITALS_RANGES)[keyof typeof VITALS_RANGES]
          ][]
        ).map(([key, range]) => (
          <div
            key={key}
            className={`form-field ${errors[key] ? "form-field-error" : ""}`}
          >
            <label htmlFor={key}>{range.label}</label>
            <div className="input-wrapper">
              <input
                type="number"
                id={key}
                name={key}
                value={vitals[key]}
                onChange={handleChange}
                min={range.min}
                max={range.max}
                step={range.step}
                required
              />
              <span className="input-unit">{range.unit}</span>
            </div>
            {errors[key] && <span className="error-text">{errors[key]}</span>}
          </div>
        ))}

        <div className={`form-field ${errors.ward ? "form-field-error" : ""}`}>
          <label htmlFor="ward">Hospital Ward</label>
          <select
            id="ward"
            name="ward"
            value={vitals.ward}
            onChange={handleChange}
            required
          >
            <option value="" disabled>Select Ward</option>
            {WARD_OPTIONS.map((ward) => (
              <option key={ward} value={ward}>
                {ward}
              </option>
            ))}
          </select>
          {errors.ward && <span className="error-text">{errors.ward}</span>}
        </div>
      </div>

      <button
        type="submit"
        className="btn-predict"
        disabled={isLoading || Object.keys(errors).length > 0}
      >
        {isLoading ? (
          <>
            <span className="spinner" /> Analyzing...
          </>
        ) : (
          <>🔬 Predict Sepsis Risk</>
        )}
      </button>
    </form>
  );
}
