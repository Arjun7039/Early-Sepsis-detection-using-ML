/**
 * API Client for the Sepsis Detection Backend
 */

import axios from "axios";
import type { PatientVitals, PredictionResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function predictSepsis(
  vitals: PatientVitals
): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>(
    "/predict",
    vitals
  );
  return response.data;
}

export async function healthCheck(): Promise<{ status: string; model: string }> {
  const response = await apiClient.get("/");
  return response.data;
}

export async function getModelInfo(): Promise<Record<string, unknown>> {
  const response = await apiClient.get("/model/info");
  return response.data;
}

export async function getSamplePatient(): Promise<Record<string, PatientVitals>> {
  const response = await apiClient.get("/model/sample");
  return response.data;
}

export { API_BASE };
