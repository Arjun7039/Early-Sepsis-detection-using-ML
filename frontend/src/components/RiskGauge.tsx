import { useMemo } from "react";
import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  PolarAngleAxis,
} from "recharts";

interface RiskGaugeProps {
  riskScore: number;
  riskLevel: "Low" | "Moderate" | "High" | "Critical";
}

const GAUGE_COLORS = {
  Low: "#10b981",
  Moderate: "#f59e0b",
  High: "#f97316",
  Critical: "#ef4444",
};

export default function RiskGauge({ riskScore, riskLevel }: RiskGaugeProps) {
  const percentage = Math.round(riskScore * 100);
  const color = GAUGE_COLORS[riskLevel];

  const data = useMemo(
    () => [
      {
        name: "Risk",
        value: percentage,
        fill: color,
      },
    ],
    [percentage, color]
  );

  return (
    <div className="risk-gauge">
      <h3 className="gauge-title">Sepsis Risk Score</h3>
      <div className="gauge-container">
        <ResponsiveContainer width="100%" height={220}>
          <RadialBarChart
            cx="50%"
            cy="80%"
            innerRadius="60%"
            outerRadius="100%"
            barSize={20}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: "rgba(255,255,255,0.05)" }}
              dataKey="value"
              angleAxisId={0}
              cornerRadius={10}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="gauge-value" style={{ color }}>
          <span className="gauge-percentage">{percentage}</span>
          <span className="gauge-percent-sign">%</span>
        </div>
      </div>
    </div>
  );
}
