import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

interface ShapChartProps {
  shapValues: Record<string, number>;
}

const FEATURE_LABELS: Record<string, string> = {
  Temperature_C: "Temperature",
  BP_Systolic: "Systolic BP",
  BP_Diastolic: "Diastolic BP",
  Heart_Rate: "Heart Rate",
  WBC_Count: "WBC Count",
  Lactate_mmol_L: "Lactate",
  "Ward_ER": "Ward: ER",
  "Ward_ICU-A": "Ward: ICU-A",
  "Ward_ICU-B": "Ward: ICU-B",
  "Ward_Ward-X": "Ward: Ward-X",
};

export default function ShapChart({ shapValues }: ShapChartProps) {
  const data = useMemo(() => {
    return Object.entries(shapValues)
      .map(([feature, value]) => ({
        feature: FEATURE_LABELS[feature] || feature,
        value: parseFloat(value.toFixed(4)),
        fullName: feature,
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [shapValues]);

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { feature: string; value: number } }> }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      const direction = item.value > 0 ? "↑ Risk Driver" : "↓ Protective";
      const color = item.value > 0 ? "#ef4444" : "#3b82f6";
      return (
        <div className="shap-tooltip">
          <p className="shap-tooltip-feature">{item.feature}</p>
          <p style={{ color }}>
            SHAP: {item.value.toFixed(4)} ({direction})
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="shap-chart">
      <div className="shap-header">
        <h3>
          <span className="shap-icon">📊</span> Feature Impact (SHAP Values)
        </h3>
        <div className="shap-legend">
          <span className="shap-legend-item">
            <span
              className="shap-legend-dot"
              style={{ backgroundColor: "#ef4444" }}
            />
            Risk Driver
          </span>
          <span className="shap-legend-item">
            <span
              className="shap-legend-dot"
              style={{ backgroundColor: "#3b82f6" }}
            />
            Protective
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: "#cbd5e1", fontSize: 13 }}
            width={90}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.value > 0 ? "#ef4444" : "#3b82f6"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
