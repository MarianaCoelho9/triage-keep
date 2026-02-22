"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Gauge, HeartPulse, UserRound } from "lucide-react";

type ExtractionData = Record<string, unknown>;

interface TriageDashboardProps {
  data: ExtractionData | string | null;
  extractionStatus?: string | null;
  lastUpdatedAt?: number | null;
  isFinalized?: boolean;
}

interface SeverityColors {
  bg: string;
  text: string;
  border: string;
}

const getSeverityStyles = (risk: string): SeverityColors => {
  const normalizedRisk = risk.toLowerCase();
  if (normalizedRisk.includes("high") || normalizedRisk.includes("critical") || normalizedRisk.includes("emergency")) {
    return { bg: "bg-red-50", text: "text-red-700", border: "border-red-100" };
  }
  if (normalizedRisk.includes("medium") || normalizedRisk.includes("moderate")) {
    return { bg: "bg-yellow-50", text: "text-yellow-700", border: "border-yellow-100" };
  }
  if (normalizedRisk.includes("low") || normalizedRisk.includes("minor")) {
    return { bg: "bg-green-50", text: "text-green-700", border: "border-green-100" };
  }
  return { bg: "bg-zinc-50", text: "text-zinc-700", border: "border-zinc-100" };
};

const renderValue = (value: unknown): string => {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const formatElapsed = (timestamp: number | null | undefined): string => {
  if (!timestamp) return "No updates yet";
  const secondsAgo = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (secondsAgo < 5) return "Updated just now";
  if (secondsAgo < 60) return `Updated ${secondsAgo}s ago`;
  return `Updated ${Math.floor(secondsAgo / 60)}m ago`;
};

export default function TriageDashboard({
  data,
  extractionStatus,
  lastUpdatedAt,
  isFinalized = false,
}: TriageDashboardProps) {
  const parsedData = useMemo<ExtractionData | null>(() => {
    if (!data) return null;
    if (typeof data === "string") {
      try {
        const parsed = JSON.parse(data);
        return typeof parsed === "object" && parsed !== null ? (parsed as ExtractionData) : null;
      } catch {
        return null;
      }
    }
    return data;
  }, [data]);

  const previousDataRef = useRef<ExtractionData | null>(null);
  const [changedFields, setChangedFields] = useState<string[]>([]);
  const [, setNowTick] = useState(0);

  useEffect(() => {
    if (!lastUpdatedAt || isFinalized) return;
    const interval = window.setInterval(() => {
      setNowTick((previous) => previous + 1);
    }, 1000);
    return () => window.clearInterval(interval);
  }, [lastUpdatedAt, isFinalized]);

  useEffect(() => {
    if (!parsedData) return;

    const previous = previousDataRef.current;
    if (!previous) {
      previousDataRef.current = parsedData;
      return;
    }

    const trackedKeys = ["main_complaint", "additional_symptoms", "medical_history", "severity_risk"];
    const nextChanged = trackedKeys.filter((key) => renderValue(previous[key]) !== renderValue(parsedData[key]));
    previousDataRef.current = parsedData;

    if (nextChanged.length === 0) return;

    setChangedFields(nextChanged);
    const timeout = window.setTimeout(() => setChangedFields([]), 900);
    return () => window.clearTimeout(timeout);
  }, [parsedData]);

  if (!parsedData) {
    return (
      <div className="w-96 h-full bg-zinc-50 border-r border-zinc-100 p-6 flex flex-col items-center justify-center text-zinc-400">
        <Activity className="mb-4 opacity-50" size={48} />
        <p className="text-sm font-medium">Waiting for triage data...</p>
      </div>
    );
  }

  const symptom = renderValue(parsedData.main_complaint || parsedData.symptom);
  const severityRisk = renderValue(parsedData.severity_risk);
  const additionalSymptoms = renderValue(parsedData.additional_symptoms);
  const history = renderValue(parsedData.medical_history || parsedData.history);
  const severityStyles = severityRisk ? getSeverityStyles(severityRisk) : null;
  const isFieldUpdated = (field: string) =>
    changedFields.includes(field) ? "ring-1 ring-indigo-200 bg-indigo-50/60 transition-colors duration-700" : "";
  const statusText = extractionStatus || "Connected";
  const unresolvedValue = isFinalized ? "-" : "Listening...";

  return (
    <div className="w-96 h-full bg-white border-r border-zinc-100 flex flex-col overflow-y-auto animate-in fade-in slide-in-from-left duration-500">
      <div className="p-6 border-b border-zinc-50">
        <h2 className="text-lg font-bold text-zinc-800 flex items-center gap-2">
          <Gauge className="text-indigo-500" size={20} />
          Triage Dashboard
        </h2>
        <div className="flex items-center gap-2 mt-2 text-xs">
          <span
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-full border ${
              isFinalized
                ? "border-indigo-100 text-indigo-700 bg-indigo-50"
                : "border-emerald-100 text-emerald-700 bg-emerald-50"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                isFinalized ? "bg-indigo-500" : "bg-emerald-500 animate-pulse"
              }`}
            />
            {isFinalized ? "Finalized" : "Live"}
          </span>
          <span className="text-zinc-500">{statusText}</span>
        </div>
        <p className="text-xs text-zinc-400 mt-2">{formatElapsed(lastUpdatedAt)}</p>
      </div>

      <div className="p-6 space-y-5">
        <div className="space-y-2">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-[0.12em] flex items-center gap-1">
            <Activity size={12} /> Main Complaint
          </label>
          <div className={`p-3 bg-zinc-50 border border-zinc-100 rounded-xl text-zinc-800 font-medium ${isFieldUpdated("main_complaint")}`}>
            {symptom || unresolvedValue}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-[0.12em] flex items-center gap-1">
            <HeartPulse size={12} /> Severity Risk
          </label>
          <div
            className={`p-3 rounded-xl font-bold uppercase tracking-wide border ${
              severityStyles ? `${severityStyles.bg} ${severityStyles.text} ${severityStyles.border}` : "bg-zinc-50 text-zinc-700 border-zinc-100"
            } ${isFieldUpdated("severity_risk")}`}
          >
            {severityRisk || "unknown"}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-[0.12em]">
            Additional Symptoms
          </label>
          <p className={`text-sm text-zinc-700 bg-zinc-50 p-3 rounded-lg border border-zinc-100 ${isFieldUpdated("additional_symptoms")}`}>
            {additionalSymptoms || unresolvedValue}
          </p>
        </div>

        <div className="space-y-2 pt-1">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-[0.12em] flex items-center gap-1">
            <UserRound size={12} /> Medical History
          </label>
          <div className={`text-sm text-zinc-600 leading-relaxed italic p-3 bg-zinc-50 rounded-lg border border-zinc-100 ${isFieldUpdated("medical_history")}`}>
            {history ? `“${history}”` : unresolvedValue}
          </div>
        </div>

        {Object.entries(parsedData).map(([key, value]) => {
          if (["symptom", "pain_level", "history", "severity_risk", "main_complaint", "additional_symptoms", "medical_history"].includes(key)) return null;
          return (
            <div key={key} className="space-y-1">
              <label className="text-xs font-semibold text-zinc-500 uppercase tracking-[0.12em]">
                {key.replace(/_/g, " ")}
              </label>
              <p className="text-sm text-zinc-700 bg-zinc-50 p-2 rounded-lg border border-zinc-100">
                {renderValue(value)}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
