"use client";

import React from "react";
import { ClipboardList, AlertTriangle, Loader2, User, Stethoscope, Signpost, ListChecks } from "lucide-react";

type JsonRecord = Record<string, unknown>;

interface ReportPanelProps {
  reportData: JsonRecord | null;
  loading: boolean;
  error: string | null;
  onExportFhir?: () => void;
  isFhirExporting?: boolean;
  fhirExportError?: string | null;
  fhirExportSuccess?: string | null;
  fhirExportWarnings?: string[];
  onRetry?: () => void;
  onRestart?: () => void;
}

const asRecord = (value: unknown): JsonRecord => {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as JsonRecord;
  }
  return {};
};

const renderValue = (value: unknown): string => {
  if (value === null || value === undefined) return "Not specified";
  if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "Not specified";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
};

const formatTimestamp = (value: unknown): string => {
  if (!value) return "Not specified";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
};

export default function ReportPanel({
  reportData,
  loading,
  error,
  onExportFhir,
  isFhirExporting = false,
  fhirExportError = null,
  fhirExportSuccess = null,
  fhirExportWarnings = [],
  onRetry,
  onRestart,
}: ReportPanelProps) {
  const report = asRecord(reportData);
  const administrative = asRecord(report.administrative);
  const patientInformation = asRecord(report.patient_information);
  const assessment = asRecord(report.assessment);
  const hpi = asRecord(assessment.hpi);
  const disposition = asRecord(report.disposition);
  const plan = asRecord(report.plan);

  return (
    <div className="flex-1 flex flex-col bg-white relative">
      <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-white via-white to-transparent z-10 pointer-events-none"></div>

      <div className="flex-1 overflow-y-auto p-8 pt-8 pb-24 space-y-8 scroll-smooth">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center">
            <ClipboardList size={20} className="text-indigo-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-zinc-900">Final Triage Report</h2>
            <p className="text-xs text-zinc-400">Generated from the conversation history</p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center gap-3 text-zinc-500 bg-zinc-50 border border-zinc-100 rounded-xl p-4">
            <Loader2 className="animate-spin" size={18} />
            <span>Generating report...</span>
          </div>
        )}

        {error && !loading && (
          <div className="flex items-start gap-3 text-rose-600 bg-rose-50 border border-rose-100 rounded-xl p-4">
            <AlertTriangle size={18} className="mt-0.5" />
            <div className="flex-1">
              <p className="font-medium">Failed to generate report</p>
              <p className="text-sm text-rose-500 mt-1">{error}</p>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="mt-3 text-sm font-medium text-rose-600 hover:text-rose-700"
                >
                  Retry
                </button>
              )}
            </div>
          </div>
        )}

        {!loading && !error && !reportData && (
          <div className="text-zinc-400 bg-zinc-50 border border-zinc-100 rounded-xl p-6">
            No report data available.
          </div>
        )}

        {!loading && !error && reportData && (
          <div className="space-y-6">
            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider">FHIR Export</h3>
                  <p className="text-xs text-zinc-400 mt-1">Export this triage report as a FHIR Bundle JSON file</p>
                </div>
                {onExportFhir && (
                  <button
                    onClick={onExportFhir}
                    disabled={isFhirExporting}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                      isFhirExporting
                        ? "bg-zinc-100 text-zinc-400 cursor-not-allowed"
                        : "bg-emerald-600 text-white hover:bg-emerald-700"
                    }`}
                  >
                    {isFhirExporting ? "Exporting..." : "Export FHIR JSON"}
                  </button>
                )}
              </div>
              {fhirExportSuccess && (
                <p className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-100 rounded-lg px-3 py-2 mt-4">
                  {fhirExportSuccess}
                </p>
              )}
              {fhirExportError && (
                <p className="text-sm text-rose-600 bg-rose-50 border border-rose-100 rounded-lg px-3 py-2 mt-4">
                  {fhirExportError}
                </p>
              )}
              {fhirExportWarnings.length > 0 && (
                <div className="bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 mt-4">
                  <p className="text-sm font-medium text-amber-700">Warnings</p>
                  <ul className="text-sm text-amber-700 mt-1 list-disc list-inside">
                    {fhirExportWarnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <ClipboardList size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Administrative</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-zinc-400">Process ID</p>
                  <p className="text-zinc-800 font-medium">{renderValue(administrative.process_id)}</p>
                </div>
                <div>
                  <p className="text-zinc-400">Timestamp</p>
                  <p className="text-zinc-800 font-medium">{formatTimestamp(administrative.timestamp)}</p>
                </div>
              </div>
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <User size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Patient Information</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-zinc-400">Sex</p>
                  <p className="text-zinc-800 font-medium capitalize">{renderValue(patientInformation.sex)}</p>
                </div>
                <div>
                  <p className="text-zinc-400">Age</p>
                  <p className="text-zinc-800 font-medium">{renderValue(patientInformation.age)}</p>
                </div>
              </div>
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <Stethoscope size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Assessment</h3>
              </div>
              <div className="mt-4 space-y-4 text-sm">
                <div>
                  <p className="text-zinc-400">Chief complaint</p>
                  <p className="text-zinc-800 font-medium">{renderValue(assessment.chief_complaint)}</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-zinc-400">Onset</p>
                    <p className="text-zinc-800 font-medium">{renderValue(hpi.onset)}</p>
                  </div>
                  <div>
                    <p className="text-zinc-400">Duration</p>
                    <p className="text-zinc-800 font-medium">{renderValue(hpi.duration)}</p>
                  </div>
                  <div>
                    <p className="text-zinc-400">Severity</p>
                    <p className="text-zinc-800 font-medium">{renderValue(hpi.severity)}</p>
                  </div>
                  <div>
                    <p className="text-zinc-400">Associated symptoms</p>
                    <p className="text-zinc-800 font-medium">{renderValue(hpi.associated_symptoms)}</p>
                  </div>
                </div>
                <div>
                  <p className="text-zinc-400">Red flags checked</p>
                  <p className="text-zinc-800 font-medium">{renderValue(assessment.red_flags_checked)}</p>
                </div>
                <div>
                  <p className="text-zinc-400">Medical history</p>
                  <p className="text-zinc-800 font-medium">{renderValue(assessment.medical_history)}</p>
                </div>
              </div>
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <Signpost size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Disposition</h3>
              </div>
              <div className="mt-3 space-y-2 text-sm">
                <div>
                  <p className="text-zinc-400">Triage level</p>
                  <p className="text-zinc-800 font-medium">{renderValue(disposition.triage_level)}</p>
                </div>
                <div>
                  <p className="text-zinc-400">Reasoning</p>
                  <p className="text-zinc-800 font-medium">{renderValue(disposition.reasoning)}</p>
                </div>
              </div>
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <ListChecks size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Plan</h3>
              </div>
              <div className="mt-3 space-y-2 text-sm">
                <div>
                  <p className="text-zinc-400">Care advice</p>
                  <p className="text-zinc-800 font-medium">{renderValue(plan.care_advice)}</p>
                </div>
                <div>
                  <p className="text-zinc-400">Worsening instructions</p>
                  <p className="text-zinc-800 font-medium">{renderValue(plan.worsening_instructions)}</p>
                </div>
              </div>
            </section>

            <section className="bg-white border border-zinc-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <ClipboardList size={16} className="text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Additional fields</h3>
              </div>
              <div className="mt-3 space-y-2 text-sm text-zinc-700">
                {Object.keys(report).filter((key) => {
                  return ![
                    "administrative",
                    "patient_information",
                    "assessment",
                    "disposition",
                    "plan",
                  ].includes(key);
                }).length === 0 && (
                  <p className="text-zinc-400">None</p>
                )}
                {Object.entries(report).map(([key, value]) => {
                  if ([
                    "administrative",
                    "patient_information",
                    "assessment",
                    "disposition",
                    "plan",
                  ].includes(key)) {
                    return null;
                  }
                  return (
                    <div key={key} className="bg-zinc-50 border border-zinc-100 rounded-lg p-3">
                      <p className="text-xs text-zinc-400 uppercase tracking-wider">{key}</p>
                      <pre className="mt-2 text-xs text-zinc-700 whitespace-pre-wrap">{renderValue(value)}</pre>
                    </div>
                  );
                })}
              </div>
            </section>
          </div>
        )}
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/80 backdrop-blur-md border-t border-zinc-50 flex flex-col items-center gap-3">
        {onRestart && (
          <button
            onClick={onRestart}
            className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white text-sm font-medium rounded-full shadow-lg shadow-indigo-200 transition-all transform hover:scale-105"
          >
            Start New Conversation
          </button>
        )}
        <p className="text-[10px] text-zinc-300 uppercase tracking-widest font-semibold">Processed by MedGemma</p>
      </div>
    </div>
  );
}
