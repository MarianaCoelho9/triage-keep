"use client";

import { useCallback, useState } from "react";
import { type JsonRecord } from "@/components/voice/types";
import { postFhirExport } from "@/lib/voice/http";

interface UseFhirExportResult {
  isFhirExporting: boolean;
  fhirExportError: string | null;
  fhirExportSuccess: string | null;
  fhirExportWarnings: string[];
  exportFhir: (reportData: JsonRecord | null) => Promise<void>;
  resetFhirExportState: () => void;
}

const toRecord = (value: unknown): JsonRecord => {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as JsonRecord;
  }
  return {};
};

const toStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item));
};

const getProcessId = (report: JsonRecord | null): string => {
  if (!report) return "unknown";
  const administrative = toRecord(report.administrative);
  const processId = administrative.process_id;
  if (typeof processId === "string" && processId.trim()) return processId.trim();
  return "unknown";
};

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(href);
};

export function useFhirExport(): UseFhirExportResult {
  const [isFhirExporting, setIsFhirExporting] = useState(false);
  const [fhirExportError, setFhirExportError] = useState<string | null>(null);
  const [fhirExportSuccess, setFhirExportSuccess] = useState<string | null>(null);
  const [fhirExportWarnings, setFhirExportWarnings] = useState<string[]>([]);

  const exportFhir = useCallback(
    async (reportData: JsonRecord | null) => {
      if (!reportData || isFhirExporting) return;
      setIsFhirExporting(true);
      setFhirExportError(null);
      setFhirExportSuccess(null);
      setFhirExportWarnings([]);

      try {
        const response = await postFhirExport({ report: reportData, include_validation: true });
        if (!response.success) {
          setFhirExportError(response.error?.message || "Failed to export FHIR bundle.");
          return;
        }

        const exportData = toRecord(response.data);
        const bundleCandidate = "bundle" in exportData ? exportData.bundle : exportData;
        const bundle = toRecord(bundleCandidate);
        const warnings = "warnings" in exportData ? toStringArray(exportData.warnings) : [];
        if (warnings.length > 0) {
          setFhirExportWarnings(warnings);
        }

        if (bundle.resourceType !== "Bundle") {
          setFhirExportError("FHIR export succeeded but response did not contain a valid Bundle.");
          return;
        }

        const processId = getProcessId(reportData).replace(/[^a-zA-Z0-9-_]/g, "_");
        downloadJson(`triage-fhir-${processId}.json`, bundle);
        setFhirExportSuccess(warnings.length > 0 ? "FHIR exported with warnings." : "FHIR exported successfully.");
      } catch (unknownError: unknown) {
        const error = unknownError as Error;
        setFhirExportError(error.message || "Failed to export FHIR bundle.");
      } finally {
        setIsFhirExporting(false);
      }
    },
    [isFhirExporting],
  );

  const resetFhirExportState = useCallback(() => {
    setIsFhirExporting(false);
    setFhirExportError(null);
    setFhirExportSuccess(null);
    setFhirExportWarnings([]);
  }, []);

  return {
    isFhirExporting,
    fhirExportError,
    fhirExportSuccess,
    fhirExportWarnings,
    exportFhir,
    resetFhirExportState,
  };
}
