import {
  type ErrorWithStatus,
  type FhirExportEnvelope,
  type FhirExportPayload,
  type ReportEnvelope,
  type ReportPayload,
} from "@/components/voice/types";
import { FHIR_REPORT_PATH, REPORT_PATH } from "@/lib/voice/endpoints";

export const postReport = async (payload: ReportPayload): Promise<ReportEnvelope> => {
  const response = await fetch(REPORT_PATH, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = new Error(`Report request failed (${response.status})`) as ErrorWithStatus;
    error.status = response.status;
    throw error;
  }

  return (await response.json()) as ReportEnvelope;
};

export const postFhirExport = async (payload: FhirExportPayload): Promise<FhirExportEnvelope> => {
  const response = await fetch(FHIR_REPORT_PATH, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = new Error(`FHIR export request failed (${response.status})`) as ErrorWithStatus;
    error.status = response.status;
    throw error;
  }

  return (await response.json()) as FhirExportEnvelope;
};
