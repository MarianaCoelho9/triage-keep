"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TriageDashboard from "./TriageDashboard";
import ReportPanel from "./ReportPanel";
import SessionControlsPanel from "@/components/voice/SessionControlsPanel";
import TranscriptPanel from "@/components/voice/TranscriptPanel";
import { type InteractionMode } from "@/components/voice/types";
import { useFhirExport } from "@/hooks/useFhirExport";
import { useReportFlow } from "@/hooks/useReportFlow";
import { useVoiceSession } from "@/hooks/useVoiceSession";

const DEFAULT_INTERACTION_MODE: InteractionMode =
  process.env.NEXT_PUBLIC_VOICE_INTERACTION_MODE === "auto_turn" ? "auto_turn" : "ptt";

export default function VoiceAgent() {
  const [sessionId, setSessionId] = useState(0);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  const reportFlow = useReportFlow();
  const fhirExport = useFhirExport();
  const voiceSessionConfig = useMemo(
    () => ({
      mode: DEFAULT_INTERACTION_MODE,
      silenceMs: 1000,
      commitCooldownMs: 1200,
      minSpeechMs: 300,
      allowBargeIn: true,
    }),
    [],
  );

  const voiceSession = useVoiceSession({
    sessionId,
    config: voiceSessionConfig,
    onReportStatus: reportFlow.handleReportStatus,
    onReportMessage: reportFlow.handleReportMessage,
  });

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [voiceSession.messages]);

  const endSession = useCallback(async () => {
    await reportFlow.endSession({
      messages: voiceSession.messages,
      isRecording: voiceSession.isRecording,
      stopRecording: voiceSession.stopRecording,
      disconnect: voiceSession.disconnect,
    });
  }, [reportFlow, voiceSession]);

  const restartSession = useCallback(() => {
    voiceSession.disconnect();
    voiceSession.resetSessionState();
    reportFlow.resetReportState();
    fhirExport.resetFhirExportState();
    setSessionId((previous) => previous + 1);
  }, [fhirExport, reportFlow, voiceSession]);

  return (
    <div className="flex flex-col md:flex-row h-[700px] w-full bg-white rounded-[32px] overflow-hidden shadow-2xl border border-white/40 ring-1 ring-zinc-900/5 backdrop-blur-xl">
      <TriageDashboard
        data={voiceSession.extractionData}
        extractionStatus={voiceSession.extractionStatus}
        lastUpdatedAt={voiceSession.lastExtractionUpdateAt}
        isFinalized={voiceSession.isExtractionFinal}
      />

      {!reportFlow.reportRequested && (
        <>
          <SessionControlsPanel
            status={voiceSession.status}
            isRecording={voiceSession.isRecording}
            isConnected={voiceSession.isConnected}
            extractionStatus={voiceSession.extractionStatus}
            isReportLoading={reportFlow.isReportLoading}
            reportRequested={reportFlow.reportRequested}
            messagesCount={voiceSession.messages.length}
            interactionMode={voiceSession.interactionMode}
            onToggleRecording={voiceSession.toggleRecording}
            onEndSession={() => void endSession()}
          />
          <TranscriptPanel messages={voiceSession.messages} bottomRef={chatBottomRef} />
        </>
      )}

      {reportFlow.reportRequested && (
        <ReportPanel
          reportData={reportFlow.reportData}
          loading={reportFlow.isReportLoading}
          error={reportFlow.reportError}
          onExportFhir={() => void fhirExport.exportFhir(reportFlow.reportData)}
          isFhirExporting={fhirExport.isFhirExporting}
          fhirExportError={fhirExport.fhirExportError}
          fhirExportSuccess={fhirExport.fhirExportSuccess}
          fhirExportWarnings={fhirExport.fhirExportWarnings}
          onRetry={() => void endSession()}
          onRestart={restartSession}
        />
      )}
    </div>
  );
}
