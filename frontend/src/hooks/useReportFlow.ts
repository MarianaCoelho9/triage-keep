"use client";

import { useCallback, useState } from "react";
import {
  type ErrorWithStatus,
  type JsonRecord,
  type ReportStatus,
  type TranscriptMessage,
} from "@/components/voice/types";
import { postReport } from "@/lib/voice/http";

interface EndSessionParams {
  messages: TranscriptMessage[];
  isRecording: boolean;
  stopRecording: () => void;
  disconnect: () => void;
}

interface UseReportFlowResult {
  reportRequested: boolean;
  isReportLoading: boolean;
  reportData: JsonRecord | null;
  reportError: string | null;
  endSession: (params: EndSessionParams) => Promise<void>;
  handleReportStatus: (status: ReportStatus) => void;
  handleReportMessage: (success: boolean, data?: JsonRecord, error?: { message: string } | null) => void;
  resetReportState: () => void;
}

const buildChatHistory = (messages: TranscriptMessage[]) => {
  return messages.map((message) => ({
    role: (message.role === "agent" ? "assistant" : "user") as "assistant" | "user",
    content: message.text,
  }));
};

export function useReportFlow(): UseReportFlowResult {
  const [reportRequested, setReportRequested] = useState(false);
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [reportData, setReportData] = useState<JsonRecord | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handleReportStatus = useCallback((status: ReportStatus) => {
    if (status === "running") {
      setReportRequested(true);
      setIsReportLoading(true);
      setReportError(null);
      return;
    }

    if (status === "failed") {
      setIsReportLoading(false);
      setReportRequested(true);
    }
  }, []);

  const handleReportMessage = useCallback((success: boolean, data?: JsonRecord, error?: { message: string } | null) => {
    setReportRequested(true);
    setIsReportLoading(false);
    if (success) {
      setReportData(data ?? {});
      setReportError(null);
      return;
    }

    setReportData(null);
    setReportError(error?.message || "Failed to generate report.");
  }, []);

  const endSession = useCallback(
    async ({ messages, isRecording, stopRecording, disconnect }: EndSessionParams) => {
      if (isReportLoading) return;
      setReportRequested(true);
      setIsReportLoading(true);
      setReportError(null);

      if (isRecording) stopRecording();
      disconnect();

      const history = buildChatHistory(messages);
      if (history.length === 0) {
        setIsReportLoading(false);
        setReportError("No conversation history available.");
        return;
      }

      try {
        const reportResponse = await postReport({ chat_history: history });
        if (reportResponse.success) {
          setReportData(reportResponse.data);
        } else {
          setReportError(reportResponse.error?.message || "Failed to generate report.");
        }
      } catch (unknownError: unknown) {
        const error = unknownError as ErrorWithStatus;
        if (error.status === 422) {
          const lastUser = [...messages].reverse().find((message) => message.role === "user");
          if (lastUser) {
            const trimmedHistory = [...history];
            for (let index = trimmedHistory.length - 1; index >= 0; index -= 1) {
              if (
                trimmedHistory[index].role === "user" &&
                trimmedHistory[index].content === lastUser.text
              ) {
                trimmedHistory.splice(index, 1);
                break;
              }
            }
            try {
              const fallbackResponse = await postReport({
                chat_history: trimmedHistory,
                user_input: lastUser.text,
              });
              if (fallbackResponse.success) {
                setReportData(fallbackResponse.data);
              } else {
                setReportError(fallbackResponse.error?.message || "Failed to generate report.");
              }
            } catch (fallbackError: unknown) {
              const castError = fallbackError as Error;
              setReportError(castError.message || "Failed to generate report.");
            } finally {
              setIsReportLoading(false);
            }
            return;
          }
        }
        setReportError(error.message || "Failed to generate report.");
      } finally {
        setIsReportLoading(false);
      }
    },
    [isReportLoading],
  );

  const resetReportState = useCallback(() => {
    setReportRequested(false);
    setIsReportLoading(false);
    setReportData(null);
    setReportError(null);
  }, []);

  return {
    reportRequested,
    isReportLoading,
    reportData,
    reportError,
    endSession,
    handleReportStatus,
    handleReportMessage,
    resetReportState,
  };
}
