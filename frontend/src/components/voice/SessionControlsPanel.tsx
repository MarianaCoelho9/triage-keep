import React from "react";
import { Mic, MicOff } from "lucide-react";
import { type InteractionMode, type SessionStatus } from "@/components/voice/types";

interface SessionControlsPanelProps {
  status: SessionStatus;
  isRecording: boolean;
  isConnected: boolean;
  extractionStatus: string | null;
  isReportLoading: boolean;
  reportRequested: boolean;
  messagesCount: number;
  interactionMode: InteractionMode;
  onToggleRecording: () => void;
  onEndSession: () => void;
}

export default function SessionControlsPanel({
  status,
  isRecording,
  isConnected,
  extractionStatus,
  isReportLoading,
  reportRequested,
  messagesCount,
  interactionMode,
  onToggleRecording,
  onEndSession,
}: SessionControlsPanelProps) {
  const recordingHint =
    interactionMode === "auto_turn"
      ? isRecording
        ? "Hands-free active: tap to pause"
        : "Tap microphone to start hands-free"
      : isRecording
        ? "Tap to stop"
        : "Tap microphone to start";

  return (
    <div className="w-full md:w-[320px] lg:w-[380px] flex flex-col items-center justify-between bg-zinc-50/50 p-8 border-r border-zinc-100 relative">
      <div className="w-full flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}></div>
          <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{isConnected ? "Online" : "Offline"}</span>
        </div>
        <button
          onClick={onEndSession}
          disabled={!isConnected || reportRequested || messagesCount === 0}
          className={`text-xs font-semibold uppercase tracking-wider px-3 py-2 rounded-full border transition-all
            ${
              !isConnected || reportRequested || messagesCount === 0
                ? "bg-zinc-100 text-zinc-300 border-zinc-200 cursor-not-allowed"
                : "bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50"
            }`}
        >
          {isReportLoading ? "Generating..." : "End Connection"}
        </button>
      </div>

      <div className="flex flex-col items-center justify-center flex-1 w-full relative">
        <div className="absolute top-10 flex flex-col items-center transition-all duration-500">
          <span
            className={`text-2xl font-semibold tracking-tight transition-colors duration-300
              ${
                status === "listening"
                  ? "text-rose-500"
                  : status === "speaking"
                    ? "text-indigo-500"
                    : status === "processing"
                      ? "text-amber-500"
                      : "text-zinc-400"
              }`}
          >
            {status === "idle" && "Ready"}
            {status === "listening" && "Listening"}
            {status === "processing" && "Thinking"}
            {status === "speaking" && "Speaking"}
          </span>
          {status === "listening" && <span className="text-xs text-rose-400 mt-1 animate-pulse">Detecting voice...</span>}
          {extractionStatus && <span className="text-xs text-zinc-500 mt-1">{extractionStatus}</span>}
        </div>

        <div className="relative w-48 h-48 flex items-center justify-center">
          <div
            className={`absolute inset-0 rounded-full border border-zinc-200 transition-all duration-700
              ${status === "listening" ? "scale-110 border-rose-200 opacity-100" : "scale-100 opacity-30"}`}
          ></div>
          <div
            className={`absolute inset-4 rounded-full border border-zinc-200 transition-all duration-700 delay-75
              ${status === "listening" ? "scale-110 border-rose-300 opacity-100" : "scale-100 opacity-30"}`}
          ></div>

          <button
            onClick={onToggleRecording}
            disabled={!isConnected}
            className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl focus:outline-none focus:ring-4 focus:ring-offset-2 focus:ring-zinc-200
              ${
                isRecording
                  ? "bg-rose-500 text-white shadow-rose-500/40 scale-105"
                  : "bg-white text-zinc-800 hover:scale-105 shadow-zinc-200/50 hover:shadow-zinc-300/50 border border-zinc-100"
              }`}
          >
            {isRecording ? <MicOff size={32} strokeWidth={2} /> : <Mic size={32} strokeWidth={2} />}
          </button>

          {status === "speaking" && (
            <>
              <div className="absolute inset-0 rounded-full border-2 border-indigo-400 opacity-20 animate-ping"></div>
              <div className="absolute inset-8 rounded-full border-2 border-indigo-400 opacity-40 animate-ping animation-delay-300"></div>
            </>
          )}
        </div>
      </div>

      <div className="w-full text-center">
        <p className="text-xs text-zinc-400 font-medium">{recordingHint}</p>
      </div>
    </div>
  );
}
