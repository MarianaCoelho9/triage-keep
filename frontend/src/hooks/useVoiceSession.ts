"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ApiError,
  type InteractionMode,
  type JsonRecord,
  type ReportStatus,
  type SessionStatus,
  type TranscriptMessage,
  type VoiceSessionConfig,
  type WindowWithWebkitAudio,
} from "@/components/voice/types";
import { INTRO_AUDIO_PATH, resolveWebSocketUrl } from "@/lib/voice/endpoints";
import { convertFloat32ToInt16, downsampleBuffer } from "@/lib/voice/audio";
import { parseTextEvent } from "@/lib/voice/protocol";

interface UseVoiceSessionProps {
  sessionId: number;
  config?: Partial<VoiceSessionConfig>;
  onReportStatus?: (status: ReportStatus) => void;
  onReportMessage?: (success: boolean, data?: JsonRecord, error?: ApiError | null) => void;
}

interface UseVoiceSessionResult {
  isRecording: boolean;
  isConnected: boolean;
  interactionMode: InteractionMode;
  status: SessionStatus;
  messages: TranscriptMessage[];
  extractionData: JsonRecord | null;
  extractionStatus: string | null;
  lastExtractionUpdateAt: number | null;
  isExtractionFinal: boolean;
  toggleRecording: () => void;
  stopRecording: () => void;
  disconnect: () => void;
  resetSessionState: () => void;
}

const DEFAULT_SESSION_CONFIG: VoiceSessionConfig = {
  mode: "ptt",
  silenceMs: 1000,
  commitCooldownMs: 1200,
  minSpeechMs: 300,
  allowBargeIn: true,
};

const AUTO_TURN_SPEECH_START_RMS = 0.02;
const AUTO_TURN_MIN_COMMIT_FRAMES = 5;
const AUTO_TURN_POST_COMMIT_HOLDOFF_MS = 2200;
const AUTO_TURN_POST_PLAYBACK_HOLDOFF_MS = 600;
const PROCESSING_WATCHDOG_MS = 8000;
const STATUS_WATCHDOG_INTERVAL_MS = 2000;
const WATCHDOG_LOG_COOLDOWN_MS = 5000;
const WS_KEEPALIVE_INTERVAL_MS = 5000;

type CommitReason = "manual_stop" | "cleanup" | "auto_vad";
type VoiceEventLevel = "info" | "warn" | "error";

export function useVoiceSession({
  sessionId,
  config,
  onReportStatus,
  onReportMessage,
}: UseVoiceSessionProps): UseVoiceSessionResult {
  const resolvedConfig = useMemo<VoiceSessionConfig>(
    () => ({
      ...DEFAULT_SESSION_CONFIG,
      ...config,
    }),
    [config],
  );

  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [messages, setMessages] = useState<TranscriptMessage[]>([]);
  const [extractionData, setExtractionData] = useState<JsonRecord | null>(null);
  const [extractionStatus, setExtractionStatus] = useState<string | null>(null);
  const [lastExtractionUpdateAt, setLastExtractionUpdateAt] = useState<number | null>(null);
  const [isExtractionFinal, setIsExtractionFinal] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const playbackSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const isPlayingRef = useRef(false);
  const isRecordingRef = useRef(false);
  const statusRef = useRef<SessionStatus>("idle");
  const shouldReconnectRef = useRef(true);
  const sessionEndingRef = useRef(false);
  const introPlayedRef = useRef(false);
  const lastCommitAtRef = useRef(0);
  const reportReceivedRef = useRef(false);
  const reportFailureNotifiedRef = useRef(false);

  const processingFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const processingWatchdogTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const runtimeWatchdogIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsKeepaliveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const awaitingAssistantAudioRef = useRef(false);
  const processingStartedAtRef = useRef<number | null>(null);

  const isSpeechActiveRef = useRef(false);
  const speechStartedAtRef = useRef<number | null>(null);
  const lastSpeechAtRef = useRef<number | null>(null);
  const speechFrameCountRef = useRef(0);
  const speechPeakRmsRef = useRef(0);
  const autoTurnMutedUntilRef = useRef(0);
  const assistantPlaybackEndedAtRef = useRef(0);

  const lastListeningWatchdogLogAtRef = useRef(0);
  const lastSpeakingWatchdogLogAtRef = useRef(0);

  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const playNextChunkRef = useRef<(() => Promise<void>) | null>(null);

  const logVoiceEvent = useCallback(
    (event: string, details: Record<string, unknown> = {}, level: VoiceEventLevel = "info") => {
      const payload = {
        event,
        sessionId,
        mode: resolvedConfig.mode,
        timestamp: new Date().toISOString(),
        ...details,
      };

      if (level === "warn") {
        console.warn("[voice-event]", payload);
        return;
      }
      if (level === "error") {
        console.error("[voice-event]", payload);
        return;
      }
      console.info("[voice-event]", payload);
    },
    [resolvedConfig.mode, sessionId],
  );

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const clearProcessingFallbackTimer = useCallback(() => {
    if (!processingFallbackTimerRef.current) return;
    clearTimeout(processingFallbackTimerRef.current);
    processingFallbackTimerRef.current = null;
  }, []);

  const clearProcessingWatchdogTimer = useCallback(() => {
    if (!processingWatchdogTimerRef.current) return;
    clearTimeout(processingWatchdogTimerRef.current);
    processingWatchdogTimerRef.current = null;
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (!reconnectTimerRef.current) return;
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;
  }, []);

  const clearWsKeepaliveInterval = useCallback(() => {
    if (!wsKeepaliveIntervalRef.current) return;
    clearInterval(wsKeepaliveIntervalRef.current);
    wsKeepaliveIntervalRef.current = null;
  }, []);

  const resetProcessingTracking = useCallback(() => {
    awaitingAssistantAudioRef.current = false;
    processingStartedAtRef.current = null;
    clearProcessingWatchdogTimer();
  }, [clearProcessingWatchdogTimer]);

  const scheduleProcessingFallback = useCallback(() => {
    clearProcessingFallbackTimer();
    processingFallbackTimerRef.current = setTimeout(() => {
      setStatus((previous) => {
        if (previous !== "processing") return previous;
        return isRecordingRef.current ? "listening" : "idle";
      });
    }, 1000);
  }, [clearProcessingFallbackTimer]);

  const startProcessingWatchdog = useCallback(() => {
    awaitingAssistantAudioRef.current = true;
    processingStartedAtRef.current = Date.now();
    clearProcessingWatchdogTimer();
    processingWatchdogTimerRef.current = setTimeout(() => {
      if (!awaitingAssistantAudioRef.current || statusRef.current !== "processing") return;
      logVoiceEvent(
        "unexpected_state_transition",
        {
          state: "processing_timeout_without_audio",
          timeout_ms: PROCESSING_WATCHDOG_MS,
        },
        "warn",
      );
    }, PROCESSING_WATCHDOG_MS);
  }, [clearProcessingWatchdogTimer, logVoiceEvent]);

  const stopAudioPlayback = useCallback(() => {
    if (playbackSourceRef.current) {
      try {
        playbackSourceRef.current.stop();
      } catch {
        // no-op: source might already be stopped
      }
      playbackSourceRef.current.disconnect();
      playbackSourceRef.current = null;
    }

    audioQueueRef.current = [];
    isPlayingRef.current = false;

    if (statusRef.current === "speaking") {
      assistantPlaybackEndedAtRef.current = Date.now();
      setStatus(isRecordingRef.current ? "listening" : "idle");
    }
  }, []);

  const resetSpeechState = useCallback(() => {
    isSpeechActiveRef.current = false;
    speechStartedAtRef.current = null;
    lastSpeechAtRef.current = null;
    speechFrameCountRef.current = 0;
    speechPeakRmsRef.current = 0;
  }, []);

  const stopSessionCapture = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    resetSpeechState();

    setIsRecording(false);
    if (statusRef.current === "listening") {
      setStatus("idle");
    }
  }, [resetSpeechState]);

  const commitTurn = useCallback(
    (reason: CommitReason): boolean => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false;

      const now = Date.now();
      if (now - lastCommitAtRef.current < resolvedConfig.commitCooldownMs) {
        logVoiceEvent("turn_commit_dropped_duplicate", {
          reason,
          cooldown_ms: resolvedConfig.commitCooldownMs,
        });
        return false;
      }

      wsRef.current.send("COMMIT");
      lastCommitAtRef.current = now;
      setStatus("processing");
      scheduleProcessingFallback();
      startProcessingWatchdog();
      logVoiceEvent("turn_commit_sent", { reason });
      if (reason === "auto_vad") {
        autoTurnMutedUntilRef.current = Date.now() + AUTO_TURN_POST_COMMIT_HOLDOFF_MS;
        resetSpeechState();
      }
      return true;
    },
    [
      logVoiceEvent,
      resetSpeechState,
      resolvedConfig.commitCooldownMs,
      scheduleProcessingFallback,
      startProcessingWatchdog,
    ],
  );

  const handleAutoTurnFrame = useCallback(
    (downsampled: Float32Array) => {
      if (resolvedConfig.mode !== "auto_turn") return;
      if (!isRecordingRef.current) return;
      if (sessionEndingRef.current) return;

      let sumSquares = 0;
      for (let index = 0; index < downsampled.length; index += 1) {
        const sample = downsampled[index];
        sumSquares += sample * sample;
      }
      const rms = Math.sqrt(sumSquares / Math.max(downsampled.length, 1));
      const now = Date.now();
      const currentStatus = statusRef.current;
      const canEvaluateAfterPlayback =
        assistantPlaybackEndedAtRef.current === 0 ||
        now - assistantPlaybackEndedAtRef.current >= AUTO_TURN_POST_PLAYBACK_HOLDOFF_MS;

      if (now < autoTurnMutedUntilRef.current || !canEvaluateAfterPlayback) {
        resetSpeechState();
        return;
      }

      if (currentStatus === "processing") {
        resetSpeechState();
        return;
      }

      if (currentStatus === "speaking") {
        if (!resolvedConfig.allowBargeIn) return;
        if (rms < AUTO_TURN_SPEECH_START_RMS) return;

        if (!isSpeechActiveRef.current) {
          isSpeechActiveRef.current = true;
          speechStartedAtRef.current = now;
          lastSpeechAtRef.current = now;
          speechFrameCountRef.current = 1;
          speechPeakRmsRef.current = rms;
          logVoiceEvent("vad_speech_start", {
            rms: Number(rms.toFixed(4)),
          });
        } else {
          lastSpeechAtRef.current = now;
          speechFrameCountRef.current += 1;
          speechPeakRmsRef.current = Math.max(speechPeakRmsRef.current, rms);
        }

        logVoiceEvent("barge_in_triggered", {
          reason: "speech_start_while_speaking",
        });
        stopAudioPlayback();
        clearProcessingFallbackTimer();
        setStatus("listening");
        return;
      }

      if (currentStatus !== "listening") {
        resetSpeechState();
        return;
      }

      if (rms >= AUTO_TURN_SPEECH_START_RMS) {
        lastSpeechAtRef.current = now;

        if (!isSpeechActiveRef.current) {
          isSpeechActiveRef.current = true;
          speechStartedAtRef.current = now;
          speechFrameCountRef.current = 1;
          speechPeakRmsRef.current = rms;
          logVoiceEvent("vad_speech_start", {
            rms: Number(rms.toFixed(4)),
          });
        } else {
          speechFrameCountRef.current += 1;
          speechPeakRmsRef.current = Math.max(speechPeakRmsRef.current, rms);
        }
        return;
      }

      if (!isSpeechActiveRef.current) return;
      if (!lastSpeechAtRef.current || !speechStartedAtRef.current) return;
      if (now - lastSpeechAtRef.current < resolvedConfig.silenceMs) return;

      isSpeechActiveRef.current = false;
      const speechMs = Math.max(0, lastSpeechAtRef.current - speechStartedAtRef.current);
      const speechFrames = speechFrameCountRef.current;
      const speechPeakRms = speechPeakRmsRef.current;
      logVoiceEvent("vad_speech_end", { speech_ms: speechMs });
      resetSpeechState();

      if (speechMs < resolvedConfig.minSpeechMs) return;
      if (speechFrames < AUTO_TURN_MIN_COMMIT_FRAMES) return;
      if (speechPeakRms < AUTO_TURN_SPEECH_START_RMS) return;
      commitTurn("auto_vad");
    },
    [
      clearProcessingFallbackTimer,
      commitTurn,
      logVoiceEvent,
      resetSpeechState,
      resolvedConfig.allowBargeIn,
      resolvedConfig.minSpeechMs,
      resolvedConfig.mode,
      resolvedConfig.silenceMs,
      stopAudioPlayback,
    ],
  );

  useEffect(() => {
    if (!isConnected || introPlayedRef.current) return;
    introPlayedRef.current = true;
    const audio = new Audio(INTRO_AUDIO_PATH);
    audio.play().catch((error: unknown) => console.error("Error playing intro audio:", error));
  }, [isConnected]);

  const playNextChunk = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0 || !audioContextRef.current) return;

    isPlayingRef.current = true;
    const chunk = audioQueueRef.current.shift();
    if (!chunk) {
      isPlayingRef.current = false;
      return;
    }

    try {
      const audioBuffer = await audioContextRef.current.decodeAudioData(chunk);
      const source = audioContextRef.current.createBufferSource();
      playbackSourceRef.current = source;
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);

      source.onended = () => {
        isPlayingRef.current = false;
        playbackSourceRef.current = null;

        if (audioQueueRef.current.length > 0) {
          const play = playNextChunkRef.current;
          if (play) void play();
          return;
        }

        assistantPlaybackEndedAtRef.current = Date.now();
        setStatus(isRecordingRef.current ? "listening" : "idle");
      };

      source.start();
    } catch (error: unknown) {
      console.error("Error decoding audio chunk", error);
      isPlayingRef.current = false;
      playbackSourceRef.current = null;
      const play = playNextChunkRef.current;
      if (play) void play();
    }
  }, []);

  useEffect(() => {
    playNextChunkRef.current = playNextChunk;
  }, [playNextChunk]);

  const stopRecording = useCallback(() => {
    stopSessionCapture();
    commitTurn("manual_stop");
  }, [commitTurn, stopSessionCapture]);

  const disconnect = useCallback(() => {
    clearReconnectTimer();
    clearWsKeepaliveInterval();
    clearProcessingFallbackTimer();
    resetProcessingTracking();
    stopAudioPlayback();
    stopSessionCapture();
    shouldReconnectRef.current = false;
    wsRef.current?.close();
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close();
    }
  }, [
    clearReconnectTimer,
    clearWsKeepaliveInterval,
    clearProcessingFallbackTimer,
    resetProcessingTracking,
    stopAudioPlayback,
    stopSessionCapture,
  ]);

  useEffect(() => {
    if (runtimeWatchdogIntervalRef.current) {
      clearInterval(runtimeWatchdogIntervalRef.current);
    }

    runtimeWatchdogIntervalRef.current = setInterval(() => {
      const now = Date.now();

      if (statusRef.current === "listening" && isRecordingRef.current) {
        const hasActiveTrack = Boolean(
          streamRef.current?.getTracks().some((track) => track.readyState === "live" && track.enabled),
        );

        if (!hasActiveTrack && now - lastListeningWatchdogLogAtRef.current > WATCHDOG_LOG_COOLDOWN_MS) {
          lastListeningWatchdogLogAtRef.current = now;
          logVoiceEvent(
            "unexpected_state_transition",
            {
              state: "listening_without_active_track",
            },
            "error",
          );
        }
      }

      if (
        statusRef.current === "speaking" &&
        !isPlayingRef.current &&
        audioQueueRef.current.length === 0 &&
        !playbackSourceRef.current &&
        now - lastSpeakingWatchdogLogAtRef.current > WATCHDOG_LOG_COOLDOWN_MS
      ) {
        lastSpeakingWatchdogLogAtRef.current = now;
        logVoiceEvent(
          "unexpected_state_transition",
          {
            state: "speaking_without_playback_source",
          },
          "warn",
        );
      }
    }, STATUS_WATCHDOG_INTERVAL_MS);

    return () => {
      if (!runtimeWatchdogIntervalRef.current) return;
      clearInterval(runtimeWatchdogIntervalRef.current);
      runtimeWatchdogIntervalRef.current = null;
    };
  }, [logVoiceEvent]);

  useEffect(() => {
    shouldReconnectRef.current = true;

    const webkitWindow = window as WindowWithWebkitAudio;
    const AudioContextCtor = window.AudioContext || webkitWindow.webkitAudioContext;
    if (!AudioContextCtor) {
      console.error("AudioContext not available in this browser");
      return undefined;
    }
    audioContextRef.current = new AudioContextCtor();

    const wsUrl = resolveWebSocketUrl();

    const connect = () => {
      if (!shouldReconnectRef.current && wsRef.current) return;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (wsRef.current !== ws) return;
        setIsConnected(true);
        clearWsKeepaliveInterval();
        wsKeepaliveIntervalRef.current = setInterval(() => {
          if (wsRef.current !== ws || ws.readyState !== WebSocket.OPEN) return;
          try {
            ws.send("PING");
          } catch {
            // no-op
          }
        }, WS_KEEPALIVE_INTERVAL_MS);
      };

      ws.onmessage = async (event) => {
        if (wsRef.current !== ws) return;
        if (event.data instanceof Blob) {
          clearProcessingFallbackTimer();
          if (awaitingAssistantAudioRef.current) {
            awaitingAssistantAudioRef.current = false;
            clearProcessingWatchdogTimer();
            const latencyMs = processingStartedAtRef.current
              ? Date.now() - processingStartedAtRef.current
              : null;
            logVoiceEvent("assistant_first_audio_chunk", {
              latency_ms: latencyMs,
            });
          }

          const arrayBuffer = await event.data.arrayBuffer();
          audioQueueRef.current.push(arrayBuffer);
          setStatus("speaking");
          void playNextChunk();
          return;
        }

        if (typeof event.data !== "string") return;
        const message = parseTextEvent(event.data);
        if (!message) return;

        if (message.type === "extraction" && message.data) {
          setExtractionData(message.data);
          setLastExtractionUpdateAt(Date.now());
          setIsExtractionFinal(false);
          return;
        }

        if (message.type === "extraction_status" && message.status) {
          if (message.is_final) {
            sessionEndingRef.current = true;
            shouldReconnectRef.current = false;
            clearReconnectTimer();
            clearProcessingFallbackTimer();
            stopSessionCapture();
            setStatus("processing");
            setExtractionStatus("Final extraction locked for report");
            setLastExtractionUpdateAt(Date.now());
            setIsExtractionFinal(true);
          } else if (message.status === "completed") {
            setExtractionStatus("Triage data updated");
            setIsExtractionFinal(false);
          } else if (message.status === "timed_out") {
            setExtractionStatus("Extraction is taking longer than expected");
            setIsExtractionFinal(false);
          } else if (message.status === "running") {
            setExtractionStatus("Updating triage data...");
            setIsExtractionFinal(false);
          }
          return;
        }

        if (message.type === "report_status" && message.status) {
          if (message.status === "running") {
            sessionEndingRef.current = true;
            shouldReconnectRef.current = false;
            clearReconnectTimer();
            clearProcessingFallbackTimer();
            stopSessionCapture();
            setStatus("processing");
          }
          onReportStatus?.(message.status as ReportStatus);
          return;
        }

        if (message.type === "report") {
          reportReceivedRef.current = true;
          reportFailureNotifiedRef.current = false;
          sessionEndingRef.current = true;
          shouldReconnectRef.current = false;
          onReportMessage?.(Boolean(message.success), message.data ?? {}, message.error ?? null);
          wsRef.current?.close();
          return;
        }

        if (message.type === "user" || message.type === "agent") {
          let text = message.text ?? "";
          text = text.replace(/<\/s>/g, "").trim();
          if (text.includes("tool_")) text = "Analyzing your input...";
          if (!text) return;
          const role = message.type === "user" ? "user" : "agent";
          setMessages((previous) => [...previous, { role, text }]);
        }
      };

      ws.onerror = () => {
        if (wsRef.current !== ws) return;
        logVoiceEvent(
          "websocket_error",
          {
            ready_state: ws.readyState,
            session_ending: sessionEndingRef.current,
            should_reconnect: shouldReconnectRef.current,
          },
          "warn",
        );
      };

      ws.onclose = (closeEvent) => {
        if (wsRef.current !== ws) return;
        clearWsKeepaliveInterval();
        logVoiceEvent(
          "websocket_closed",
          {
            code: closeEvent.code,
            reason: closeEvent.reason || "none",
            was_clean: closeEvent.wasClean,
            session_ending: sessionEndingRef.current,
            should_reconnect: shouldReconnectRef.current,
          },
          closeEvent.wasClean ? "info" : "warn",
        );

        if (
          sessionEndingRef.current &&
          !shouldReconnectRef.current &&
          !reportReceivedRef.current &&
          !reportFailureNotifiedRef.current
        ) {
          reportFailureNotifiedRef.current = true;
          onReportMessage?.(false, {}, {
            code: "WEBSOCKET_CLOSED_BEFORE_REPORT",
            message: "Connection closed before the final report arrived. Retry to generate report from transcript history.",
          });
        }

        setIsConnected(false);
        if (shouldReconnectRef.current) {
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            if (!shouldReconnectRef.current) return;
            connect();
          }, 3000);
        }
      };
    };

    connect();

    return () => {
      clearReconnectTimer();
      clearWsKeepaliveInterval();
      clearProcessingFallbackTimer();
      clearProcessingWatchdogTimer();
      shouldReconnectRef.current = false;
      stopAudioPlayback();
      stopSessionCapture();
      wsRef.current?.close();
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        void audioContextRef.current.close();
      }
    };
  }, [
    clearReconnectTimer,
    clearWsKeepaliveInterval,
    clearProcessingFallbackTimer,
    clearProcessingWatchdogTimer,
    commitTurn,
    logVoiceEvent,
    onReportMessage,
    onReportStatus,
    playNextChunk,
    sessionId,
    stopAudioPlayback,
    stopSessionCapture,
  ]);

  const startSessionCapture = useCallback(async () => {
    if (streamRef.current) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Microphone not supported");
      return;
    }

    if (!audioContextRef.current) return;

    try {
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      const source = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !audioContextRef.current) return;

        const inputData = event.inputBuffer.getChannelData(0);
        const downsampled = downsampleBuffer(inputData, audioContextRef.current.sampleRate, 16000);
        const pcmData = convertFloat32ToInt16(downsampled);
        wsRef.current.send(pcmData);

        handleAutoTurnFrame(downsampled);
      };

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      setIsRecording(true);
      clearProcessingFallbackTimer();
      setStatus("listening");
    } catch (error: unknown) {
      console.error("Error accessing microphone", error);
    }
  }, [clearProcessingFallbackTimer, handleAutoTurnFrame]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
      return;
    }
    void startSessionCapture();
  }, [isRecording, startSessionCapture, stopRecording]);

  const resetSessionState = useCallback(() => {
    clearProcessingFallbackTimer();
    resetProcessingTracking();
    stopAudioPlayback();

    setMessages([]);
    setExtractionData(null);
    setExtractionStatus(null);
    setLastExtractionUpdateAt(null);
    setIsExtractionFinal(false);
    setStatus("idle");
    setIsRecording(false);
    setIsConnected(false);
    introPlayedRef.current = false;
    sessionEndingRef.current = false;
    reportReceivedRef.current = false;
    reportFailureNotifiedRef.current = false;
  }, [clearProcessingFallbackTimer, resetProcessingTracking, stopAudioPlayback]);

  return {
    isRecording,
    isConnected,
    interactionMode: resolvedConfig.mode,
    status,
    messages,
    extractionData,
    extractionStatus,
    lastExtractionUpdateAt,
    isExtractionFinal,
    toggleRecording,
    stopRecording,
    disconnect,
    resetSessionState,
  };
}
