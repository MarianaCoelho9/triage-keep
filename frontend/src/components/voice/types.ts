export type AgentRole = "user" | "agent";

export type SessionStatus = "idle" | "listening" | "processing" | "speaking";
export type InteractionMode = "ptt" | "auto_turn";

export type JsonRecord = Record<string, unknown>;

export type ExtractionStatus =
  | "scheduled"
  | "running"
  | "stale_discarded"
  | "completed"
  | "timed_out";

export type ReportStatus = "running" | "completed" | "failed";

export interface TranscriptMessage {
  role: AgentRole;
  text: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: string;
}

export interface WebsocketTextEvent {
  type: AgentRole | "extraction" | "extraction_status" | "report_status" | "report";
  text?: string;
  data?: JsonRecord;
  status?: ExtractionStatus | ReportStatus;
  revision?: number;
  is_final?: boolean;
  success?: boolean;
  error?: ApiError | null;
}

export interface ReportEnvelope {
  success: boolean;
  data: JsonRecord;
  error: ApiError | null;
}

export interface ReportPayload {
  chat_history: Array<{ role: "assistant" | "user"; content: string }>;
  user_input?: string;
}

export interface FhirExportPayload {
  report: JsonRecord;
  include_validation?: boolean;
}

export interface FhirExportEnvelope {
  success: boolean;
  data: JsonRecord;
  error: ApiError | null;
}

export interface ErrorWithStatus extends Error {
  status?: number;
}

export interface WindowWithWebkitAudio extends Window {
  webkitAudioContext?: typeof AudioContext;
}

export interface VoiceSessionConfig {
  mode: InteractionMode;
  silenceMs: number;
  commitCooldownMs: number;
  minSpeechMs: number;
  allowBargeIn: boolean;
}
