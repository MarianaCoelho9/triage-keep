export const REPORT_PATH = "/report";
export const FHIR_REPORT_PATH = "/report/fhir";
export const WS_AUDIO_PATH = "/ws/audio";
export const INTRO_AUDIO_PATH = "/static/intro.wav";

export const resolveWebSocketUrl = (): string => {
  const overrideOrigin = process.env.NEXT_PUBLIC_BACKEND_ORIGIN;
  if (overrideOrigin) {
    const wsOrigin = overrideOrigin.replace(/^http/, "ws");
    return `${wsOrigin}${WS_AUDIO_PATH}`;
  }

  // Local dev fallback: frontend runs on :3000 while backend websocket runs on :8000.
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${wsScheme}://127.0.0.1:8000${WS_AUDIO_PATH}`;
  }

  const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${wsScheme}://${window.location.host}${WS_AUDIO_PATH}`;
};
