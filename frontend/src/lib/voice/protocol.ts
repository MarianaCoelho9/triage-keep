import { type WebsocketTextEvent } from "@/components/voice/types";

export const parseTextEvent = (raw: string): WebsocketTextEvent | null => {
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    return parsed as WebsocketTextEvent;
  } catch {
    return null;
  }
};
