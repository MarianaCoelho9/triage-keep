import React, { type RefObject } from "react";
import { Volume2 } from "lucide-react";
import { type TranscriptMessage } from "@/components/voice/types";

interface TranscriptPanelProps {
  messages: TranscriptMessage[];
  bottomRef: RefObject<HTMLDivElement | null>;
}

export default function TranscriptPanel({ messages, bottomRef }: TranscriptPanelProps) {
  return (
    <div className="flex-1 flex flex-col bg-white relative">
      <div className="absolute top-0 left-0 right-0 h-24 bg-gradient-to-b from-white via-white to-transparent z-10 pointer-events-none"></div>

      <div className="flex-1 overflow-y-auto p-8 pt-24 pb-32 space-y-8 scroll-smooth">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-zinc-300 space-y-4 opacity-0 animate-fadeIn" style={{ opacity: 1 }}>
            <div className="w-16 h-16 rounded-2xl bg-zinc-50 border border-zinc-100 flex items-center justify-center">
              <Volume2 size={32} className="text-zinc-200" />
            </div>
            <p className="font-medium">Conversation Empty</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex w-full ${message.role === "user" ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-4 duration-500`}
          >
            <div className={`flex max-w-[85%] md:max-w-[75%] gap-4 ${message.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm
                  ${message.role === "user" ? "bg-zinc-800 text-white" : "bg-white border border-indigo-100 text-indigo-600"}`}
              >
                {message.role === "user" ? <span className="text-xs font-bold">ME</span> : <Volume2 size={18} />}
              </div>
              <div
                className={`p-5 rounded-2xl text-[15px] leading-relaxed shadow-sm transition-all hover:shadow-md
                  ${
                    message.role === "user"
                      ? "bg-zinc-50 text-zinc-800 border border-zinc-100 rounded-tr-none"
                      : "bg-indigo-600 text-white shadow-indigo-200 rounded-tl-none"
                  }`}
              >
                {message.text}
              </div>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/80 backdrop-blur-md border-t border-zinc-50 text-center">
        <p className="text-[10px] text-zinc-300 uppercase tracking-widest font-semibold">Processed by MedGemma</p>
      </div>
    </div>
  );
}
