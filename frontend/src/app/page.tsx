import VoiceAgent from "@/components/VoiceAgent";
import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4 md:p-12 bg-zinc-50 text-zinc-900 font-sans">
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 via-white to-sky-50 -z-10"></div>
      
      <main className="flex flex-col items-center w-full max-w-[1400px] gap-6 z-10">
        <div className="flex flex-col items-center text-center space-y-4 mb-8">
          <h1 className="flex items-center gap-3 text-4xl md:text-5xl font-bold bg-gradient-to-r from-zinc-900 to-zinc-600 bg-clip-text text-transparent tracking-tight pb-2">
            <span className="inline-flex h-14 w-14 items-center justify-center">
              <Image src="/med_icon.png" alt="Med icon" width={48} height={48} className="h-12 w-12 object-contain" />
            </span>
            AI Triage System
          </h1>
          <p className="text-zinc-500 text-lg max-w-3xl leading-relaxed">
            AI-Assisted Emergency Triage System - Voice-enabled interaction for rapid patient assessment
          </p>
        </div>
        
        <VoiceAgent />
      </main>
    </div>
  );
}
