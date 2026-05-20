"use client";

import { useEffect, useRef, useState } from "react";
import { createSessionClient, StreamChunk } from "@/lib/websocket";
import { confirmSmote } from "@/lib/api";
import { Check, Copy, AlertCircle } from "lucide-react";

interface StreamingOutputProps {
  sessionId: string;
  token: string;
}

type Step = {
  name: string;
  label: string;
  content: string;
  done: boolean;
};

const STEP_LABELS: Record<string, string> = {
  ANALYSIS:        "Dataset Analysis",
  RECOMMENDATION:  "Model Recommendation",
  CLARIFICATION:   "Clarification",
  CODE_GENERATION: "Code Generation",
};

const STEP_ORDER = ["ANALYSIS", "RECOMMENDATION", "CLARIFICATION", "CODE_GENERATION"];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
      }}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium
                 bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white
                 border border-white/10 transition-all"
    >
      {copied
        ? <><Check className="h-3 w-3 text-emerald-400" /> Copied</>
        : <><Copy className="h-3 w-3" /> Copy code</>}
    </button>
  );
}

function StatusDot({ done, active }: { done: boolean; active: boolean }) {
  if (done) return (
    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/15 ring-1 ring-emerald-500/40">
      <Check className="h-3 w-3 text-emerald-400" />
    </span>
  );
  if (active) return (
    <span className="relative flex h-5 w-5 items-center justify-center">
      <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-violet-400 opacity-40" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-400" />
    </span>
  );
  return <span className="h-2 w-2 rounded-full bg-gray-700 mx-1.5" />;
}

export default function StreamingOutput({ sessionId, token }: StreamingOutputProps) {
  const [steps, setSteps] = useState<Record<string, Step>>({});
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [awaitingSmote, setAwaitingSmote] = useState(false);
  const clientRef = useRef<ReturnType<typeof createSessionClient> | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const handleChunkRef = useRef<(chunk: StreamChunk) => void>(() => {});
  const currentStepRef = useRef<string | null>(null);

  useEffect(() => {
    clientRef.current = createSessionClient(sessionId, (chunk) => handleChunkRef.current(chunk));
    return () => { clientRef.current?.deactivate(); };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps, currentStep]);

  handleChunkRef.current = handleChunk;

  function handleChunk(chunk: StreamChunk) {
    if (chunk.type === "step" && chunk.step) {
      const prev = currentStepRef.current;
      if (prev) setSteps((p) => ({ ...p, [prev]: { ...p[prev], done: true } }));
      currentStepRef.current = chunk.step;
      setCurrentStep(chunk.step);
      setSteps((p) => ({
        ...p,
        [chunk.step!]: { name: chunk.step!, label: STEP_LABELS[chunk.step!] ?? chunk.step!, content: "", done: false },
      }));
      if (chunk.step === "CLARIFICATION") setAwaitingSmote(true);
    } else if (chunk.type === "token" && currentStepRef.current) {
      const step = currentStepRef.current;
      setSteps((p) => ({ ...p, [step]: { ...p[step], content: (p[step]?.content ?? "") + chunk.content } }));
    } else if (chunk.type === "done") {
      const step = currentStepRef.current;
      if (step) setSteps((p) => ({ ...p, [step]: { ...p[step], done: true } }));
      setDone(true);
    } else if (chunk.type === "error") {
      setError(chunk.content);
    }
  }

  if (error) {
    return (
      <div className="flex items-start gap-3 rounded-xl bg-red-500/5 border border-red-500/20 p-4 text-red-400">
        <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
        <span className="text-sm leading-relaxed">{error}</span>
      </div>
    );
  }

  // Show steps in fixed order, only ones we've received
  const visibleSteps = STEP_ORDER.filter((s) => steps[s]).map((s) => steps[s]);

  return (
    <div className="space-y-3">
      {visibleSteps.map((step, i) => {
        const isActive = step.name === currentStep && !step.done;
        const isCode = step.name === "CODE_GENERATION";

        return (
          <div
            key={step.name}
            className={`rounded-2xl border transition-all duration-300 overflow-hidden
              ${step.done
                ? "border-white/8 bg-white/[0.02]"
                : isActive
                  ? "border-violet-500/30 bg-violet-500/[0.04] shadow-[0_0_24px_-8px_rgba(139,92,246,0.3)]"
                  : "border-white/5 bg-white/[0.01]"
              }`}
          >
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-3.5">
              <StatusDot done={step.done} active={isActive} />
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className="text-xs font-medium text-gray-600 tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className={`text-sm font-medium truncate ${
                  step.done ? "text-gray-300" : isActive ? "text-white" : "text-gray-500"
                }`}>
                  {step.label}
                </span>
              </div>
              {isCode && step.content && <CopyButton text={step.content} />}
            </div>

            {/* Content */}
            {step.content && (
              <div className={`border-t ${isActive ? "border-violet-500/20" : "border-white/5"}`}>
                {isCode ? (
                  <div>
                    {/* Terminal chrome */}
                    <div className="flex items-center gap-1.5 px-4 py-2.5 bg-black/40">
                      <span className="h-2.5 w-2.5 rounded-full bg-red-500/60" />
                      <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/60" />
                      <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/60" />
                      <span className="ml-2 text-xs text-gray-600 font-mono">pipeline.py</span>
                    </div>
                    <pre className="px-5 py-4 text-[13px] text-emerald-300/90 font-mono
                                   whitespace-pre-wrap overflow-x-auto leading-relaxed bg-black/30">
                      <code>{step.content}</code>
                    </pre>
                  </div>
                ) : (
                  <p className="px-5 py-4 text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
                    {step.content}
                  </p>
                )}
              </div>
            )}

            {/* SMOTE clarification buttons */}
            {step.name === "CLARIFICATION" && awaitingSmote && (
              <div className={`px-5 pb-5 flex gap-3 border-t ${isActive ? "border-violet-500/20" : "border-white/5"} pt-4`}>
                <button
                  onClick={async () => { setAwaitingSmote(false); await import("@/lib/api").then(m => m.confirmSmote(sessionId, true, token)); }}
                  className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-sm font-medium transition-colors"
                >
                  Apply SMOTE
                </button>
                <button
                  onClick={async () => { setAwaitingSmote(false); await import("@/lib/api").then(m => m.confirmSmote(sessionId, false, token)); }}
                  className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm font-medium text-gray-300 transition-colors"
                >
                  Skip
                </button>
              </div>
            )}
          </div>
        );
      })}

      {done && (
        <div className="flex items-center gap-2.5 rounded-xl bg-emerald-500/5 border border-emerald-500/20 px-5 py-3.5 text-emerald-400">
          <Check className="h-4 w-4 shrink-0" />
          <span className="text-sm font-medium">Analysis complete</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
