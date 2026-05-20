"use client";

import { useEffect, useRef, useState } from "react";
import { createSessionClient, StreamChunk } from "@/lib/websocket";
import { confirmSmote } from "@/lib/api";
import { CheckCircle, AlertCircle, Loader2, Copy, Check } from "lucide-react";

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
  ANALYSIS: "1. Dataset Analysis",
  RECOMMENDATION: "2. Model Recommendation",
  CLARIFICATION: "3. Clarification",
  CODE_GENERATION: "4. Code Generation",
};

const STEP_COLORS: Record<string, { border: string; header: string; icon: string }> = {
  ANALYSIS:       { border: "border-blue-800",   header: "bg-blue-950/60",   icon: "text-blue-400" },
  RECOMMENDATION: { border: "border-violet-800", header: "bg-violet-950/60", icon: "text-violet-400" },
  CLARIFICATION:  { border: "border-amber-800",  header: "bg-amber-950/60",  icon: "text-amber-400" },
  CODE_GENERATION:{ border: "border-emerald-800",header: "bg-emerald-950/60",icon: "text-emerald-400" },
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button
      onClick={copy}
      className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
    >
      {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
      {copied ? "Copied!" : "Copy"}
    </button>
  );
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
    clientRef.current = createSessionClient(
      sessionId,
      (chunk) => handleChunkRef.current(chunk)
    );
    return () => { clientRef.current?.deactivate(); };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps, currentStep]);

  handleChunkRef.current = handleChunk;

  function handleChunk(chunk: StreamChunk) {
    if (chunk.type === "step" && chunk.step) {
      // Mark the previous step as done when the next one starts
      const prevStep = currentStepRef.current;
      if (prevStep) {
        setSteps((prev) => ({
          ...prev,
          [prevStep]: { ...prev[prevStep], done: true },
        }));
      }
      currentStepRef.current = chunk.step;
      setCurrentStep(chunk.step);
      setSteps((prev) => ({
        ...prev,
        [chunk.step!]: {
          name: chunk.step!,
          label: STEP_LABELS[chunk.step!] ?? chunk.step!,
          content: "",
          done: false,
        },
      }));
      if (chunk.step === "CLARIFICATION") setAwaitingSmote(true);
    } else if (chunk.type === "token" && currentStepRef.current) {
      const step = currentStepRef.current;
      setSteps((prev) => ({
        ...prev,
        [step]: { ...prev[step], content: (prev[step]?.content ?? "") + chunk.content },
      }));
    } else if (chunk.type === "done") {
      const step = currentStepRef.current;
      if (step) {
        setSteps((prev) => ({ ...prev, [step]: { ...prev[step], done: true } }));
      }
      setDone(true);
    } else if (chunk.type === "error") {
      setError(chunk.content);
    }
  }

  async function handleSmoteChoice(apply: boolean) {
    setAwaitingSmote(false);
    await confirmSmote(sessionId, apply, token);
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-red-950/40 border border-red-800 p-4 text-red-400">
        <AlertCircle className="h-5 w-5 shrink-0" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {Object.values(steps).map((step) => {
        const colors = STEP_COLORS[step.name] ?? { border: "border-gray-800", header: "bg-gray-900", icon: "text-violet-400" };
        return (
          <div key={step.name} className={`rounded-xl border ${colors.border} bg-gray-900/60 overflow-hidden`}>
            <div className={`flex items-center gap-2 px-4 py-3 border-b ${colors.border} ${colors.header}`}>
              {step.done ? (
                <CheckCircle className={`h-4 w-4 ${colors.icon} shrink-0`} />
              ) : (
                <Loader2 className={`h-4 w-4 ${colors.icon} animate-spin shrink-0`} />
              )}
              <span className="text-sm font-medium text-gray-100 flex-1">{step.label}</span>
              {step.name === "CODE_GENERATION" && step.content && (
                <CopyButton text={step.content} />
              )}
            </div>
            <div className="p-4">
              {step.name === "CODE_GENERATION" ? (
                <pre className="text-xs text-emerald-300 font-mono whitespace-pre-wrap overflow-x-auto leading-relaxed">
                  <code>{step.content}</code>
                </pre>
              ) : (
                <p className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">{step.content}</p>
              )}
            </div>

            {step.name === "CLARIFICATION" && awaitingSmote && (
              <div className="px-4 pb-4 flex gap-3">
                <button
                  onClick={() => handleSmoteChoice(true)}
                  className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-sm font-medium transition-colors"
                >
                  Yes, apply SMOTE
                </button>
                <button
                  onClick={() => handleSmoteChoice(false)}
                  className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm font-medium transition-colors"
                >
                  No, continue as-is
                </button>
              </div>
            )}
          </div>
        );
      })}

      {done && (
        <div className="flex items-center gap-2 rounded-lg bg-emerald-950/40 border border-emerald-800 p-4 text-emerald-400">
          <CheckCircle className="h-5 w-5 shrink-0" />
          <span className="text-sm font-medium">Analysis complete</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
