"use client";

import { useEffect, useRef, useState } from "react";
import { createSessionClient, StreamChunk } from "@/lib/websocket";
import { confirmSmote } from "@/lib/api";
import { CheckCircle, AlertCircle, Loader2 } from "lucide-react";

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

export default function StreamingOutput({ sessionId, token }: StreamingOutputProps) {
  const [steps, setSteps] = useState<Record<string, Step>>({});
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [awaitingSmote, setAwaitingSmote] = useState(false);
  const clientRef = useRef<ReturnType<typeof createSessionClient> | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Ref so the STOMP subscription always calls the latest handleChunk (fixes stale closure)
  const handleChunkRef = useRef<(chunk: StreamChunk) => void>(() => {});
  // Ref so token handler always sees the latest currentStep without re-creating the subscription
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

  // Keep the ref pointing at the latest version of handleChunk on every render
  handleChunkRef.current = handleChunk;

  function handleChunk(chunk: StreamChunk) {
    if (chunk.type === "step" && chunk.step) {
      currentStepRef.current = chunk.step;
      setCurrentStep(chunk.step);
      setSteps((prev) => ({
        ...prev,
        [chunk.step!]: { name: chunk.step!, label: STEP_LABELS[chunk.step!] ?? chunk.step!, content: "", done: false },
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
      {Object.values(steps).map((step) => (
        <div key={step.name} className="rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-800 bg-gray-900">
            {step.done ? (
              <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0" />
            ) : (
              <Loader2 className="h-4 w-4 text-violet-400 animate-spin shrink-0" />
            )}
            <span className="text-sm font-medium text-gray-200">{step.label}</span>
          </div>
          <div className="p-4">
            {step.name === "CODE_GENERATION" ? (
              <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto">
                <code>{step.content}</code>
              </pre>
            ) : (
              <p className="text-sm text-gray-300 whitespace-pre-wrap">{step.content}</p>
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
      ))}

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
