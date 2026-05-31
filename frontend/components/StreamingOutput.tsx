"use client";

import { useEffect, useRef, useState } from "react";
import { createSessionClient, StreamChunk } from "@/lib/websocket";

interface StreamingOutputProps {
  sessionId: string;
  token: string;
}

type Step = {
  name: string;
  label: string;
  sublabel: string;
  content: string;
  done: boolean;
};

const STEP_META: Record<string, { label: string; sublabel: string }> = {
  ANALYSIS:        { label: "Dataset Analysis",      sublabel: "Inspecting shape, types & distributions" },
  RECOMMENDATION:  { label: "Model Recommendation",  sublabel: "Selecting the optimal algorithm" },
  CLARIFICATION:   { label: "Clarification",         sublabel: "Checking for class imbalance" },
  CODE_GENERATION: { label: "Code Generation",       sublabel: "Writing production-ready scikit-learn" },
};

const STEP_ORDER = ["ANALYSIS", "RECOMMENDATION", "CLARIFICATION", "CODE_GENERATION"];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
      }}
      style={{
        display: "flex", alignItems: "center", gap: "6px",
        padding: "5px 10px",
        borderRadius: "6px",
        border: `1px solid ${copied ? "var(--emerald-20)" : "var(--border)"}`,
        background: copied ? "var(--emerald-10)" : "var(--bg-raised)",
        color: copied ? "var(--emerald)" : "var(--text-3)",
        fontSize: "11px",
        fontFamily: "var(--font-mono)",
        cursor: "pointer",
        transition: "all 0.2s ease",
        flexShrink: 0,
      }}
    >
      {copied ? (
        <>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          copied
        </>
      ) : (
        <>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          copy
        </>
      )}
    </button>
  );
}

function StepIndicator({ index, done, active }: { index: number; done: boolean; active: boolean }) {
  if (done) {
    return (
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: "var(--emerald-10)",
        border: "1px solid var(--emerald-20)",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
    );
  }
  if (active) {
    return (
      <div style={{ position: "relative", width: 28, height: 28, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div className="animate-ping-amber" style={{
          position: "absolute",
          width: 12, height: 12,
          borderRadius: "50%",
          background: "var(--amber-20)",
        }} />
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: "var(--amber)",
          boxShadow: "0 0 8px 1px var(--amber-40)",
        }} />
      </div>
    );
  }
  return (
    <div style={{
      width: 28, height: 28, borderRadius: "50%",
      border: "1px solid var(--border)",
      display: "flex", alignItems: "center", justifyContent: "center",
      flexShrink: 0,
    }}>
      <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-4)" }}>
        {String(index + 1).padStart(2, "0")}
      </span>
    </div>
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
      const meta = STEP_META[chunk.step] ?? { label: chunk.step, sublabel: "" };
      setSteps((p) => ({
        ...p,
        [chunk.step!]: { name: chunk.step!, ...meta, content: "", done: false },
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
      <div className="animate-appear" style={{
        display: "flex", alignItems: "flex-start", gap: "12px",
        padding: "16px 20px",
        borderRadius: "10px",
        border: "1px solid var(--red-20)",
        background: "var(--red-10)",
        color: "var(--red)",
        fontSize: "13px",
        lineHeight: 1.6,
      }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        {error}
      </div>
    );
  }

  const visibleSteps = STEP_ORDER.filter((s) => steps[s]).map((s) => steps[s]);
  const globalIndex = (name: string) => STEP_ORDER.indexOf(name);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      {visibleSteps.map((step) => {
        const idx = globalIndex(step.name);
        const isActive = step.name === currentStep && !step.done;
        const isCode = step.name === "CODE_GENERATION";

        return (
          <div
            key={step.name}
            className={`animate-appear ${isActive ? "step-active-scanline animate-pulse-shadow" : ""}`}
            style={{
              borderRadius: "10px",
              border: `1px solid ${
                isActive ? "var(--amber-20)"
                : step.done ? "var(--border-light)"
                : "var(--border-light)"
              }`,
              background: isActive ? "var(--amber-50)" : "var(--bg-surface)",
              overflow: "hidden",
              transition: "border-color 0.3s ease, background 0.3s ease",
              position: "relative",
            }}
          >
            {/* Amber left accent bar on active */}
            {isActive && (
              <div style={{
                position: "absolute", left: 0, top: 0, bottom: 0,
                width: "2px",
                background: "var(--amber)",
                borderRadius: "10px 0 0 10px",
              }} />
            )}

            {/* Large decorative step number */}
            <div style={{
              position: "absolute", right: 16, top: "50%",
              transform: "translateY(-50%)",
              fontSize: "80px", fontWeight: 700,
              fontFamily: "var(--font-display)",
              color: step.done ? "rgba(255,255,255,0.025)" : isActive ? "var(--amber-10)" : "rgba(255,255,255,0.018)",
              lineHeight: 1,
              pointerEvents: "none",
              userSelect: "none",
              transition: "color 0.3s ease",
            }}>
              {String(idx + 1).padStart(2, "0")}
            </div>

            {/* Header */}
            <div style={{
              display: "flex", alignItems: "center", gap: "12px",
              padding: "14px 20px",
              position: "relative",
            }}>
              <StepIndicator index={idx} done={step.done} active={isActive} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{
                  fontSize: "13px",
                  fontWeight: 500,
                  color: isActive ? "var(--text)" : step.done ? "var(--text-2)" : "var(--text-3)",
                  transition: "color 0.3s ease",
                  letterSpacing: "0.01em",
                }}>
                  {step.label}
                </p>
                {isActive && (
                  <p className="animate-fade-in" style={{ fontSize: "11px", color: "var(--amber)", marginTop: "2px", fontFamily: "var(--font-mono)" }}>
                    processing…
                  </p>
                )}
                {step.done && (
                  <p className="animate-fade-in" style={{ fontSize: "11px", color: "var(--text-3)", marginTop: "2px" }}>
                    {step.sublabel}
                  </p>
                )}
              </div>
              {isCode && step.content && <CopyButton text={step.content} />}
            </div>

            {/* Content */}
            {step.content && (
              <div style={{
                borderTop: `1px solid ${isActive ? "var(--amber-10)" : "var(--border-light)"}`,
                transition: "border-color 0.3s ease",
              }}>
                {isCode ? (
                  <>
                    {/* Terminal chrome */}
                    <div style={{
                      display: "flex", alignItems: "center", gap: "6px",
                      padding: "8px 16px",
                      background: "rgba(0,0,0,0.3)",
                      borderBottom: "1px solid var(--border-light)",
                    }}>
                      <span style={{ width: 9, height: 9, borderRadius: "50%", background: "rgba(248,113,113,0.5)" }} />
                      <span style={{ width: 9, height: 9, borderRadius: "50%", background: "rgba(251,191,36,0.5)" }} />
                      <span style={{ width: 9, height: 9, borderRadius: "50%", background: "rgba(52,211,153,0.5)" }} />
                      <span style={{ marginLeft: 8, fontSize: "11px", color: "var(--text-3)", fontFamily: "var(--font-mono)" }}>
                        pipeline.py
                      </span>
                    </div>
                    <pre style={{
                      padding: "20px",
                      fontSize: "12.5px",
                      fontFamily: "var(--font-mono)",
                      color: "var(--cyan)",
                      whiteSpace: "pre-wrap",
                      overflowX: "auto",
                      lineHeight: 1.7,
                      background: "rgba(0,0,0,0.2)",
                      margin: 0,
                    }}>
                      <code>{step.content}</code>
                    </pre>
                  </>
                ) : (
                  <p style={{
                    padding: "16px 20px",
                    fontSize: "13px",
                    color: "var(--text-2)",
                    lineHeight: 1.75,
                    whiteSpace: "pre-wrap",
                  }}>
                    {step.content}
                  </p>
                )}
              </div>
            )}

            {/* SMOTE clarification */}
            {step.name === "CLARIFICATION" && awaitingSmote && (
              <div style={{
                display: "flex", gap: "10px",
                padding: "14px 20px",
                borderTop: "1px solid var(--border-light)",
              }}>
                <button
                  onClick={async () => {
                    setAwaitingSmote(false);
                    await import("@/lib/api").then((m) => m.confirmSmote(sessionId, true, token));
                  }}
                  style={{
                    padding: "8px 18px",
                    borderRadius: "7px",
                    border: "1px solid var(--amber-40)",
                    background: "var(--amber-10)",
                    color: "var(--amber)",
                    fontSize: "13px",
                    fontWeight: 500,
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background = "var(--amber-20)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background = "var(--amber-10)";
                  }}
                >
                  Apply SMOTE
                </button>
                <button
                  onClick={async () => {
                    setAwaitingSmote(false);
                    await import("@/lib/api").then((m) => m.confirmSmote(sessionId, false, token));
                  }}
                  style={{
                    padding: "8px 18px",
                    borderRadius: "7px",
                    border: "1px solid var(--border)",
                    background: "var(--bg-raised)",
                    color: "var(--text-2)",
                    fontSize: "13px",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--text-3)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)";
                  }}
                >
                  Skip
                </button>
              </div>
            )}
          </div>
        );
      })}

      {done && (
        <div className="animate-appear" style={{
          display: "flex", alignItems: "center", gap: "12px",
          padding: "14px 20px",
          borderRadius: "10px",
          border: "1px solid var(--emerald-20)",
          background: "var(--emerald-10)",
        }}>
          <div style={{
            width: 28, height: 28, borderRadius: "50%",
            border: "1px solid var(--emerald-20)",
            background: "var(--emerald-10)",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0,
          }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </div>
          <div>
            <p style={{ fontSize: "13px", fontWeight: 500, color: "var(--emerald)" }}>Analysis complete</p>
            <p style={{ fontSize: "11px", color: "var(--text-3)", marginTop: "2px" }}>Your scikit-learn pipeline is ready to use</p>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
