"use client";

import { useState, useEffect } from "react";
import { useAuth, UserButton } from "@clerk/nextjs";
import { uploadDataset } from "@/lib/api";
import FileUpload from "@/components/FileUpload";
import StreamingOutput from "@/components/StreamingOutput";

type HistoryEntry = {
  sessionId: string;
  fileName: string;
  timestamp: number;
};

const HISTORY_KEY = "data2model_history";

function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveToHistory(entry: HistoryEntry) {
  const history = loadHistory();
  const updated = [entry, ...history.filter((h) => h.sessionId !== entry.sessionId)].slice(0, 10);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60_000);
  const hrs = Math.floor(diff / 3_600_000);
  const days = Math.floor(diff / 86_400_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (hrs < 24) return `${hrs}h ago`;
  return `${days}d ago`;
}

export default function Home() {
  const { getToken } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [token, setToken] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  async function handleFile(file: File) {
    setError(null);
    setSessionId(null);
    setFileName(file.name);
    setUploading(true);

    try {
      const jwt = await getToken();
      if (!jwt) throw new Error("Not authenticated");
      setToken(jwt);
      const session = await uploadDataset(file, jwt);
      setSessionId(session.sessionId);

      const entry: HistoryEntry = {
        sessionId: session.sessionId,
        fileName: file.name,
        timestamp: Date.now(),
      };
      saveToHistory(entry);
      setHistory(loadHistory());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function handleClearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)" }}>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 32px",
        height: "60px",
        borderBottom: "1px solid var(--border-light)",
        position: "sticky", top: 0, zIndex: 10,
        background: "var(--bg)",
        backdropFilter: "blur(12px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Monogram */}
          <div style={{
            width: 30, height: 30,
            borderRadius: "7px",
            border: "1px solid var(--amber-20)",
            background: "var(--amber-10)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <span style={{
              fontSize: "11px", fontWeight: 700,
              fontFamily: "var(--font-mono)",
              color: "var(--amber)",
              letterSpacing: "-0.03em",
            }}>
              D2
            </span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "1px" }}>
            <span style={{
              fontSize: "14px", fontWeight: 500,
              letterSpacing: "-0.01em",
              color: "var(--text)",
              lineHeight: 1,
            }}>
              Data2Model
            </span>
            <span style={{
              fontSize: "10px",
              fontFamily: "var(--font-mono)",
              color: "var(--text-3)",
              lineHeight: 1,
            }}>
              AI
            </span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{
            fontSize: "11px",
            fontFamily: "var(--font-mono)",
            color: "var(--text-3)",
            display: "none",
          }}
            className="sm:block"
          >
            csv → model → code
          </span>
          <UserButton />
        </div>
      </header>

      {/* Main */}
      <main style={{
        flex: 1,
        maxWidth: "680px",
        width: "100%",
        margin: "0 auto",
        padding: "56px 24px 80px",
        display: "flex",
        flexDirection: "column",
        gap: "32px",
      }}>

        {/* Hero */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{
              display: "inline-flex", alignItems: "center", gap: "5px",
              padding: "3px 9px",
              borderRadius: "20px",
              border: "1px solid var(--amber-20)",
              background: "var(--amber-50)",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
              color: "var(--amber)",
              letterSpacing: "0.04em",
            }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--amber)" }} />
              multi-agent AI
            </span>
          </div>
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(32px, 5vw, 44px)",
            fontWeight: 400,
            color: "var(--text)",
            lineHeight: 1.1,
            letterSpacing: "-0.02em",
            margin: 0,
          }}>
            Upload your dataset,<br />
            <em style={{ color: "var(--amber)" }}>get a model.</em>
          </h1>
          <p style={{
            fontSize: "14px",
            color: "var(--text-2)",
            lineHeight: 1.7,
            maxWidth: "480px",
            margin: 0,
          }}>
            Our AI pipeline analyzes your CSV, recommends the optimal scikit-learn model,
            and streams production-ready code — in seconds.
          </p>
        </div>

        {/* Upload */}
        <FileUpload onFile={handleFile} disabled={uploading} />

        {/* Uploading state */}
        {uploading && (
          <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            fontSize: "13px",
            color: "var(--text-2)",
          }}>
            <svg
              width="14" height="14"
              viewBox="0 0 24 24" fill="none"
              stroke="var(--amber)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={{ animation: "spin 1s linear infinite", flexShrink: 0 }}
            >
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
            Uploading{" "}
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--text)" }}>
              {fileName}
            </span>{" "}
            and starting analysis…
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            display: "flex", alignItems: "flex-start", gap: "10px",
            padding: "14px 18px",
            borderRadius: "9px",
            border: "1px solid var(--red-20)",
            background: "var(--red-10)",
            fontSize: "13px",
            color: "var(--red)",
            lineHeight: 1.6,
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </div>
        )}

        {/* Streaming output */}
        {sessionId && (
          <div className="animate-appear">
            <div style={{
              display: "flex", alignItems: "center", gap: "8px",
              marginBottom: "14px",
            }}>
              <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-3)" }}>
                session
              </span>
              <span style={{
                fontSize: "11px", fontFamily: "var(--font-mono)",
                color: "var(--text-3)",
                padding: "2px 7px",
                borderRadius: "4px",
                border: "1px solid var(--border)",
                background: "var(--bg-surface)",
                letterSpacing: "0.03em",
              }}>
                {sessionId.slice(0, 8)}…
              </span>
            </div>
            <StreamingOutput sessionId={sessionId} token={token} />
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{
              height: "1px",
              background: "var(--border-light)",
            }} />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
                <span style={{ fontSize: "12px", color: "var(--text-3)", letterSpacing: "0.02em" }}>
                  Recent analyses
                </span>
              </div>
              <button
                onClick={handleClearHistory}
                style={{
                  display: "flex", alignItems: "center", gap: "5px",
                  fontSize: "11px",
                  color: "var(--text-3)",
                  background: "none", border: "none", cursor: "pointer",
                  padding: "4px 6px",
                  borderRadius: "5px",
                  transition: "color 0.2s ease",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "var(--red)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "var(--text-3)"; }}
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6l-1 14H6L5 6" />
                  <path d="M10 11v6M14 11v6" />
                  <path d="M9 6V4h6v2" />
                </svg>
                Clear
              </button>
            </div>

            <div style={{
              borderRadius: "10px",
              border: "1px solid var(--border-light)",
              overflow: "hidden",
            }}>
              {history.map((entry, i) => (
                <button
                  key={entry.sessionId}
                  onClick={() => setSessionId(entry.sessionId)}
                  style={{
                    width: "100%",
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "12px 16px",
                    background: entry.sessionId === sessionId ? "var(--amber-50)" : "var(--bg-surface)",
                    borderTop: i > 0 ? "1px solid var(--border-light)" : "none",
                    borderLeft: entry.sessionId === sessionId ? "2px solid var(--amber)" : "2px solid transparent",
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "background 0.15s ease",
                  }}
                  onMouseEnter={(e) => {
                    if (entry.sessionId !== sessionId) {
                      (e.currentTarget as HTMLButtonElement).style.background = "var(--bg-raised)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (entry.sessionId !== sessionId) {
                      (e.currentTarget as HTMLButtonElement).style.background = "var(--bg-surface)";
                    }
                  }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <p style={{
                      fontSize: "13px",
                      fontWeight: 500,
                      color: entry.sessionId === sessionId ? "var(--amber)" : "var(--text)",
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                      marginBottom: "2px",
                    }}>
                      {entry.fileName}
                    </p>
                    <p style={{ fontSize: "11px", color: "var(--text-3)", fontFamily: "var(--font-mono)" }}>
                      {timeAgo(entry.timestamp)}
                    </p>
                  </div>
                  <svg
                    width="14" height="14"
                    viewBox="0 0 24 24" fill="none"
                    stroke={entry.sessionId === sessionId ? "var(--amber)" : "var(--text-3)"}
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    style={{ flexShrink: 0, marginLeft: "12px" }}
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (min-width: 640px) {
          .sm\\:block { display: block !important; }
        }
      `}</style>
    </div>
  );
}
