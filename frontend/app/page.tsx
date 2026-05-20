"use client";

import { useState, useEffect } from "react";
import { useAuth, UserButton } from "@clerk/nextjs";
import { uploadDataset } from "@/lib/api";
import FileUpload from "@/components/FileUpload";
import StreamingOutput from "@/components/StreamingOutput";
import { Brain, History, ChevronRight, Trash2 } from "lucide-react";

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
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Brain className="h-6 w-6 text-violet-400" />
          <span className="font-semibold text-lg tracking-tight">Data2Model AI</span>
        </div>
        <UserButton />
      </header>

      <main className="flex-1 max-w-3xl w-full mx-auto px-6 py-12 space-y-8">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Upload your dataset</h1>
          <p className="text-gray-400">
            Our multi-agent AI analyzes your CSV, recommends the best ML model, and streams
            production-ready scikit-learn code in real time.
          </p>
        </div>

        <FileUpload onFile={handleFile} disabled={uploading} />

        {uploading && (
          <p className="text-sm text-gray-400 animate-pulse">
            Uploading <span className="text-white">{fileName}</span> and starting analysis…
          </p>
        )}

        {error && (
          <div className="rounded-lg bg-red-950/40 border border-red-800 p-4 text-red-400 text-sm">
            {error}
          </div>
        )}

        {sessionId && (
          <div className="space-y-4">
            <p className="text-xs text-gray-600 font-mono">session: {sessionId}</p>
            <StreamingOutput sessionId={sessionId} token={token} />
          </div>
        )}

        {/* History panel — persisted in localStorage, last 10 analyses */}
        {history.length > 0 && (
          <div className="space-y-3 pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-400">
                <History className="h-4 w-4" />
                Recent analyses
              </div>
              <button
                onClick={handleClearHistory}
                className="flex items-center gap-1 text-xs text-gray-600 hover:text-red-400 transition-colors"
              >
                <Trash2 className="h-3 w-3" />
                Clear
              </button>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-900/40 divide-y divide-gray-800 overflow-hidden">
              {history.map((entry) => (
                <button
                  key={entry.sessionId}
                  onClick={() => setSessionId(entry.sessionId)}
                  className={`w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800/60 transition-colors text-left group ${
                    entry.sessionId === sessionId ? "bg-violet-950/30 border-l-2 border-violet-500" : ""
                  }`}
                >
                  <div className="min-w-0">
                    <p className="text-sm text-gray-200 truncate font-medium">{entry.fileName}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{timeAgo(entry.timestamp)}</p>
                  </div>
                  <ChevronRight className="h-4 w-4 text-gray-600 group-hover:text-gray-400 shrink-0 ml-3 transition-colors" />
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
