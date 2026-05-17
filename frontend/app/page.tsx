"use client";

import { useState } from "react";
import { useAuth, UserButton } from "@clerk/nextjs";
import { uploadDataset } from "@/lib/api";
import FileUpload from "@/components/FileUpload";
import StreamingOutput from "@/components/StreamingOutput";
import { Brain } from "lucide-react";

export default function Home() {
  const { getToken } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [token, setToken] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

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
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
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
            Our multi-agent AI analyzes your CSV, recommends the best ML model, and streams production-ready scikit-learn code in real time.
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
            <p className="text-xs text-gray-500 font-mono">session: {sessionId}</p>
            <StreamingOutput sessionId={sessionId} token={token} />
          </div>
        )}
      </main>
    </div>
  );
}
