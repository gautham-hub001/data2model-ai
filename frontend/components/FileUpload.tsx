"use client";

import { useRef, useState } from "react";

interface FileUploadProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function FileUpload({ onFile, disabled }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith(".csv")) onFile(file);
  }

  const isActive = dragging && !disabled;

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`upload-zone ${isActive ? "dragging" : ""}`}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "20px",
        padding: "52px 32px",
        borderRadius: "12px",
        border: `1px solid ${isActive ? "var(--amber-40)" : "var(--border)"}`,
        background: isActive
          ? "var(--amber-50)"
          : "radial-gradient(ellipse at 50% 0%, rgba(245,166,35,0.03) 0%, transparent 70%), var(--bg-surface)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease",
        boxShadow: isActive ? "var(--amber-glow)" : "none",
      }}
    >
      {/* Corner helpers for ::before on .corner-tr / .corner-bl */}
      <span className="corner-tr" style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
      <span className="corner-bl" style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />

      {/* Upload icon */}
      <div style={{
        width: 52, height: 52,
        borderRadius: "50%",
        border: `1px solid ${isActive ? "var(--amber-40)" : "var(--border)"}`,
        background: isActive ? "var(--amber-10)" : "var(--bg-raised)",
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "all 0.25s ease",
      }}>
        <svg
          width="22" height="22"
          viewBox="0 0 24 24" fill="none"
          stroke={isActive ? "var(--amber)" : "var(--text-3)"}
          strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transition: "stroke 0.25s ease" }}
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>

      {/* Text */}
      <div style={{ textAlign: "center" }}>
        <p style={{
          fontSize: "14px",
          color: isActive ? "var(--amber)" : "var(--text-2)",
          marginBottom: "6px",
          transition: "color 0.25s ease",
        }}>
          {isActive
            ? "Release to upload"
            : <>Drop your <span style={{ color: "var(--amber)", fontFamily: "var(--font-mono)", fontSize: "13px" }}>.csv</span> here, or <span style={{ color: "var(--text)", textDecoration: "underline", textUnderlineOffset: "3px" }}>browse</span></>
          }
        </p>
        <p style={{ fontSize: "12px", color: "var(--text-3)" }}>
          CSV files only
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        style={{ display: "none" }}
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
        }}
      />
    </div>
  );
}
