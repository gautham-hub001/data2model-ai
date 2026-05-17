"use client";

import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

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

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`
        flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-12 cursor-pointer transition-colors
        ${dragging ? "border-violet-400 bg-violet-950/30" : "border-gray-700 hover:border-gray-500"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <UploadCloud className="h-10 w-10 text-gray-400" />
      <p className="text-sm text-gray-400">
        Drop your <span className="text-violet-400 font-medium">.csv</span> file here, or click to browse
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
        }}
      />
    </div>
  );
}
