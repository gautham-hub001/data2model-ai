const API_URL = process.env.NEXT_PUBLIC_JAVA_API_URL ?? "http://localhost:8080";

export type AnalysisSession = {
  sessionId: string;
  datasetName: string;
};

export async function uploadDataset(
  file: File,
  token: string
): Promise<AnalysisSession> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed: ${res.status}`);
  }

  return res.json();
}

export async function getSession(sessionId: string, token: string) {
  const res = await fetch(`${API_URL}/api/session/${sessionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Session fetch failed: ${res.status}`);
  return res.json();
}

export async function confirmSmote(
  sessionId: string,
  applySmote: boolean,
  token: string
) {
  const res = await fetch(`${API_URL}/api/session/${sessionId}/smote`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ applySmote }),
  });
  if (!res.ok) throw new Error(`SMOTE confirm failed: ${res.status}`);
  return res.json();
}
