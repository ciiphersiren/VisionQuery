import type { AskResponse, Meta, Pipeline } from "./types";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  meta: () => req<Meta>("/api/meta"),
  samples: () => req<{ name: string; url: string }[]>("/api/samples"),
  loadSample: (file: string) => req<{ image_id: string; width: number; height: number }>(`/api/samples/${encodeURIComponent(file)}/load`, { method: "POST" }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ image_id: string; width: number; height: number }>("/api/images", { method: "POST", body: fd });
  },
  ask: (image_id: string, query: string, pipeline: Pipeline, count_threshold: number | null, model: string | null) =>
    req<AskResponse>("/api/ask", json({ image_id, query, pipeline, count_threshold, model: model || null })),
  metrics: () => req<Record<string, any>>("/api/metrics"),
};
