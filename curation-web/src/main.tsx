import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Run = { id: string; source: string; requirement: string; created: string; status: string; message: string; conversations: number; candidates: number; processed: number; selected_conversations: number; selected_events: number; errors: number; download_ready: boolean };
const REQUIREMENT = "conversations where there is an interruption";
const active = (r: Run) => ["queued", "downloading", "preparing", "curating", "packaging"].includes(r.status);
let token = "";
async function response<T>(r: Response): Promise<T> {
  if (!r.ok) { const data = await r.json(); throw new Error(typeof data.detail === "string" ? data.detail : "Request failed. Check your input and try again."); }
  return r.json();
}
function App() {
  const [file, setFile] = useState<File | null>(null), [url, setUrl] = useState(""), [runs, setRuns] = useState<Run[]>([]);
  const [ready, setReady] = useState(false), [configured, setConfigured] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const running = runs.some(active);
  async function refresh() { setRuns((await response<{items: Run[]}>(await fetch("/api/curation/runs"))).items); }
  useEffect(() => {
    void (async () => { try { const b = await response<{token: string; gemini_configured: boolean}>(await fetch("/api/curation/bootstrap")); token = b.token; setConfigured(b.gemini_configured); await refresh(); setReady(true); } catch (e) { setError(String(e)); } })();
  }, []);
  useEffect(() => { if (!running && !busy) return; const timer = window.setInterval(() => { void refresh().catch(e => setError(String(e))); }, 2000); return () => window.clearInterval(timer); }, [running, busy]);
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError("");
    try {
      if (file && file.size > 512 * 1024 * 1024) throw new Error("Upload a ZIP of at most 512 MB.");
      const headers: Record<string, string> = { "x-repair-token": token };
      if (file) {
        headers["Content-Type"] = "application/zip";
        await response(await fetch(`/api/curation/runs/upload?filename=${encodeURIComponent(file.name)}&requirement=${encodeURIComponent(REQUIREMENT)}`, { method: "POST", headers, body: file }));
      } else {
        headers["Content-Type"] = "application/json";
        await response(await fetch("/api/curation/runs/huggingface", { method: "POST", headers, body: JSON.stringify({url: url.trim(), requirement: REQUIREMENT}) }));
      }
      await refresh();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  return <main>
    <header><h1>Curate</h1><p>A dataset in. Selected conversations out.</p></header>
    <form onSubmit={submit}>
      <label htmlFor="dataset">Upload a dataset</label>
      <input id="dataset" type="file" accept=".zip" disabled={busy || running} onChange={e => {setFile(e.target.files?.[0] || null); if (e.target.files?.[0]) setUrl("");}} />
      <p className="help">ZIP with dataset.jsonl and audio · up to 512 MB. <a href="/api/curation/format">View format</a></p>
      <label htmlFor="link">Or paste a Hugging Face dataset link</label>
      <input id="link" type="url" placeholder="https://huggingface.co/datasets/owner/name" value={url} disabled={busy || running} onChange={e => {setUrl(e.target.value); if (e.target.value) {setFile(null); const input = document.getElementById("dataset") as HTMLInputElement; input.value = "";}}} />
      <p className="help">Supports the upload format and mundo-ai/turn-benchmark-dev. Source audio and timed speaker transcripts are required.</p>
      <label htmlFor="requirements">Requirements</label>
      <textarea id="requirements" value={REQUIREMENT} readOnly rows={2} aria-describedby="fixed" />
      <p id="fixed" className="help">Interruption curation only, for now. Audio and transcript context are sent to Gemini. Results are model-selected, not human-verified.</p>
      {error && <p role="alert" className="error">{error}</p>}
      {ready && !configured && <p role="alert">Gemini is not configured on the server.</p>}
      <button type="submit" disabled={!ready || !configured || busy || running || (!file && !url.trim())}>{busy ? "Submitting…" : running ? "Run in progress…" : "Submit"}</button>
    </form>
    <section aria-labelledby="history"><h2 id="history">History</h2>
      {!runs.length ? <p className="empty">{ready ? "Your runs will appear here." : "Loading…"}</p> : <ol>{runs.map(r => <li key={r.id}>
        <div className="row"><h3>{r.source}</h3><time dateTime={r.created}>{new Date(r.created).toLocaleString()}</time></div>
        <p>{r.requirement}</p>
        <div className="row"><p role={active(r) ? "status" : undefined}>{r.status.replaceAll("_", " ")} · {r.message}{r.candidates > 0 && ` · ${r.processed}/${r.candidates} events`}</p>
        {r.download_ready && <a className="download" href={`/api/curation/runs/${r.id}/download`} download>Download ZIP</a>}</div>
        {r.download_ready && <p className="help">{r.selected_conversations} of {r.conversations} conversations selected · {r.selected_events} matching events{r.errors > 0 ? ` · ${r.errors} analysis errors` : ""}</p>}
      </li>)}</ol>}
    </section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
