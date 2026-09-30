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
type Feedback = {text: string; updated: string | null};
type Detail = {run: Run; feedback: Feedback; offset: number; limit: number; total: number; items: {id: string; conversation_id: string; target_turn_index: number; roles_reversed: boolean; context_truncated: boolean; answer: {evidence_note: string}; evidence: {clip_start_s: number; clip_end_s: number}; turns: {index: number; role: string; text: string; start_s: number | null; end_s: number | null}[]}[]};
function RunDetail({run, close}: {run: Run; close: () => void}) {
  const [detail, setDetail] = useState<Detail | null>(null), [offset, setOffset] = useState(0);
  const [feedback, setFeedback] = useState(""), [saved, setSaved] = useState("");
  const [error, setError] = useState(""), [saving, setSaving] = useState(false), [loaded, setLoaded] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setDetail(null); setError("");
    void (async () => { try {
      const value = await response<Detail>(await fetch(`/api/curation/runs/${run.id}?offset=${offset}&limit=10`, {signal: controller.signal}));
      setDetail(value);
      if (!loaded) { setFeedback(value.feedback.text); setLoaded(true); }
    } catch (e) { if (!controller.signal.aborted) setError(String(e)); } })();
    return () => controller.abort();
  }, [run.id, run.download_ready, offset]);
  async function save(e: React.FormEvent) {
    e.preventDefault(); setSaving(true); setSaved(""); setError("");
    try {
      await response<Feedback>(await fetch(`/api/curation/runs/${run.id}/feedback`, {method: "PUT", headers: {"Content-Type": "application/json", "x-repair-token": token}, body: JSON.stringify({text: feedback})}));
      setSaved("Feedback saved.");
    } catch (e) {setError(String(e));} finally {setSaving(false);}
  }
  return <section className="run-detail" aria-label="Run details">
    <button className="text-button" onClick={close}>Close details</button>
    <p className="help">Model-selected examples, not human-verified. Feedback is saved with this run; it does not change the export or retrain the model.</p>
    {error && <p role="alert" className="error">{error}</p>}
    {!detail ? (!error && <p role="status">Loading details…</p>) : <>
      {!detail.run.download_ready ? <p>{active(run) ? "Examples will appear when this run finishes." : "No completed selection is available for this run."}</p> : detail.total === 0 ? <p>No datapoints were selected.</p> : <>
        <p>{offset + 1}–{Math.min(offset + detail.limit, detail.total)} of {detail.total} selected events. Each player contains the event clip.</p>
        {detail.items.map(item => <article className="datapoint" key={item.id}>
          <h3>Conversation {item.conversation_id} · Event {item.id}</h3>
          <p className="help">{item.evidence.clip_start_s.toFixed(1)}–{item.evidence.clip_end_s.toFixed(1)} seconds in source audio{item.roles_reversed ? " · Opposite speaker direction" : ""}</p>
          <audio controls preload="none" aria-label={`Listen to event ${item.id}`} src={`/api/curation/runs/${run.id}/clips/${item.id}`} />
          <p>{item.answer.evidence_note}</p>
          <details><summary>Transcript context</summary>
            <p className="help">Times are relative to the source recording. Speaker labels retain the source roles. The highlighted turn is the selected interruption.</p>
            {item.turns.map(t => <p key={t.index} className={t.index === item.target_turn_index ? "target-turn" : ""}><strong>{t.role}</strong>{t.start_s != null && ` (${t.start_s.toFixed(1)}s)`}: {t.text}</p>)}
            {item.context_truncated && <p className="help">Context shortened for preview. Download the ZIP for full transcripts.</p>}
          </details>
        </article>)}
        <nav className="pagination" aria-label="Selected event pages"><button onClick={() => setOffset(Math.max(0, offset - 10))} disabled={offset === 0}>Previous</button><button onClick={() => setOffset(offset + 10)} disabled={offset + detail.limit >= detail.total}>Next</button></nav>
      </>}
    </>}
    <form onSubmit={save}>
      <label htmlFor={`feedback-${run.id}`}>Feedback on this curation</label>
      <textarea id={`feedback-${run.id}`} rows={4} maxLength={10000} value={feedback} disabled={!loaded || saving} placeholder="What was useful? What was missed or incorrectly selected? Include an event ID if helpful." onChange={e => {setFeedback(e.target.value); setSaved("");}} />
      <button type="submit" disabled={!loaded || saving}>{saving ? "Saving…" : "Save feedback"}</button>
      {saved && <p role="status">{saved}</p>}
    </form>
  </section>;
}
function App() {
  const [file, setFile] = useState<File | null>(null), [url, setUrl] = useState(""), [runs, setRuns] = useState<Run[]>([]);
  const [ready, setReady] = useState(false), [configured, setConfigured] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
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
        <div className="row"><h3><button className="history-title" aria-expanded={selected === r.id} onClick={() => setSelected(selected === r.id ? null : r.id)}>{r.source}</button></h3><time dateTime={r.created}>{new Date(r.created).toLocaleString()}</time></div>
        <p>{r.requirement}</p>
        <div className="row"><p role={active(r) ? "status" : undefined}>{r.status.replaceAll("_", " ")} · {r.message}{r.candidates > 0 && ` · ${r.processed}/${r.candidates} events`}</p>
        {r.download_ready && <a className="download" href={`/api/curation/runs/${r.id}/download`} download>Download ZIP</a>}</div>
        {r.download_ready && <p className="help">{r.selected_conversations} of {r.conversations} conversations selected · {r.selected_events} matching events{r.errors > 0 ? ` · ${r.errors} analysis errors` : ""}</p>}
        {selected === r.id && <RunDetail key={r.id} run={r} close={() => setSelected(null)} />}
      </li>)}</ol>}
    </section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
