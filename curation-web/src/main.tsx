import React, {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import {active, api, base, message, requirement, resumable, Run, send, setToken, status} from "./api";
import {RunDetail} from "./RunDetail";
import "./style.css";

function App() {
  const [file, setFile] = useState<File | null>(null), [url, setUrl] = useState("");
  const [runs, setRuns] = useState<Run[]>([]), [ready, setReady] = useState(false), [configured, setConfigured] = useState(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [notice, setNotice] = useState("");
  const [selected, setSelected] = useState<string | null>(null), [checked, setChecked] = useState<string | null>(null);
  const [search, setSearch] = useState(""), [archived, setArchived] = useState(false), [limit, setLimit] = useState(10);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const running = runs.some(active);
  const checkedRun = runs.find(r => r.id === checked);
  async function refresh() {setRuns((await api<{items: Run[]}>("/api/curation/runs")).items);}
  useEffect(() => {
    let mounted = true;
    api<{token: string; gemini_configured: boolean}>("/api/curation/bootstrap").then(async b => {
      if (!mounted) return;
      setToken(b.token); setConfigured(b.gemini_configured); await refresh(); setReady(true);
    }).catch(e => setError(message(e)));
    return () => {mounted = false;};
  }, []);
  useEffect(() => {
    // Refresh also when another open window starts or reviews a run.
    const timer = window.setInterval(() => {void refresh().catch(e => setError(message(e)));}, running ? 2000 : 10000);
    return () => window.clearInterval(timer);
  }, [running]);
  async function action(r: Run, name: string, body?: unknown, method = "POST") {
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await send<Run>(`${base(r.id)}${name ? `/${name}` : ""}`, body, method);
      if (name === "rerun") {setChecked(result.id); setSelected(result.id); setArchived(false); setSearch("");}
      if (method === "DELETE") {sessionStorage.removeItem(`feedback-draft-${r.id}`); setSelected(null); setConfirmDelete(null); if (checked === r.id) setChecked(null); setNotice("Run deleted. Shared download and model caches retained.");}
      await refresh();
    } catch (e) {setError(message(e));} finally {setBusy(false);}
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (checkedRun?.status === "ready") {await action(checkedRun, "resume"); return;}
    setBusy(true); setError(""); setNotice("");
    try {
      let value: Run;
      if (file) {
        if (file.size > 512 * 1024 * 1024) throw new Error("Upload a ZIP of at most 512 MB.");
        value = await api<Run>(`/api/curation/runs/upload?preflight=true&filename=${encodeURIComponent(file.name)}`, {method: "POST", headers: {"Content-Type": "application/zip"}, body: file});
      } else {
        value = await send<Run>("/api/curation/runs/huggingface", {url: url.trim(), requirement, preflight: true});
      }
      setChecked(value.id); setSelected(value.id); setArchived(false); setSearch(""); await refresh();
    } catch (e) {setError(message(e));} finally {setBusy(false);}
  }
  const visible = runs.filter(r => Boolean(r.archived) === archived && `${r.source} ${r.requirement} ${r.id}`.toLowerCase().includes(search.toLowerCase()));
  return <main>
    <header><h1>Curate</h1><p>A dataset in. Reviewed examples out.</p></header>
    <form onSubmit={submit}>
      <label htmlFor="dataset">Upload a dataset</label>
      <input id="dataset" type="file" accept=".zip" disabled={busy || running} onChange={e => {setFile(e.target.files?.[0] || null); setChecked(null); if (e.target.files?.[0]) setUrl("");}} />
      <p className="help">ZIP · up to 512 MB · <a href="/api/curation/format" target="_blank" rel="noreferrer">Format</a> · <a href="/api/curation/example.zip" download>Example ZIP</a> (silent format template)</p>
      <label htmlFor="link">Or a Hugging Face dataset link</label>
      <input id="link" type="url" placeholder="https://huggingface.co/datasets/owner/name" value={url} disabled={busy || running} onChange={e => {setUrl(e.target.value); setChecked(null); if (e.target.value) {setFile(null); (document.getElementById("dataset") as HTMLInputElement).value = "";}}} />
      <p className="help">Requires audio and timed speaker transcripts. Supports our ZIP schema and TurnBench; other schemas are rejected.</p>
      <label htmlFor="requirement">Requirement</label>
      <textarea id="requirement" value={requirement} readOnly rows={2} aria-describedby="scope" />
      <p id="scope" className="help">Interruption curation only. Checking downloads and validates the source without model calls. Curation sends audio and transcript context to Gemini.</p>
      {checkedRun && <p role="status" className="source-check">{checkedRun.status === "ready" ? `Compatible · ${checkedRun.conversations} conversations · ${checkedRun.candidates} candidate events. Ready to curate.` : `${status(checkedRun)} · ${checkedRun.message}`}</p>}
      {!configured && ready && <p role="alert">Gemini is not configured on the server.</p>}
      <button disabled={!ready || (!configured && checkedRun?.status === "ready") || busy || running || (!file && !url.trim() && checkedRun?.status !== "ready")}>{busy ? "Working…" : running ? "Run in progress…" : checkedRun?.status === "ready" ? "Curate dataset" : "Check dataset"}</button>
    </form>
    {error && <p role="alert" className="error">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <section aria-labelledby="history"><h2 id="history">History</h2>
      {runs.length > 0 && <div className="history-filters"><input aria-label="Search history" type="search" placeholder="Search history" value={search} onChange={e => {setSearch(e.target.value); setLimit(10);}} /><label className="checkbox"><input type="checkbox" checked={archived} onChange={e => {setArchived(e.target.checked); setLimit(10);}} />Archived</label></div>}
      {!visible.length ? <p className="empty">{ready ? runs.length ? "No matching runs." : "Your runs will appear here." : "Loading…"}</p> : <ol>{visible.slice(0, limit).map(r => <li key={r.id}>
        <div className="row"><h3><button className="history-title" aria-expanded={selected === r.id} onClick={() => setSelected(selected === r.id ? null : r.id)}>{r.source}</button></h3><time dateTime={r.created}>{new Date(r.created).toLocaleString()}</time></div>
        <div className="row"><p role={active(r) ? "status" : undefined}><strong>{status(r)}</strong>{r.download_ready ? ` · ${r.selected_conversations}/${r.conversations} conversations selected · ${r.selected_events} events` : ` · ${r.message}`}</p>{r.download_ready && <a className="download" href={`${base(r.id)}/download`} download>Model ZIP</a>}</div>
        {(active(r) || r.errors > 0 || r.status === "paused") && <p className="help">{r.processed}/{r.candidates} candidates attempted · {r.errors} errors{r.status === "paused" && " · Partial results; remaining work is not labeled negative"}</p>}
        <div className="run-actions">
          {resumable(r) && !(r.status === "ready" && checked === r.id) && <button className="small-button" disabled={busy || running || !configured} onClick={() => action(r, "resume")}>{r.status === "ready" ? "Curate dataset" : r.status === "completed_with_errors" ? "Retry failed items" : "Resume"}</button>}
          {active(r) && r.status !== "packaging" && <button className="small-button" disabled={busy || r.status === "pausing"} onClick={() => action(r, "pause")}>Pause</button>}
          {!active(r) && <details className="more"><summary aria-label={`More actions for ${r.source}`}>More</summary><div>
            <button className="text-button" disabled={busy || running} onClick={() => action(r, "rerun")}>Run again from same source</button>
            <button className="text-button" disabled={busy} onClick={() => action(r, "archive", {archived: !r.archived}, "PUT")}>{r.archived ? "Restore" : "Archive"}</button>
            <button className="text-button" disabled={busy} onClick={() => setConfirmDelete(r.id)}>Delete local run…</button>
          </div></details>}
        </div>
        {confirmDelete === r.id && <div className="delete-confirm" role="group" aria-label="Confirm deletion"><p>Delete this run’s audio, reviews and exports permanently? Shared download and model caches remain. Archive keeps everything.</p><button className="small-button" disabled={busy} onClick={() => action(r, "", undefined, "DELETE")}>Delete permanently</button> <button className="text-button" onClick={() => setConfirmDelete(null)}>Cancel</button></div>}
        {selected === r.id && <RunDetail key={r.id} run={r} />}
      </li>)}</ol>}
      {visible.length > limit && <button className="small-button" onClick={() => setLimit(x => x + 10)}>Show more runs</button>}
    </section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
