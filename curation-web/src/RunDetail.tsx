import {useEffect, useRef, useState} from "react";
import {api, base, Decision, Detail, Export, message, Run, active, send} from "./api";

export function RunDetail({run}: {run: Run}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [view, setView] = useState("selected"), [offset, setOffset] = useState(0), [sample, setSample] = useState(false);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false), [notice, setNotice] = useState("");
  const [feedback, setFeedback] = useState(""), [savedText, setSavedText] = useState("");
  const [reload, setReload] = useState(0), [loading, setLoading] = useState(false);
  const working = busy || loading;
  const initialized = useRef(false);
  const [undo, setUndo] = useState<{id: string; decision: Decision} | null>(null);
  const dirty = feedback !== savedText;
  useEffect(() => {
    const controller = new AbortController();
    setError(""); setLoading(true);
    api<Detail>(`${base(run.id)}?view=${view}&offset=${offset}&sample=${sample}`, {signal: controller.signal})
      .then(value => {
        setDetail(value);
        if (!initialized.current) {setFeedback(sessionStorage.getItem(`feedback-draft-${run.id}`) ?? value.feedback.text); setSavedText(value.feedback.text); initialized.current = true;}
      }).catch(e => {if (!controller.signal.aborted) setError(message(e));}).finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [run.id, run.status, offset, view, sample, reload]);
  useEffect(() => {
    if (!dirty) return;
    const handler = (e: BeforeUnloadEvent) => {e.preventDefault();};
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);
  async function review(id: string, decision: Decision, isUndo = false) {
    if (!detail) return;
    setBusy(true); setError(""); setNotice("");
    const previous = detail.reviews.items[id]?.decision || "unreviewed";
    try {
      await send(`${base(run.id)}/reviews/${id}`, {decision, revision: detail.reviews.revision}, "PUT");
      setUndo(isUndo ? null : {id, decision: previous}); setReload(x => x + 1);
      setNotice(isUndo ? "Review undone." : `Marked ${decision}. Original model ZIP unchanged.`);
    } catch (e) {setError(message(e));} finally {setBusy(false);}
  }
  async function exportReviewed() {
    if (!detail) return;
    setBusy(true); setError("");
    try {
      const value = await send<Export>(`${base(run.id)}/reviewed-exports`, {revision: detail.reviews.revision});
      setNotice(`Reviewed ZIP ready: ${value.selected_events} kept events.`); setReload(x => x + 1);
    } catch (e) {setError(message(e));} finally {setBusy(false);}
  }
  async function saveFeedback(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError("");
    try {
      await send(`${base(run.id)}/feedback`, {text: feedback}, "PUT");
      setSavedText(feedback); sessionStorage.removeItem(`feedback-draft-${run.id}`); setNotice("Note saved. Notes do not change labels or exports.");
    } catch (e) {setError(message(e));} finally {setBusy(false);}
  }
  const reviewItems = Object.values(detail?.reviews.items || {});
  const kept = reviewItems.filter(r => r.decision === "keep").length;
  const reviewed = reviewItems.filter(r => r.decision !== "unreviewed").length;
  const latest = detail?.reviewed_exports.at(-1);
  return <section className="run-detail" aria-label="Run details" aria-busy={loading}>
    <p className="help">Model suggestions need review. Keep confirms an example for a reviewed export; Exclude and Unsure leave it out.</p>
    {error && <p role="alert" className="error">{error} <button className="text-button" onClick={() => setReload(x => x + 1)}>Refresh details</button></p>}
    {notice && <p role="status">{notice} {undo && <button className="text-button" disabled={working} onClick={() => review(undo.id, undo.decision, true)}>Undo review</button>}</p>}
    {loading && detail && <p role="status" className="help">Updating examples…</p>}
    {detail ? <>
      {detail.coverage && <details className="coverage"><summary>What was searched</summary>
        <p>{run.conversations} conversations · {detail.coverage.timed_turns} timed incoming turns · {detail.coverage.candidate_events} overlap candidates · {detail.coverage.not_proposed_turns} turns not proposed · {detail.coverage.untimed_turns} untimed turns.</p>
        <p className="help">Only overlap candidates go to the model. “Not proposed” is not a negative label. Counts do not measure accuracy or recall. Human–human TurnBench data is searched in both speaker directions.</p>
      </details>}
      {detail.legacy && <p className="help">Older run: inspection is limited to conversations retained in its original ZIP. Submit the source again for complete coverage and resumable processing.</p>}
      <div className="review-summary"><span>{reviewed} reviewed · {kept} kept</span>
        {detail.reviews.revision > 0 && <button className="small-button" disabled={working || active(run)} onClick={exportReviewed}>Prepare reviewed ZIP</button>}
      </div>
      {latest && <p><a href={`${base(run.id)}/reviewed-exports/${latest.id}`} download>Download reviewed ZIP</a> · {latest.selected_events} kept events{latest.review_revision !== detail.reviews.revision && " · Older review version; prepare a new ZIP to include your changes"}</p>}
      {(run.exports?.length || 0) > 1 && <details><summary>Earlier model exports</summary>{run.exports!.slice(0, -1).map(v => <p key={v.path}><a href={`${base(run.id)}/model-exports/${v.path}`} download>{new Date(v.created).toLocaleString()}</a> · {v.status.replaceAll("_", " ")}</p>)}</details>}
      {detail.reviewed_exports.length > 1 && <details><summary>Earlier reviewed exports</summary>{detail.reviewed_exports.slice(0, -1).map(v => <p key={v.id}><a href={`${base(run.id)}/reviewed-exports/${v.id}`} download>Review version {v.review_revision}</a> · {v.selected_events} events</p>)}</details>}
      <div className="filters">
        <label>Examples<select aria-label="Examples" value={view} disabled={working} onChange={e => {setView(e.target.value); setOffset(0);}}>
          <option value="selected">Selected ({detail.counts?.selected || 0})</option>
          <option value="not_selected">Not selected ({detail.counts?.not_selected || 0})</option>
          <option value="unresolved">Unresolved ({detail.counts?.unresolved || 0})</option>
          <option value="not_proposed">Not proposed ({detail.counts?.not_proposed || 0})</option>
          <option value="all">All examples</option>
        </select></label>
        <label className="checkbox"><input type="checkbox" checked={sample} disabled={working} onChange={e => {setSample(e.target.checked); setOffset(0);}} />Sample order</label>
      </div>
      {sample && <p className="help">A reproducible shuffled order for spot checks. The first page is a sample, not an accuracy estimate.</p>}
      {view === "not_proposed" && <p className="help">These turns did not trigger the overlap detector. Listen to check for omissions; the model did not judge them.</p>}
      {!detail.total ? <p>{active(run) || run.status === "ready" ? "No results in this view yet." : "No examples in this view."}</p> : <>
        <p className="help">{offset + 1}–{Math.min(offset + detail.limit, detail.total)} of {detail.total} events / turns</p>
        {detail.items.map(item => <article className="datapoint" key={item.id}>
          <h3>{item.conversation_id} · {item.id}</h3>
          <p className="help">{item.category.replaceAll("_", " ")} · {item.evidence.clip_start_s.toFixed(1)}–{item.evidence.clip_end_s.toFixed(1)}s in source{item.roles_reversed && " · Opposite speaker direction"}</p>
          <audio controls preload="none" aria-label={`Listen to ${item.id}`} src={`${base(run.id)}/events/${item.id}/audio`} />
          {item.evidence.target_truncated && <p className="help">Target extends beyond this preview. Check the full source before confirming.</p>}
          <p>{item.answer.evidence_note}</p>
          {item.error && <p className="help">Analysis error: {item.error}</p>}
          <div className="decisions" role="group" aria-label={`Review ${item.id}`}>
            {(["keep", "exclude", "unsure"] as Decision[]).map(d => <button key={d} disabled={working} aria-pressed={item.review.decision === d} onClick={() => review(item.id, d)}>{d.charAt(0).toUpperCase() + d.slice(1)}</button>)}
            {item.review.decision !== "unreviewed" && <button className="text-button" disabled={working} onClick={() => review(item.id, "unreviewed")}>Clear</button>}
          </div>
          <details><summary>Transcript context</summary><p className="help">Source speaker labels and times. Highlighted text is the target turn.</p>
            {item.turns.map(t => <p key={t.index} className={t.index === item.target_turn_index ? "target-turn" : ""}><strong>{t.role}</strong>{t.start_s !== null && ` (${t.start_s.toFixed(1)}s)`}: {t.text}</p>)}
            {item.context_truncated && <p className="help">Preview shortened. Full conversations are retained in exports for kept examples.</p>}
          </details>
        </article>)}
        <nav className="pagination" aria-label="Example pages"><button disabled={offset === 0 || working} onClick={() => setOffset(Math.max(0, offset - 10))}>Previous</button><button disabled={offset + 10 >= detail.total || working} onClick={() => setOffset(offset + 10)}>Next</button></nav>
      </>}
    </> : !error && <p role="status">Loading details…</p>}
    <details><summary>Notes on this run{dirty ? " · Unsaved" : ""}</summary><form onSubmit={saveFeedback}>
      <label htmlFor={`note-${run.id}`}>Feedback</label>
      <textarea id={`note-${run.id}`} rows={3} maxLength={10000} value={feedback} disabled={!detail || working} onChange={e => {setFeedback(e.target.value); sessionStorage.setItem(`feedback-draft-${run.id}`, e.target.value);}} />
      <p className="help">Notes are saved separately from example labels. They do not retrain the model.</p>
      <button className="small-button" disabled={!detail || working || !dirty}>Save note</button>
    </form></details>
  </section>;
}
