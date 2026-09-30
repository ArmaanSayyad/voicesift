import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
type Turn = {
  role: string;
  text: string;
  start_s: number | null;
  end_s: number | null;
};
type Review = {
  agent_outcome?: string;
  interruption_result?: string;
  label: string;
  include: boolean;
  note: string;
  reviewer: string;
  version: number;
};
type Candidate = {
  id: string;
  conversation_id: string;
  turn_index: number;
  conversation: {
    id: string;
    split: string;
    provenance: string;
    timing_source: string;
    turns: Turn[];
  };
  detection: {
    user_onset_s: number;
    overlap_s: number;
    agent_turn_indices: number[];
  };
  analysis?: {
    status: string;
    policy_version?: string;
    disposition?: string;
    error?: string;
    cached?: boolean;
    model?: string;
    answer?: { intent: string; agent_outcome: string; evidence_note: string; interruption_result?: string };
    evidence?: {
      clip_start_s: number;
      clip_end_s: number;
      target_truncated: boolean;
    };
  };
  review: Review | null;
  audio: { duration: number } | null;
};
type Metrics = {
  conversations: number;
  candidates: number;
  reviewed: number;
  included: number;
  timed_user_turns: number;
  untimed_user_turns: number;
};
let token = "";
async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch("/api/curation" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "x-repair-token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json();
    throw new Error(
      typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail),
    );
  }
  return r.json();
}
function Timeline({ c }: { c: Candidate }) {
  const turns = c.conversation.turns;
  const max = Math.max(1, ...turns.map((t) => t.end_s || 0));
  return (
    <svg
      className="timeline"
      viewBox="0 0 660 120"
      role="img"
      aria-label="Speech intervals: agent above, user below"
    >
      <text x="0" y="31">
        Agent
      </text>
      <text x="0" y="76">
        User
      </text>
      {turns.map((t, i) =>
        t.start_s === null ? null : (
          <rect
            key={i}
            className={
              i === c.turn_index
                ? "target-span"
                : t.role === "assistant"
                  ? "agent-span"
                  : "user-span"
            }
            x={70 + (t.start_s / max) * 570}
            y={t.role === "assistant" ? 15 : 60}
            width={Math.max(1, (((t.end_s || 0) - t.start_s) / max) * 570)}
            height="23"
            rx="3"
          >
            <title>
              {t.role}: {t.start_s}–{t.end_s} s
            </title>
          </rect>
        ),
      )}
      <text x="70" y="108">
        0 s
      </text>
      <text x="580" y="108">
        {max.toFixed(1)} s
      </text>
    </svg>
  );
}
function Detail({
  c,
  refresh,
  policyVersion,
}: {
  c: Candidate;
  refresh: () => Promise<void>;
  policyVersion: string;
}) {
  const [outcome, setOutcome] = useState(c.review?.agent_outcome || "unknown"),
    [label, setLabel] = useState(c.review?.label || "uncertain"),
    [success, setSuccess] = useState(c.review?.interruption_result || "unknown"),
    [include, setInclude] = useState(Boolean(c.review?.include && c.review?.interruption_result === "successful")),
    [note, setNote] = useState(c.review?.note || ""),
    [reviewer, setReviewer] = useState(c.review?.reviewer || "local-reviewer"),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);
  async function save() {
    setBusy(true);
    setMessage("");
    try {
      await api("/reviews/" + c.id, {
        label,
        agent_outcome: outcome,
        interruption_result: success,
        include,
        note,
        reviewer,
        expected_version: c.review?.version || 0,
      });
      await refresh();
      setMessage("Review saved.");
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function attach(file: File) {
    setBusy(true);
    try {
      const r = await fetch(
        `/api/curation/conversations/${c.conversation_id}/audio`,
        {
          method: "POST",
          headers: { "Content-Type": "audio/wav", "x-repair-token": token },
          body: file,
        },
      );
      if (!r.ok) throw new Error((await r.json()).detail);
      await refresh();
      setMessage("Source audio attached.");
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="detail-heading">
        <div>
          <div className="eyebrow">
            {c.conversation.split.toUpperCase()} /{" "}
            {c.conversation.timing_source.replaceAll("_", " ")}
          </div>
          <h2>{c.conversation.id}</h2>
        </div>
        <span className="badge">{c.review ? "reviewed" : "candidate"}</span>
      </div>
      <div className="result">
        <strong>User starts during agent speech</strong>
        <p>
          Onset {c.detection.user_onset_s.toFixed(3)} s · overlap{" "}
          {c.detection.overlap_s.toFixed(3)} s
        </p>
        <Timeline c={c} />
        <small>
          Timing detects overlap. Intent and whether the agent yielded require
          review.
        </small>
      </div>
      <div className="source">{c.conversation.provenance}</div>
      {c.audio ? (
        <div className="source-audio">
          <audio
            ref={audio}
            controls
            preload="metadata"
            src={`/api/curation/conversations/${c.conversation_id}/audio`}
          />
          <button
            className="secondary"
            onClick={() => {
              if (audio.current) {
                audio.current.currentTime = Math.max(
                  0,
                  c.detection.user_onset_s - 2,
                );
                void audio.current.play().catch((e) => setMessage(String(e)));
              }
            }}
          >
            Listen from 2 s before onset
          </button>
        </div>
      ) : (
        <div className="no-audio">
          <strong>No source audio attached.</strong>
          <p>
            Timestamp-only review cannot independently verify audible
            interruption. Attach the original WAV to listen.
          </p>
          <label className="file-label">
            Attach source WAV (≤50 MB, ≤15 min)
            <input
              type="file"
              accept=".wav"
              disabled={busy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void attach(f);
              }}
            />
          </label>
        </div>
      )}
      {c.analysis && (
        <section className="result">
          <h3>Gemini suggestion · {c.analysis.status}</h3>
          <p>{c.analysis.policy_version !== policyVersion ? "Legacy analysis — reanalyze with the current policy." : `Curation: ${c.analysis.disposition?.replaceAll("_", " ") || "needs review"}`}</p>
          {c.analysis.answer ? (
            <>
              <strong>{c.analysis.answer.intent.replaceAll("_", " ")}</strong>
              <p>Interruption result: {c.analysis.answer.interruption_result || "not assessed"}</p>
              <p>Agent outcome: {c.analysis.answer.agent_outcome}</p>
              <p>{c.analysis.answer.evidence_note}</p>
              <small>
                {c.analysis.model} ·{" "}
                {c.analysis.cached ? "cached result" : "new analysis"} ·
                Requires your review
              </small>
              {c.analysis.evidence && (
                <p>
                  Clip {c.analysis.evidence.clip_start_s.toFixed(2)}–
                  {c.analysis.evidence.clip_end_s.toFixed(2)} s{" "}
                  {c.analysis.evidence.target_truncated
                    ? "· target truncated"
                    : ""}
                </p>
              )}
              <audio controls src={`/api/curation/candidates/${c.id}/clip`} />
            </>
          ) : (
            <p>
              {c.analysis.error} — kept for manual review; retry with Analyze
              selected.
            </p>
          )}
        </section>
      )}
      <h3>Conversation context</h3>
      <div className="transcript-list">
        {c.conversation.turns.map((t, i) => (
          <div
            key={i}
            className={"utterance " + (i === c.turn_index ? "focus-turn" : "")}
          >
            <span>
              {t.role} ·{" "}
              {t.start_s === null
                ? "untimed"
                : `${t.start_s.toFixed(2)}–${t.end_s?.toFixed(2)} s`}
              {i === c.turn_index ? " · candidate onset" : ""}
            </span>
            <p>{t.text}</p>
          </div>
        ))}
      </div>
      <div className="review-form">
        <h3>Your annotation</h3>
        <label>
          Decision
          <select
            value={label}
            onChange={(e) => {
              setLabel(e.target.value);
              if (e.target.value !== "interruption") { setInclude(false); setSuccess("unknown"); }
            }}
          >
            <option value="uncertain">Uncertain / insufficient evidence</option>
            <option value="interruption">Confirmed interruption attempt</option>
            <option value="backchannel">Backchannel / acknowledgement</option>
            <option value="other_overlap">Other overlap / timing issue</option>
          </select>
        </label>
        <p>Confirm who held the floor, whether entry preceded a natural completion point, and whether the incoming speaker actually gained the turn. Overlap or a stopped segment alone is insufficient.</p>
        <label>
          Did the interruption succeed?
          <select value={success} onChange={(e) => { setSuccess(e.target.value); if (e.target.value !== "successful") setInclude(false); }}>
            <option value="unknown">Unknown / not confirmed</option>
            <option value="successful">Successful takeover, confirmed from evidence</option>
            <option value="unsuccessful">Attempt only; did not take the floor</option>
          </select>
        </label>
        <label>
          Observed agent outcome
          <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
            <option value="unknown">Unknown / insufficient evidence</option>
            <option value="continued">Continued speaking</option>
            <option value="stopped">Stopped</option>
            <option value="resumed">Resumed</option>
          </select>
        </label>
        <label>
          Reviewer
          <input
            value={reviewer}
            maxLength={80}
            onChange={(e) => setReviewer(e.target.value)}
          />
        </label>
        <label>
          Evidence or notes
          <textarea
            value={note}
            maxLength={2000}
            onChange={(e) => setNote(e.target.value)}
            placeholder="What did you hear? Note timing uncertainty or annotation-only review."
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={include}
            disabled={label !== "interruption" || success !== "successful"}
            onChange={(e) => setInclude(e.target.checked)}
          />{" "}
          Include this confirmed successful interruption in exports
        </label>
        <button disabled={busy || !reviewer.trim()} onClick={save}>
          Save review
        </button>
        <p role="status">{message}</p>
      </div>
    </>
  );
}
function App() {
  const [configured, setConfigured] = useState(false),
    [policyVersion, setPolicyVersion] = useState(""),
    [jobs, setJobs] = useState<
      {
        id: string;
        status: string;
        total: number;
        completed: number;
        errors: number;
      }[]
    >([]),
    [items, setItems] = useState<Candidate[]>([]),
    [metrics, setMetrics] = useState<Metrics | null>(null),
    [selected, setSelected] = useState(""),
    [draft, setDraft] = useState(""),
    [showImport, setShowImport] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false),
    [filter, setFilter] = useState("all"),
    [exportId, setExportId] = useState("");
  async function refresh() {
    const [c, m, j] = await Promise.all([
      api<{ items: Candidate[]; total: number }>("/candidates"),
      api<Metrics>("/metrics"),
      api<{
        items: {
          id: string;
          status: string;
          total: number;
          completed: number;
          errors: number;
        }[];
      }>("/jobs"),
    ]);
    setJobs(j.items);
    setItems(c.items);
    setMetrics(m);
    setSelected((old) => old || c.items[0]?.id || "");
  }
  useEffect(() => {
    api<{ token: string; gemini_configured: boolean; policy_version: string }>("/bootstrap")
      .then(async (b) => {
        token = b.token;
        setConfigured(b.gemini_configured);
        setPolicyVersion(b.policy_version);
        await refresh();
      })
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    if (!jobs.some((j) => j.status === "queued" || j.status === "running"))
      return;
    const id = window.setInterval(() => {
      void refresh().catch((e) => setError(String(e)));
    }, 1500);
    return () => window.clearInterval(id);
  }, [jobs]);
  async function analyze(ids: string[]) {
    setBusy(true);
    setError("");
    try {
      await api("/analyze", { candidate_ids: ids });
      await refresh();
      setNotice(
        `Audio analysis queued for ${ids.length} candidates. Estimated cost $${(ids.length * 0.02).toFixed(2)}.`,
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function ingest(text: string) {
    setBusy(true);
    setError("");
    try {
      const r = await api<{
        imported: number;
        duplicates: number;
        candidates: number;
      }>("/import", { jsonl: text });
      await refresh();
      setNotice(
        `${r.imported} conversations imported · ${r.candidates} overlap candidates · ${r.duplicates} duplicates skipped`,
      );
      setShowImport(false);
      setDraft("");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function demo() {
    const r = await fetch("/api/curation/example");
    await ingest(await r.text());
  }
  async function exp() {
    try {
      const r = await api<{ id: string; records: number }>("/exports", {});
      setExportId(r.id);
      setNotice(
        `Created immutable export with ${r.records} reviewed examples. Audio is referenced, not copied.`,
      );
    } catch (e) {
      setError(String(e));
    }
  }
  const visible = items
    .filter(
      (c) =>
        filter === "all" ||
        (filter === "unreviewed" && !c.review) ||
        (filter === "included" && c.review?.include && c.review?.interruption_result === "successful") ||
        (filter === "suggested" &&
          c.analysis?.policy_version === policyVersion && c.analysis?.disposition === "shortlist") ||
        (filter === "needs_attention" &&
          (c.analysis?.status === "error" || c.analysis?.policy_version !== policyVersion ||
            c.analysis?.disposition === "needs_review")),
    )
    .sort(
      (a, b) =>
        Number(b.analysis?.policy_version === policyVersion && b.analysis?.disposition === "shortlist") -
        Number(a.analysis?.policy_version === policyVersion && a.analysis?.disposition === "shortlist"),
    );
  const c = items.find((x) => x.id === selected);
  return (
    <div className="layout">
      <aside>
        <a className="brand" href="/">
          ↳{" "}
          <span>
            Interruption
            <br />
            Curation
          </span>
        </a>
        <div className="small-label">DATASET WORKSPACE</div>
        <div className="nav-item">◉ &nbsp; Overlap candidates</div>
        <div className="roadmap">
          01 Import timed conversations
          <br />
          02 Inspect candidate overlaps
          <br />
          03 Review and export
        </div>
        <div className="local">
          <span className="dot" />
          Local workspace · Gemini analysis
          <br />
          <small>Goal 01 / interruptions</small>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <div className="eyebrow">CURATION / TIMING FIRST</div>
            <h1>Find where the user cuts in.</h1>
            <p>
              Turn timestamped conversations into reviewed interruption
              examples.
            </p>
          </div>
          <button onClick={() => setShowImport(!showImport)}>
            Import dataset
          </button>
        </header>
        <div className="notice">
          <strong>Overlap is a candidate, not a verdict.</strong>
          <span>
            Gemini shortlists clear successful interruptions from audio and context. You confirm the
            successful interruption; models cannot include records in exports.
          </span>
        </div>
        {error && (
          <div role="alert" className="error">
            {error}
            <button className="secondary" onClick={() => setError("")}>
              Dismiss
            </button>
          </div>
        )}
        {notice && (
          <p role="status" className="import-status">
            {notice}
          </p>
        )}
        {showImport && (
          <section className="import-panel">
            <h2>Import conversational JSONL</h2>
            <p>
              One conversation per line: id, source_group, split, provenance,
              timing_source, and turns with role, text, start_s, end_s. Times
              are seconds on the same source-audio clock. Speaker roles must be
              known.
            </p>
            <input
              aria-label="Choose JSONL dataset"
              type="file"
              accept=".jsonl,.ndjson,.txt"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  if (f.size > 2_000_000) setError("JSONL must be under 2 MB");
                  else void f.text().then(setDraft);
                }
              }}
            />
            <textarea
              aria-label="JSONL conversations"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Paste JSONL or choose a file"
            />
            <button
              disabled={busy || !draft.trim()}
              onClick={() => ingest(draft)}
            >
              Import and detect
            </button>
            <a href="/api/curation/example">Download format example</a>
          </section>
        )}
        <div className="stats">
          {[
            ["Conversations", metrics?.conversations || 0],
            ["Overlap candidates", metrics?.candidates || 0],
            ["Reviewed", metrics?.reviewed || 0],
            ["Included", metrics?.included || 0],
          ].map(([name, value]) => (
            <div key={name}>
              <small>{name}</small>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
        <div className="coverage">
          {metrics?.timed_user_turns || 0} timed user turns searched ·{" "}
          {metrics?.untimed_user_turns || 0} untimed user turns not searched. No
          recall claim.
        </div>
        <section className="analysis-panel">
          <h2>Analyze with Gemini</h2>
          <p>
            {configured
              ? "Gemini is ready. Source clips and nearby transcript context are sent for analysis."
              : "Gemini is not configured on the server. Manual review remains available."}{" "}
            Up to 50 candidates per batch; estimated cost $0.02 per candidate.
            Matching results are cached.
          </p>
          <button
            disabled={
              !configured ||
              busy ||
              !c?.audio ||
              jobs.some((j) => j.status === "running" || j.status === "queued")
            }
            onClick={() => c && analyze([c.id])}
          >
            Analyze selected
          </button>{" "}
          <button
            className="secondary"
            disabled={
              !configured ||
              busy ||
              !items.some(
                (x) => x.audio && (x.analysis?.status !== "complete" || x.analysis?.policy_version !== policyVersion),
              ) ||
              jobs.some((j) => j.status === "running" || j.status === "queued")
            }
            onClick={() =>
              analyze(
                items
                  .filter((x) => x.audio && (x.analysis?.status !== "complete" || x.analysis?.policy_version !== policyVersion))
                  .slice(0, 50)
                  .map((x) => x.id),
              )
            }
          >
            Analyze pending audio (up to 50)
          </button>
          {jobs[0] && (
            <p role="status">
              Analysis {jobs[0].status}: {jobs[0].completed}/{jobs[0].total}{" "}
              processed · {jobs[0].errors} errors. Errors remain in review.
            </p>
          )}
        </section>
        <div className="workspace">
          <section className="run-list">
            <div className="queue-head">
              <h2>Review queue</h2>
              <select
                aria-label="Filter candidates"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                <option value="all">All</option>
                <option value="unreviewed">Unreviewed</option>
                <option value="included">Included</option>
                <option value="suggested">Model shortlist: successful</option>
                <option value="needs_attention">Needs review / reanalysis</option>
              </select>
            </div>
            {visible.map((x) => (
              <button
                key={x.id}
                className={"run-card " + (selected === x.id ? "chosen" : "")}
                onClick={() => setSelected(x.id)}
              >
                <strong>{x.conversation.id}</strong>
                <span>
                  {x.detection.user_onset_s.toFixed(2)} s ·{" "}
                  {x.detection.overlap_s.toFixed(2)} s overlap
                </span>
                <span className="badge">
                  {x.review?.label ||
                    x.analysis?.answer?.intent ||
                    x.analysis?.status ||
                    "unreviewed"}
                </span>
              </button>
            ))}
            {!items.length && (
              <div className="empty">
                <p>Import timestamps to find overlap candidates.</p>
                <button className="secondary" disabled={busy} onClick={demo}>
                  Load constructed demo
                </button>
                <p>Three fictional conversations, no audio or gold labels.</p>
              </div>
            )}
            <div className="export-panel">
              <button
                className="secondary"
                disabled={!metrics?.included}
                onClick={exp}
              >
                Export reviewed examples
              </button>
              {exportId && (
                <>
                  <a href={"/api/curation/exports/" + exportId}>
                    Download JSONL
                  </a>
                  <a href={"/api/curation/exports/" + exportId + "/bundle"}>
                    Download audio clips + annotations ZIP
                  </a>
                </>
              )}
              <small>
                Latest review decisions only. Source context and audio
                references retained.
              </small>
            </div>
          </section>
          <section className="detail">
            {c ? (
              <Detail
                policyVersion={policyVersion}
                key={c.id + ":" + (c.review?.version || 0) + ":" + !!c.audio}
                c={c}
                refresh={refresh}
              />
            ) : (
              <div className="welcome">
                <span>FIRST CURATION GOAL</span>
                <h2>Did the user interrupt the agent?</h2>
                <p>
                  We find the temporal event first. You determine whether it was
                  an interruption, a backchannel, or uncertain.
                </p>
                <p>
                  Conversations without timestamps are retained but not
                  searched. Unlabelled mixed audio requires a separate
                  speaker-detection step that is not implemented here.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
