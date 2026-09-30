export type Run = {
  id: string; source: string; requirement: string; created: string; status: string;
  message: string; conversations: number; candidates: number; processed: number;
  selected_conversations: number; selected_events: number; errors: number;
  exports?: {path: string; created: string; status: string}[];
  mode?: "audio" | "interruption";
  objective_plan?: {supported: boolean; summary: string; criteria: string[]; reason: string};
  download_ready: boolean; archived?: boolean; compatibility?: string;
  coverage?: {timed_turns: number; untimed_turns: number; candidate_events: number; not_proposed_turns: number};
};
export type Decision = "keep" | "exclude" | "unsure" | "unreviewed";
export type Reviews = {revision: number; items: Record<string, {decision: Decision}>};
export type Export = {id: string; review_revision: number; selected_events: number; created: string};
export type Item = {
  id: string; conversation_id: string; target_turn_index: number; category: string;
  roles_reversed: boolean; context_truncated: boolean; status: string; error?: string;
  answer: {evidence_note: string}; review: {decision: Decision};
  evidence: {clip_start_s: number; clip_end_s: number; target_truncated?: boolean};
  turns: {index: number; role: string; text: string; start_s: number | null; end_s: number | null}[];
};
export type Detail = {
  run: Run; feedback: {text: string}; reviews: Reviews; items: Item[];
  total: number; offset: number; limit: number; legacy: boolean;
  coverage: Run["coverage"]; counts?: Record<string, number>; reviewed_exports: Export[];
};
export const requirement = "conversations where there is an interruption";
export const active = (r: Run) => ["queued", "downloading", "preparing", "curating", "packaging", "pausing"].includes(r.status);
export const resumable = (r: Run) => ["ready", "paused", "interrupted", "failed", "completed_with_errors"].includes(r.status);
export const base = (id: string) => `/api/curation/runs/${id}`;
let token = "";
export function setToken(value: string) { token = value; }
export async function api<T>(url: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(url, {...options, headers: {"x-repair-token": token, ...options.headers}});
  if (!res.ok) {
    let message = `Request failed (${res.status}). Try again.`;
    try { const data = await res.json(); if (typeof data.detail === "string") message = data.detail; } catch { /* Non-JSON server error. */ }
    throw new Error(message);
  }
  return res.json();
}
export const send = <T,>(url: string, body?: unknown, method = "POST") => api<T>(url, {
  method, headers: {"Content-Type": "application/json"}, body: body === undefined ? undefined : JSON.stringify(body),
});
export const message = (e: unknown) => e instanceof Error ? e.message : String(e);
export function status(r: Run) {
  return ({completed: "Complete", completed_with_errors: "Partial", interrupted: "Interrupted", ready: "Ready", paused: "Paused", needs_clarification: "Clarify objective"} as Record<string, string>)[r.status]
    || r.status.charAt(0).toUpperCase() + r.status.slice(1);
}
