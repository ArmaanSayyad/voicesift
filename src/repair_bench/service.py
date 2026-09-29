"""Serialized controlled-run coordinator. No physical playback is inferred."""

import json
import os
import selectors
import shutil
import signal
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from .contracts import Plan, verbalize
from .store import Store, Journal, atomic_json, sha

SYSTEM = "You help propose dinner plans, never book anything. Track the user's current request across the conversation. Incorporate corrections and preserve unmentioned details. Return JSON with day, time in HH:MM 24-hour format, party_size, and status. Use null for unknown values and needs_clarification when necessary. Status is proposed, cancelled, or needs_clarification."
EXPECTED = {"day": "Saturday", "time": "19:00", "party_size": 2, "status": "proposed"}


class Worker:
    def __init__(self, repo):
        self.repo = repo
        self.process = None
        self.log = None

    def close(self):
        if self.process:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def request(self, operation, **kwargs):
        if self.process is None or self.process.poll() is not None:
            self.close()
            executable = self.repo / ".venv-audio/bin/python"
            if not executable.exists():
                raise RuntimeError("Audio environment is missing; run setup.sh audio.")
            self.log = (self.repo / "artifacts/worker.log").open("a")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self.repo / "src")
            self.process = subprocess.Popen(
                [str(executable), "-m", "repair_bench.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
                cwd=self.repo,
            )
        self.process.stdin.write(json.dumps({"op": operation, **kwargs}) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            if not sel.select(180):
                self.close()
                raise TimeoutError("Audio worker exceeded 180 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Audio worker exited; inspect artifacts/worker.log")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["result"]


def ollama(path, payload=None):
    req = urllib.request.Request(
        "http://127.0.0.1:11434" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


class Coordinator:
    def __init__(self, repo, root=None, worker=None, chat=None):
        self.repo = repo.resolve()
        self.store = Store(root or self.repo / "artifacts/runs")
        self.worker = worker or Worker(self.repo)
        self.chat = chat or ollama
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="repair-run")
        self.lock = threading.Lock()
        self.active = None

    def start(self, scenario):
        with self.lock:
            if self.active:
                raise RuntimeError("A run is already active; wait for it to finish.")
            run = self.store.create(scenario)
            self.active = run
            self.pool.submit(self.execute, run)
            return run

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.worker.close()

    def execute(self, run):
        folder = self.store.folder(run)
        journal = Journal(folder / "events.jsonl", run)
        manifest = {
            "schema_version": 1,
            "run_id": run,
            "mode": "offline_completed_utterances",
            "variant": "C0-history-dev",
            "split": "dev",
            "synthetic_input": True,
            "human_reviewed": False,
            "physical_playback_measured": False,
            "interruption_tested": False,
            "turns": [],
            "status": "running",
            "system_prompt": SYSTEM,
            "schema": Plan.model_json_schema(),
            "sampling": {
                "temperature": 0,
                "seed": "unsupported/not requested",
                "num_predict": 160,
            },
            "voice": "af_heart",
            "created": self.store.get(run)["created"],
        }
        try:
            journal.emit(
                "run.started",
                {"mode": manifest["mode"], "variant": manifest["variant"]},
            )
            self.store.update(run, "loading")
            raw = json.loads((self.repo / "artifacts/models.json").read_text())
            manifest["models"] = {
                k: {x: y for x, y in v.items() if x != "path"}
                for k, v in raw.items()
                if k in ("kokoro", "parakeet")
            }
            tags = self.chat("/api/tags")["models"]
            model = next((x for x in tags if x["name"] == "qwen3.5:9b"), None)
            if not model:
                raise RuntimeError("Ollama qwen3.5:9b is not installed")
            manifest["llm"] = {"name": model["name"], "digest": model["digest"]}
            manifest["code_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=self.repo, text=True
            )
            manifest["code_dirty"] = bool(dirty)
            prep = journal.emit("worker.prepare.requested", {})
            self.worker.request(
                "prepare", models=str(self.repo / "artifacts/models.json")
            )
            journal.emit("worker.ready", {}, [prep])
            self.store.update(run, "running")
            history = [{"role": "system", "content": SYSTEM}]
            for i in range(2):
                source = self.repo / f"artifacts/phrase-{i}.wav"
                if not source.exists():
                    raise RuntimeError(
                        "Synthetic input is missing; run scripts/probe_audio.py first."
                    )
                input_name = f"input-{i}.wav"
                output_name = f"response-{i}.wav"
                shutil.copyfile(source, folder / input_name)
                arrival = journal.emit(
                    "input.available",
                    {
                        "turn": i,
                        "asset": input_name,
                        "sha256": sha(folder / input_name),
                        "provenance": "Kokoro af_heart synthetic development fixture",
                        "schedule": "completed utterance after previous response generation; not audio-paced",
                    },
                )
                asr_start = time.monotonic()
                asr = self.worker.request("asr", path=str(folder / input_name))
                asr_s = time.monotonic() - asr_start
                transcript_event = journal.emit(
                    "asr.final", {"turn": i, **asr, "compute_s": asr_s}, [arrival]
                )
                history.append({"role": "user", "content": asr["text"]})
                req = journal.emit(
                    "dialogue.requested",
                    {"turn": i, "messages": history.copy()},
                    [transcript_event],
                )
                start = time.monotonic()
                reply = self.chat(
                    "/api/chat",
                    {
                        "model": "qwen3.5:9b",
                        "messages": history,
                        "format": Plan.model_json_schema(),
                        "think": False,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 160},
                    },
                )
                plan = Plan.model_validate_json(reply["message"]["content"])
                llm_s = time.monotonic() - start
                result = journal.emit(
                    "dialogue.completed",
                    {
                        "turn": i,
                        "plan": plan.model_dump(),
                        "compute_s": llm_s,
                        "meaning": "model-proposed plan; no external state reducer or booking",
                    },
                    [req],
                )
                text = verbalize(plan)
                history.append({"role": "assistant", "content": text})
                start = time.monotonic()
                tts = self.worker.request(
                    "tts", text=text, path=str(folder / output_name)
                )
                tts_s = time.monotonic() - start
                journal.emit(
                    "audio.generated",
                    {
                        "turn": i,
                        "asset": output_name,
                        "text": text,
                        "compute_s": tts_s,
                        **tts,
                        "played": False,
                    },
                    [result],
                )
                manifest["turns"].append(
                    {
                        "index": i,
                        "input_asset": input_name,
                        "output_asset": output_name,
                        "transcript": asr["text"],
                        "plan": plan.model_dump(),
                        "response": text,
                        "timings_s": {"asr": asr_s, "llm": llm_s, "tts": tts_s},
                        "duration_s": tts["duration_s"],
                    }
                )
                atomic_json(folder / "progress.json", manifest)
            # Evaluator executes after all agent calls. Its answers never enter history.
            actual = manifest["turns"][-1]["plan"]
            manifest["score"] = {
                "metric": "exact_proposed_plan/v1",
                "status": "scored",
                "value": actual == EXPECTED,
                "expected": EXPECTED,
                "actual": actual,
                "denominator": 1,
                "review": "unreviewed",
                "scope": "model-proposed structured plan only; not acoustic or interruption success",
            }
            journal.emit("score.generated", manifest["score"])
            journal.emit(
                "run.completed", {"behavior_passed": manifest["score"]["value"]}
            )
            manifest["status"] = "complete"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            journal.emit("run.failed", {"error": manifest["error"]})
        finally:
            journal.close()
            manifest["last_event_hash"] = journal.previous
            manifest["artifact_hashes"] = {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.is_file() and p.name not in ("manifest.json", "progress.json")
            }
            atomic_json(folder / "manifest.json", manifest)
            self.store.update(run, manifest["status"], manifest.get("error"))
            with self.lock:
                self.active = None
