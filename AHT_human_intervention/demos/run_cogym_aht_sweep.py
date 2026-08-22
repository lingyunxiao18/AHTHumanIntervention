#!/usr/bin/env python3
"""
AHT × HINT Co-Gym sweep harness.

Runs a grid of teammate personas × conditions and writes summary.csv / summary.json.

Conditions:
  - no_intervention: supervisor-mode off
  - oracle_no_memory: oracle on, fresh memory path (or --no-memory)
  - oracle_with_memory: oracle on, shared memory path (warmed by prior oracle runs)

Example:
  python -m AHT_human_intervention.demos.run_cogym_aht_sweep \\
    --personas idle,complementary_searcher --idxs 0 --conditions no_intervention,oracle_no_memory
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("cogym_aht_sweep")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HINT_PROJECT = os.path.join(REPO_ROOT, "AHT_human_intervention")
DEMO_MODULE = "AHT_human_intervention.demos.hint_agent_cogym_demo"

DEFAULT_PERSONAS = ["idle", "complementary_searcher", "greedy_editor", "follower"]
DEFAULT_CONDITIONS = ["no_intervention", "oracle_no_memory", "oracle_with_memory"]


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_session_dir(results_root: str, session_uuid: str) -> Optional[str]:
    env_uuid = f"env_{session_uuid}"
    candidate = os.path.join(results_root, env_uuid)
    if os.path.isdir(candidate):
        return candidate
    # Fallback: newest env_* dir
    if not os.path.isdir(results_root):
        return None
    dirs = [
        os.path.join(results_root, d)
        for d in os.listdir(results_root)
        if d.startswith("env_") and os.path.isdir(os.path.join(results_root, d))
    ]
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]


def _load_metrics(session_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"session_dir": session_dir}
    perf_path = os.path.join(session_dir, "task_performance.json")
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            perf = json.load(f)
        out["task_completion"] = perf.get("task_completion")
        out["performance_rating"] = perf.get("performance_rating")
        out["commonsense_pass_rate"] = perf.get("commonsense_pass_rate")
        out["preference_pass_rate"] = perf.get("preference_pass_rate")

    hint_info = os.path.join(session_dir, "hint_agent", "info.json")
    if os.path.exists(hint_info):
        with open(hint_info) as f:
            info = json.load(f)
        out["total_steps"] = info.get("total_steps")
        out["total_interventions"] = info.get("total_interventions")

    action_log = os.path.join(session_dir, "hint_agent", "action_log.jsonl")
    events: Dict[str, int] = {}
    first_editor_t = None
    first_supervisor_t = None
    if os.path.exists(action_log):
        with open(action_log) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for ev in row.get("events") or []:
                    events[ev] = events.get(ev, 0) + 1
                act = row.get("action") or ""
                t = row.get("t")
                if first_editor_t is None and str(act).startswith("EDITOR_UPDATE"):
                    first_editor_t = t
                if first_supervisor_t is None and (row.get("supervisor_message") or "").strip():
                    first_supervisor_t = t
    out["event_histogram"] = events
    out["first_editor_t"] = first_editor_t
    out["first_supervisor_t"] = first_supervisor_t
    out["hint_before_editor"] = (
        first_supervisor_t is not None
        and first_editor_t is not None
        and first_supervisor_t <= first_editor_t
    )
    return out


def run_one(
    *,
    persona: str,
    idx: int,
    condition: str,
    model: str,
    max_steps: int,
    max_interventions: int,
    work_dir: str,
    sweep_tag: str,
    memory_path: str,
    redis_url: str,
) -> Dict[str, Any]:
    result_dir_tag = f"{sweep_tag}/{condition}/{persona}/idx{idx}"
    session_uuid = f"sweep_{condition}_{persona}_{idx}_{int(time.time())}"
    cmd = [
        sys.executable, "-m", DEMO_MODULE,
        "--task", "travel_planning",
        "--idx", str(idx),
        "--model", model,
        "--max-steps", str(max_steps),
        "--teammate-persona", persona,
        "--result-dir-tag", result_dir_tag,
        "--work-dir", work_dir,
        "--session-uuid", session_uuid,
        "--redis-url", redis_url,
        "--max-interventions", str(max_interventions),
    ]
    if condition == "no_intervention":
        cmd.extend(["--supervisor-mode", "off"])
    elif condition == "oracle_no_memory":
        cmd.extend(["--supervisor-mode", "oracle"])
        # Ephemeral memory path so patterns do not carry
        ephemeral = os.path.join(
            work_dir, "travel_planning", result_dir_tag, "ephemeral_memory.json"
        )
        os.makedirs(os.path.dirname(ephemeral), exist_ok=True)
        cmd.extend(["--memory-path", ephemeral])
    elif condition == "oracle_with_memory":
        cmd.extend(["--supervisor-mode", "oracle", "--memory-path", memory_path])
    else:
        raise ValueError(f"Unknown condition {condition}")

    logger.info("RUN %s", " ".join(cmd))
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - started
    results_root = os.path.join(work_dir, "travel_planning", result_dir_tag, "results")
    session_dir = _find_session_dir(results_root, session_uuid)
    row: Dict[str, Any] = {
        "persona": persona,
        "idx": idx,
        "condition": condition,
        "session_uuid": session_uuid,
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
    }
    if session_dir:
        row.update(_load_metrics(session_dir))
    else:
        logger.warning("No session dir found under %s", results_root)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="AHT × HINT Co-Gym sweep.")
    parser.add_argument("--personas", type=str, default=",".join(DEFAULT_PERSONAS))
    parser.add_argument("--idxs", type=str, default="0")
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-interventions", type=int, default=3)
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join(HINT_PROJECT, "run_logs", "cogym"),
    )
    parser.add_argument("--sweep-tag", type=str, default="aht_sweep")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument(
        "--shared-memory-path",
        type=str,
        default="",
        help="Shared intervention memory for oracle_with_memory (default under sweep dir).",
    )
    args = parser.parse_args()

    personas = _parse_list(args.personas)
    idxs = [int(x) for x in _parse_list(args.idxs)]
    conditions = _parse_list(args.conditions)

    sweep_root = os.path.join(args.work_dir, "sweeps", args.sweep_tag)
    os.makedirs(sweep_root, exist_ok=True)
    memory_path = args.shared_memory_path or os.path.join(
        sweep_root, "shared_intervention_memory.json"
    )

    # Prefer warming memory: run oracle_with_memory after some oracle_no_memory if both present
    ordered_conditions = list(conditions)
    if "oracle_with_memory" in ordered_conditions and "oracle_no_memory" in ordered_conditions:
        # Keep user order but ensure no_intervention first when present
        pass

    rows: List[Dict[str, Any]] = []
    for condition in ordered_conditions:
        for persona in personas:
            for idx in idxs:
                try:
                    row = run_one(
                        persona=persona,
                        idx=idx,
                        condition=condition,
                        model=args.model,
                        max_steps=args.max_steps,
                        max_interventions=args.max_interventions,
                        work_dir=args.work_dir,
                        sweep_tag=args.sweep_tag,
                        memory_path=memory_path,
                        redis_url=args.redis_url,
                    )
                except Exception as e:
                    logger.exception("Run failed")
                    row = {
                        "persona": persona,
                        "idx": idx,
                        "condition": condition,
                        "error": str(e),
                    }
                rows.append(row)
                # Incremental save
                summary_json = os.path.join(sweep_root, "summary.json")
                with open(summary_json, "w") as f:
                    json.dump(rows, f, indent=2, default=str)

    summary_csv = os.path.join(sweep_root, "summary.csv")
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = dict(r)
            if isinstance(flat.get("event_histogram"), dict):
                flat["event_histogram"] = json.dumps(flat["event_histogram"])
            writer.writerow(flat)

    logger.info("Wrote %s and %s (%d rows)", summary_csv, os.path.join(sweep_root, "summary.json"), len(rows))
    print(json.dumps({"n": len(rows), "summary_csv": summary_csv}, indent=2))


if __name__ == "__main__":
    main()
