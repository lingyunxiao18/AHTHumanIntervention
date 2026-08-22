#!/usr/bin/env python3
"""
Oracle outside supervisor for HINT-Agent × Co-Gym.

Listens on ``{env_uuid}/step`` for ego actions, holds TravelPlanner hidden
preferences, and publishes sparse free-text interventions on the same Redis
channel as ``supervisor_cli.py`` (never via teammate chat).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "hintagent", "src"))
if _HINTAGENT_SRC not in sys.path:
    sys.path.insert(0, _HINTAGENT_SRC)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("oracle_supervisor")

DEFAULT_CSV = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "..",
        "third_party",
        "collaborative-gym",
        "datasets",
        "TravelPlanner",
        "validation_with_hidden_profile.csv",
    )
)


def load_travel_prefs(idx: int, csv_path: str = DEFAULT_CSV) -> Dict[str, Any]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"TravelPlanner idx {idx} out of range (n={len(rows)})")
    row = rows[idx]
    prefs: Any = row.get("preferences") or "[]"
    local: Any = row.get("local_constraint") or "{}"
    try:
        prefs_list = ast.literal_eval(prefs) if isinstance(prefs, str) else prefs
    except Exception:
        prefs_list = [str(prefs)]
    try:
        local_dict = ast.literal_eval(local) if isinstance(local, str) else local
    except Exception:
        local_dict = {}
    return {
        "idx": idx,
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": row.get("days"),
        "people_number": row.get("people_number"),
        "budget": row.get("budget"),
        "preferences": prefs_list,
        "local_constraint": local_dict,
        "level": row.get("level"),
    }


def _hint_for_action(action: str, prefs: Dict[str, Any], sent: Set[str]) -> Optional[str]:
    """Return at most one new hint string for this ego action, or None."""
    budget = prefs.get("budget")
    people = prefs.get("people_number")
    local = prefs.get("local_constraint") or {}
    room = local.get("room type") if isinstance(local, dict) else None
    cuisine = local.get("cuisine") if isinstance(local, dict) else None

    if action.startswith("RESTAURANT_SEARCH") and "diverse_food" not in sent:
        sent.add("diverse_food")
        return (
            "Do not repeat the same restaurant across days; diversify meals. "
            + (f"Respect cuisine preferences: {cuisine}." if cuisine else "")
        ).strip()

    if action.startswith("ATTRACTION_SEARCH") and "diverse_attr" not in sent:
        sent.add("diverse_attr")
        return "Avoid repeating the same attraction across days; pick distinct sights."

    if action.startswith("ACCOMMODATION_SEARCH") and "rooms" not in sent:
        sent.add("rooms")
        bits = []
        if people:
            bits.append(f"party of {people}")
        if room:
            bits.append(f"prefer {room}")
        if budget:
            bits.append(f"stay within total budget ${budget}")
        return (
            "When choosing lodging, "
            + (", ".join(bits) if bits else "match group size and room type")
            + ". Use only accommodations returned by search."
        )

    if action.startswith("FLIGHT_SEARCH") and "flights" not in sent:
        sent.add("flights")
        return (
            "Use only flight numbers that appear in FLIGHT_SEARCH results "
            "(sandbox-valid IDs). Do not invent flight numbers."
        )

    if action.startswith("EDITOR_UPDATE"):
        if "budget_edit" not in sent and budget:
            sent.add("budget_edit")
            return (
                f"Before finishing: ensure the plan fits a total budget of ${budget} "
                f"for {people or 'the'} travelers, use sandbox-valid flights, and "
                "do not repeat restaurants or attractions across days."
            )
        if "editor_quality" not in sent:
            sent.add("editor_quality")
            return (
                "Revise the editor plan: valid flights only, no repeated venues, "
                "and align lodging with group size / entire-room preference."
            )

    if action.startswith("FINISH") and "pre_finish" not in sent:
        sent.add("pre_finish")
        return (
            "Hold FINISH until the plan meets budget, valid flights, and no repeated "
            "restaurants/attractions. Fix the editor first."
        )

    # repeated identical restaurant search — generic nudge
    return None


def _extract_step_payload(raw: Any) -> Dict[str, Any]:
    """Unwrap aact Message JSON / nested object to {role, action}."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    # Message.model_dump(): {"data": {"object": {...}}} or {"object": {...}}
    obj = raw
    if "data" in raw and isinstance(raw["data"], dict):
        obj = raw["data"]
    if isinstance(obj, dict) and "object" in obj and isinstance(obj["object"], dict):
        obj = obj["object"]
    return obj if isinstance(obj, dict) else {}


def run_oracle(
    redis_url: str,
    env_uuid: str,
    prefs: Dict[str, Any],
    max_interventions: int = 3,
    agent_node_name: str = "hint_agent",
    poll_seconds: float = 0.2,
    idle_exit_seconds: float = 180.0,
) -> int:
    import redis

    from hintagent.supervisor_channel import publish_intervention

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    step_channel = f"{env_uuid}/step"
    end_channel = f"{env_uuid}/end"
    pubsub.subscribe(step_channel, end_channel)
    logger.info(
        "Oracle listening on %s (max_interventions=%d) prefs=%s",
        step_channel,
        max_interventions,
        {k: prefs.get(k) for k in ("budget", "people_number", "dest", "level")},
    )

    sent: Set[str] = set()
    n_sent = 0
    last_actions: List[str] = []
    last_msg_at = time.time()

    while True:
        msg = pubsub.get_message(timeout=poll_seconds)
        if msg is None:
            if time.time() - last_msg_at > idle_exit_seconds:
                logger.info("Oracle idle timeout; exiting.")
                break
            continue
        if msg.get("type") != "message":
            continue
        last_msg_at = time.time()
        channel = msg.get("channel")
        data = msg.get("data")
        if channel == end_channel:
            logger.info("Oracle saw end; exiting.")
            break
        obj = _extract_step_payload(data)
        role = obj.get("role")
        action = obj.get("action")
        if not action or role != agent_node_name:
            continue

        action_s = str(action)
        last_actions.append(action_s)
        # Detect repeated restaurant searches
        if (
            len(last_actions) >= 2
            and last_actions[-1].startswith("RESTAURANT_SEARCH")
            and last_actions[-2].startswith("RESTAURANT_SEARCH")
            and "repeat_search" not in sent
            and n_sent < max_interventions
        ):
            sent.add("repeat_search")
            text = "Stop repeating the same restaurant search; move on to attractions or draft a diverse plan."
            publish_intervention(redis_url, env_uuid, text, agent_node_name)
            n_sent += 1
            logger.info("Oracle intervention %d/%d: %s", n_sent, max_interventions, text[:100])
            continue

        if n_sent >= max_interventions:
            continue
        hint = _hint_for_action(action_s, prefs, sent)
        if not hint:
            continue
        publish_intervention(redis_url, env_uuid, hint, agent_node_name)
        n_sent += 1
        logger.info("Oracle intervention %d/%d: %s", n_sent, max_interventions, hint[:120])

    return n_sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle outside supervisor for HINT-Agent.")
    parser.add_argument("--env-uuid", type=str, required=True)
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--idx", type=int, default=0, help="TravelPlanner row index.")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV)
    parser.add_argument("--max-interventions", type=int, default=3)
    parser.add_argument("--agent-node-name", type=str, default="hint_agent")
    parser.add_argument("--idle-exit-seconds", type=float, default=300.0)
    parser.add_argument(
        "--prefs-json",
        type=str,
        default="",
        help="Optional JSON prefs override (skips CSV).",
    )
    args = parser.parse_args()

    if args.prefs_json:
        prefs = json.loads(args.prefs_json)
    else:
        prefs = load_travel_prefs(args.idx, args.csv_path)

    n = run_oracle(
        redis_url=args.redis_url,
        env_uuid=args.env_uuid,
        prefs=prefs,
        max_interventions=args.max_interventions,
        agent_node_name=args.agent_node_name,
        idle_exit_seconds=args.idle_exit_seconds,
    )
    print(json.dumps({"interventions_sent": n, "prefs": prefs}, default=str))


if __name__ == "__main__":
    main()
