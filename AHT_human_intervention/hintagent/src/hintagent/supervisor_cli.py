#!/usr/bin/env python3
"""
Outside-supervisor CLI for HINT-Agent × CoGym.

Runs in a *separate* terminal from the CoGym session. Type free-text
interventions; they are published over Redis and consumed by HINT-Agent on its
next decision step — the session is NOT paused (unlike Overcooked press-P).

Usage (after the demo prints env_uuid):
    python -m hintagent.supervisor_cli \\
        --env-uuid env_travel_planning_0_abcd1234 \\
        --redis-url redis://localhost:6379/0
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "hintagent", "src"))
for p in (PROJECT_ROOT, _HINTAGENT_SRC, os.path.join(_HINTAGENT_SRC, "hintagent")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outside supervisor: inject real-time interventions into HINT-Agent (no pause)."
    )
    parser.add_argument(
        "--env-uuid",
        type=str,
        required=True,
        help="Env UUID printed by the demo (often starts with env_).",
    )
    parser.add_argument("--agent-node-name", type=str, default="hint_agent")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    args = parser.parse_args()

    # Accept either env_xxx or the session uuid; Runner uses env_{session_uuid}
    env_uuid = args.env_uuid
    if not env_uuid.startswith("env_"):
        env_uuid = f"env_{env_uuid}"

    try:
        from hintagent.supervisor_channel import intervention_channel, publish_intervention
    except ImportError:
        from supervisor_channel import intervention_channel, publish_intervention  # type: ignore

    channel = intervention_channel(env_uuid, args.agent_node_name)
    print("=" * 60)
    print("HINT-Agent outside supervisor (real-time, no pause)")
    print(f"  Redis:   {args.redis_url}")
    print(f"  Channel: {channel}")
    print("  Type an intervention and press Enter.")
    print("  Empty line = ignore.  Ctrl-C / 'quit' = exit.")
    print("  Examples:")
    print('    Focus on restaurants first; I already booked the hotel.')
    print("    Don't overwrite the editor yet.")
    print("    You seem stuck — try a different search query.")
    print("=" * 60)

    while True:
        try:
            line = input("supervisor> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        try:
            publish_intervention(
                redis_url=args.redis_url,
                env_uuid=env_uuid,
                text=line,
                agent_node_name=args.agent_node_name,
            )
            print(f"  → sent ({len(line)} chars). HINT-Agent will apply on next decision.")
        except Exception as e:
            print(f"  ! failed to publish: {e}")
            print("    Is Redis running? Is the CoGym session still up?")


if __name__ == "__main__":
    main()
