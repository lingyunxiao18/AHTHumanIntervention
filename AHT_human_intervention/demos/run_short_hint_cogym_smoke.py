#!/usr/bin/env python3
"""Short timed HINT-Agent CoGym intervention micro-demo (no full Redis session)."""
import os
import sys
import time

import toml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)

for k, v in toml.load("third_party/collaborative-gym/secrets.example.toml").items():
    if v:
        os.environ[str(k)] = str(v)

HINT = os.path.abspath("AHT_human_intervention/hintagent/src")
sys.path[:0] = [
    os.path.join(HINT, "human_intervention"),
    os.path.join(HINT, "hintagent"),
]

import hint_agent_cogym as ha  # noqa: E402


def main() -> None:
    action_space = [
        {
            "max_length": 8192,
            "min_length": 1,
            "pattern": r"^EDITOR_UPDATE\(text=(.*)\)$",
            "params": ["text"],
            "machine_readable_identifier": "EDITOR_UPDATE",
            "human_readable_name": "Update editor",
            "human_readable_description": "Overwrite plan",
        },
        {
            "max_length": 512,
            "min_length": 1,
            "pattern": r"^BUSINESS_SEARCH\(term='(.*)', location='(.*)', limit=(\d+)\)$",
            "params": ["term", "location", "limit"],
            "machine_readable_identifier": "BUSINESS_SEARCH",
            "human_readable_name": "Search",
            "human_readable_description": "Search businesses",
        },
        {
            "max_length": 4,
            "min_length": 1,
            "pattern": r"^FINISH\(\)$",
            "params": [],
            "machine_readable_identifier": "FINISH",
            "human_readable_name": "Finish",
            "human_readable_description": "Done",
        },
    ]

    print("=== SHORT HINT-AGENT DEMO (travel-like) ===", flush=True)
    agent = ha.HINTAgentCoGym(
        model="gpt-5-mini",
        enable_cot=True,
        enable_memory=True,
        disable_send_teammate_message=True,
    )
    agent.start(
        "hint_agent",
        ["hint_agent", "teammate"],
        "Plan a 3-day Vancouver trip in December for one person.",
        action_space,
        "",
        [],
    )

    obs = {"travel_plan_editor": "", "search_output": None}
    t1 = time.perf_counter()
    a1 = agent.get_action(obs, [])
    print(f"[1] baseline action ({time.perf_counter() - t1:.1f}s): {a1[:140]}", flush=True)

    agent.apply_human_intervention(
        "Focus on restaurants first; hotel is already booked."
    )
    obs2 = {
        "travel_plan_editor": "",
        "search_output": {"query": "hotels", "output": "..."},
    }
    t2 = time.perf_counter()
    a2 = agent.get_action(obs2, [])
    print(f"[2] after intervention ({time.perf_counter() - t2:.1f}s): {a2[:160]}", flush=True)
    print(
        f"    events={agent.last_events} "
        f"category={(agent.last_plan or {}).get('category')}",
        flush=True,
    )
    cot = (agent.last_plan or {}).get("chain_of_thought") or ""
    print(f"    CoT: {cot[:200]}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
