#!/usr/bin/env python3
"""Launch an action-only peer teammate AgentNode for Co-Gym."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# teammates/ -> hintagent/ -> src/ -> hintagent(pkg)/ -> AHT_human_intervention
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "hintagent", "src"))
if _HINTAGENT_SRC not in sys.path:
    sys.path.insert(0, _HINTAGENT_SRC)
_COGYM_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "third_party", "collaborative-gym"))
if os.path.isdir(_COGYM_DIR) and _COGYM_DIR not in sys.path:
    sys.path.insert(0, _COGYM_DIR)


def main() -> None:
    from hintagent.teammates.peer_personas import PERSONA_NAMES, make_peer_teammate

    parser = argparse.ArgumentParser(description="Launch action-only peer teammate.")
    parser.add_argument("--node-name", type=str, required=True)
    parser.add_argument("--env-uuid", type=str, required=True)
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument(
        "--persona",
        type=str,
        default="complementary_searcher",
        choices=list(PERSONA_NAMES),
    )
    parser.add_argument("--wait-time", type=int, default=1)
    args = parser.parse_args()

    agent = make_peer_teammate(args.persona)
    print(f"[PEER] persona={args.persona} node={args.node_name} env={args.env_uuid}", flush=True)

    from aact.cli.launch.launch import _sync_run_node
    from aact.cli.reader import NodeConfig
    from aact.cli.reader.dataflow_reader import NodeArgs
    import collaborative_gym.nodes.agent_interface  # noqa: F401

    _sync_run_node(
        NodeConfig(
            node_name=args.node_name,
            node_class="agent",
            node_args=NodeArgs(
                env_uuid=args.env_uuid,
                node_name=args.node_name,
                agent=agent,
                wait_time=args.wait_time,
            ),
        ),
        args.redis_url,
    )


if __name__ == "__main__":
    main()
