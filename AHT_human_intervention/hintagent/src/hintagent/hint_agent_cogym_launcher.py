#!/usr/bin/env python3
"""
Launcher script that instantiates a `HINTAgentCoGym` inside a Co-Gym
`AgentNode` subprocess. Called by `collaborative_gym.runner.Runner` via the
team-config TOML file `configs/teams/hint_agent_cogym_*.toml`.

Also starts a background Redis subscriber so an outside supervisor can inject
interventions in real time *without pausing* the session (see supervisor_cli.py).
"""
from __future__ import annotations

import argparse
import os
import sys

# Add project root so we can import hintagent
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure the hintagent src tree is importable
_HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "hintagent", "src"))
if _HINTAGENT_SRC not in sys.path:
    sys.path.insert(0, _HINTAGENT_SRC)

# CoGym is vendored (not necessarily pip-installed); AgentNode subprocesses need it.
_COGYM_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "third_party", "collaborative-gym"))
if os.path.isdir(_COGYM_DIR) and _COGYM_DIR not in sys.path:
    sys.path.insert(0, _COGYM_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HINTAgentCoGym as a Co-Gym AgentNode.")
    parser.add_argument("--node-name", type=str, required=True)
    parser.add_argument("--env-uuid", type=str, required=True)
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--model-name", type=str, default="gpt-5-mini")
    parser.add_argument("--wait-time", type=int, default=1)
    parser.add_argument("--history-horizon", type=int, default=3)
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought reasoning.")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory / ICL retrieval.")
    parser.add_argument(
        "--retrieval-strategy",
        type=str,
        default="semantic",
        choices=["semantic", "random", "prepend_all"],
    )
    parser.add_argument("--retrieval-seed", type=int, default=0)
    parser.add_argument("--no-progress-window", type=int, default=4)
    parser.add_argument("--cycle-action-window", type=int, default=2)
    parser.add_argument("--teammate-idle-window", type=int, default=4)
    parser.add_argument("--enhance-user-control", action="store_true")
    parser.add_argument(
        "--treat-teammate-chat-as-intervention",
        action="store_true",
        help="If set, teammate chat also counts as HumanMessage (legacy 2-party mode).",
    )
    parser.add_argument(
        "--allow-send-teammate-message",
        action="store_true",
        help="Allow ego to SEND_TEAMMATE_MESSAGE (default: suppressed for paper setting).",
    )
    parser.add_argument(
        "--no-supervisor-listener",
        action="store_true",
        help="Do not subscribe to Redis supervisor interventions.",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default="",
        help="Load/save cross-episode intervention_patterns JSON.",
    )
    parser.add_argument(
        "--secrets-path",
        type=str,
        default="",
        help="TOML secrets file. Defaults to secrets.toml, then secrets.example.toml.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load secrets into env (prefer secrets.toml, then secrets.example.toml)
    secrets_candidates = []
    if args.secrets_path:
        secrets_candidates.append(args.secrets_path)
    secrets_candidates.extend(["secrets.toml", "secrets.example.toml"])
    # Also try absolute CoGym clone secrets from repo root
    secrets_candidates.extend([
        os.path.join(PROJECT_ROOT, "..", "third_party", "collaborative-gym", "secrets.example.toml"),
        os.path.join(PROJECT_ROOT, "..", "third_party", "collaborative-gym", "secrets.toml"),
    ])
    for path in secrets_candidates:
        if path and os.path.exists(path):
            import toml
            secrets = toml.load(path)
            for k, v in secrets.items():
                os.environ[k] = str(v)
            break

    from hintagent.hint_agent_cogym import HINTAgentCoGym  # after path fix-up

    agent = HINTAgentCoGym(
        model=args.model_name,
        history_horizon=args.history_horizon,
        enable_cot=not args.no_cot,
        enable_memory=not args.no_memory,
        retrieval_strategy=args.retrieval_strategy,
        retrieval_seed=args.retrieval_seed,
        no_progress_window=args.no_progress_window,
        cycle_action_window=args.cycle_action_window,
        teammate_idle_window=args.teammate_idle_window,
        enhance_user_control=args.enhance_user_control,
        treat_teammate_chat_as_intervention=args.treat_teammate_chat_as_intervention,
        disable_send_teammate_message=not args.allow_send_teammate_message,
        memory_path=args.memory_path or None,
    )

    # Real-time outside supervisor (does not pause the session).
    if not args.no_supervisor_listener:
        try:
            from hintagent.supervisor_channel import start_intervention_subscriber

            start_intervention_subscriber(
                redis_url=args.redis_url,
                env_uuid=args.env_uuid,
                on_message=agent.apply_human_intervention,
                agent_node_name=args.node_name,
            )
            print(
                f"[HINT] Supervisor listener on "
                f"{args.env_uuid}/{args.node_name}/supervisor_intervention",
                flush=True,
            )
        except Exception as e:
            print(f"[HINT] Warning: could not start supervisor listener: {e}", flush=True)

    if args.debug:
        import pdb; pdb.set_trace()
        return

    # Import aact / CoGym only when actually running (they're heavy).
    from aact.cli.launch.launch import _sync_run_node
    from aact.cli.reader import NodeConfig
    from aact.cli.reader.dataflow_reader import NodeArgs
    import collaborative_gym.nodes.agent_interface  # noqa: F401  # register 'agent' node

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
