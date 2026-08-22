#!/usr/bin/env python3
"""
End-to-end demo: HINT-Agent × Collaborative Gym (AHT paper arc).

Roles:
  - ego: HINT-Agent (CoT + Memory)
  - teammate: action-only PeerTeammateAgent persona (default), or legacy simulated_user
  - outside supervisor: off | oracle (scripted) | human_cli (Redis CLI)

Prerequisites: conda env `cogym`, Redis, OpenAI key in secrets.example.toml.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("hint_agent_cogym_demo")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COGYM_DIR = os.path.join(REPO_ROOT, "third_party", "collaborative-gym")
HINT_AGENT_PROJECT = os.path.join(REPO_ROOT, "AHT_human_intervention")
LAUNCHER_PATH = os.path.join(
    HINT_AGENT_PROJECT, "hintagent", "src", "hintagent", "hint_agent_cogym_launcher.py"
)
PEER_LAUNCHER_PATH = os.path.join(
    HINT_AGENT_PROJECT, "hintagent", "src", "hintagent", "teammates", "peer_teammate_launcher.py"
)
ORACLE_PATH = os.path.join(
    HINT_AGENT_PROJECT, "hintagent", "src", "hintagent", "oracle_supervisor.py"
)
TEAM_CONFIG_SIMULATED = os.path.join(
    HINT_AGENT_PROJECT, "configs", "teams",
    "hint_agent_cogym_simulated_user_team_config.toml",
)
TEAM_CONFIG_REAL_HUMAN = os.path.join(
    HINT_AGENT_PROJECT, "configs", "teams",
    "hint_agent_cogym_cmd_user_team_config.toml",
)
TEAM_CONFIG_SUPERVISOR = os.path.join(
    HINT_AGENT_PROJECT, "configs", "teams",
    "hint_agent_cogym_supervisor_team_config.toml",
)
SUPERVISOR_CLI = os.path.join(
    HINT_AGENT_PROJECT, "hintagent", "src", "hintagent", "supervisor_cli.py"
)
TEAM_CONFIG_PATH = TEAM_CONFIG_SUPERVISOR

PERSONA_CHOICES = ("idle", "complementary_searcher", "greedy_editor", "follower")

CONFIG_TEMPLATES = {
    "travel_planning": (
        'env_class = "travel_planning"\n\n'
        "[env_args]\nuse_simulated_dataset = true\n"
        "travel_planner_data_point_idx = {idx}\n"
    ),
    "related_work": (
        'env_class = "lit_survey"\n\n'
        "[env_args]\nuse_simulated_dataset = true\n"
        "simulated_data_point_idx = {idx}\n"
    ),
    "tabular_analysis": (
        'env_class = "tabular_analysis"\n\n'
        "[env_args]\nuse_simulated_dataset = true\n"
        "discovery_bench_data_point_idx = {idx}\n"
    ),
}


def _ensure_cogym_on_path() -> None:
    if not os.path.isdir(COGYM_DIR):
        raise SystemExit(
            f"Cannot find Co-Gym clone at {COGYM_DIR}.\n"
            "Please run: git clone https://github.com/SALT-NLP/collaborative-gym "
            "third_party/collaborative-gym"
        )
    if COGYM_DIR not in sys.path:
        sys.path.insert(0, COGYM_DIR)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HINT-Agent × Co-Gym demo (AHT paper arc).")
    parser.add_argument(
        "--task", type=str, default="travel_planning",
        choices=list(CONFIG_TEMPLATES.keys()),
    )
    parser.add_argument("--idx", type=int, default=0, help="Data point index in the simulated set.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--wait-time", type=int, default=1)
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--work-dir", type=str, default=os.path.join(HINT_AGENT_PROJECT, "run_logs", "cogym"))
    parser.add_argument("--result-dir-tag", type=str, default="hint_agent_demo")
    parser.add_argument(
        "--secrets-path",
        type=str,
        default="",
        help="TOML secrets file. Defaults to secrets.example.toml then secrets.toml under CoGym.",
    )
    parser.add_argument("--team-config-path", type=str, default="")
    parser.add_argument(
        "--teammate-persona",
        type=str,
        default="complementary_searcher",
        choices=list(PERSONA_CHOICES) + ["legacy_simulated_user"],
        help="Action-only peer persona (default) or legacy_simulated_user.",
    )
    parser.add_argument(
        "--supervisor-mode",
        type=str,
        default="human_cli",
        choices=["off", "oracle", "human_cli"],
        help="Outside supervisor: off | oracle (scripted) | human_cli (Redis CLI).",
    )
    parser.add_argument("--max-interventions", type=int, default=3, help="Oracle intervention budget.")
    parser.add_argument(
        "--memory-path",
        type=str,
        default="",
        help="Cross-episode intervention_patterns JSON (load + save).",
    )
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Alias for --supervisor-mode off (and legacy 2-party if also legacy teammate).",
    )
    parser.add_argument(
        "--real-human",
        action="store_true",
        help="Legacy: pair HINT-Agent with a real cmd_user teammate (human IS the teammate).",
    )
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--retrieval-strategy", type=str, default="semantic",
                        choices=["semantic", "random", "prepend_all"])
    parser.add_argument("--enhance-user-control", action="store_true")
    parser.add_argument("--session-uuid", type=str, default=None,
                        help="Optional deterministic session UUID.")
    args = parser.parse_args()

    if args.no_supervisor:
        args.supervisor_mode = "off"
    args.supervisor = args.supervisor_mode != "off" and not args.real_human

    if not args.team_config_path:
        if args.real_human:
            args.team_config_path = TEAM_CONFIG_REAL_HUMAN
        elif args.teammate_persona == "legacy_simulated_user":
            args.team_config_path = TEAM_CONFIG_SIMULATED
        else:
            args.team_config_path = TEAM_CONFIG_SUPERVISOR
    if args.real_human:
        args.enhance_user_control = True
    return args


def _build_hint_launcher_command(args: argparse.Namespace) -> str:
    parts = [
        "python", f'"{LAUNCHER_PATH}"',
        f"--model-name {args.model}",
        f"--wait-time {args.wait_time}",
        f"--retrieval-strategy {args.retrieval_strategy}",
    ]
    if args.no_cot:
        parts.append("--no-cot")
    if args.no_memory:
        parts.append("--no-memory")
    if args.enhance_user_control:
        parts.append("--enhance-user-control")
    if args.memory_path:
        parts.append(f'--memory-path "{args.memory_path}"')
    if args.supervisor_mode == "off":
        parts.append("--no-supervisor-listener")
    return " ".join(parts)


def _build_peer_launcher_command(args: argparse.Namespace) -> str:
    persona = args.teammate_persona
    if persona == "legacy_simulated_user":
        persona = "complementary_searcher"
    return (
        f'python "{PEER_LAUNCHER_PATH}" '
        f"--persona {persona} --wait-time {args.wait_time}"
    )


def _write_team_config(args: argparse.Namespace, hint_cmd: str, peer_cmd: str) -> str:
    import toml as _toml
    with open(args.team_config_path, "r") as f:
        data = _toml.load(f)
    for member in data.get("team_member", []):
        name = member.get("name")
        mtype = member.get("type")
        if name == "hint_agent" or (mtype == "agent" and name != "teammate"):
            member["start_node_base_command"] = hint_cmd
        elif name == "teammate" and mtype == "agent":
            member["start_node_base_command"] = peer_cmd
    resolved_path = os.path.join(
        os.path.dirname(args.team_config_path),
        "_resolved_hint_agent_cogym_team_config.toml",
    )
    with open(resolved_path, "w") as f:
        _toml.dump(data, f)
    return resolved_path


def main() -> None:
    args = parse_arguments()
    _ensure_cogym_on_path()

    import toml as _toml
    from collaborative_gym.core import TeamMemberConfig
    from collaborative_gym.runner import Runner

    secrets_candidates = []
    if args.secrets_path:
        secrets_candidates.append(args.secrets_path)
    secrets_candidates.extend([
        os.path.join(COGYM_DIR, "secrets.example.toml"),
        os.path.join(COGYM_DIR, "secrets.toml"),
        os.path.join(REPO_ROOT, "secrets.example.toml"),
        os.path.join(REPO_ROOT, "secrets.toml"),
    ])
    for path in secrets_candidates:
        if path and os.path.exists(path):
            secrets = _toml.load(path)
            for k, v in secrets.items():
                os.environ[k] = str(v)
            logger.info("Loaded secrets from %s", path)
            break
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Export it or put it in "
            "third_party/collaborative-gym/secrets.example.toml."
        )

    # Absolute memory path before baking into the AgentNode launcher command
    # (ego subprocess cwd is the CoGym clone).
    if args.memory_path:
        args.memory_path = os.path.abspath(args.memory_path)
        os.makedirs(os.path.dirname(args.memory_path) or ".", exist_ok=True)

    hint_cmd = _build_hint_launcher_command(args)
    peer_cmd = _build_peer_launcher_command(args)
    resolved_team_config = _write_team_config(args, hint_cmd, peer_cmd)

    result_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, "results")
    env_config_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, "env_config_tmp")
    os.makedirs(env_config_dir, exist_ok=True)
    env_config_path = os.path.join(env_config_dir, f"{args.task}_{args.idx}.toml")
    with open(env_config_path, "w") as f:
        f.write(CONFIG_TEMPLATES[args.task].format(idx=args.idx))

    session_uuid = args.session_uuid or f"{args.task}_{args.idx}_{uuid.uuid4().hex[:8]}"
    env_uuid = f"env_{session_uuid}"
    logger.info("Starting HINT-Agent × Co-Gym session %s", session_uuid)
    logger.info(
        "  task=%s idx=%d model=%s cot=%s memory=%s persona=%s supervisor=%s",
        args.task, args.idx, args.model, not args.no_cot, not args.no_memory,
        args.teammate_persona, args.supervisor_mode,
    )
    logger.info("  result_dir=%s env_uuid=%s", result_dir, env_uuid)
    if args.memory_path:
        logger.info("  memory_path=%s", args.memory_path)

    session_info_path = os.path.join(
        args.work_dir, args.task, args.result_dir_tag, "latest_session.json"
    )
    os.makedirs(os.path.dirname(session_info_path), exist_ok=True)
    with open(session_info_path, "w") as f:
        json.dump(
            {
                "session_uuid": session_uuid,
                "env_uuid": env_uuid,
                "redis_url": args.redis_url,
                "agent_node_name": "hint_agent",
                "task": args.task,
                "idx": args.idx,
                "teammate_persona": args.teammate_persona,
                "supervisor_mode": args.supervisor_mode,
                "memory_path": args.memory_path or None,
            },
            f,
            indent=2,
        )

    oracle_proc: subprocess.Popen | None = None
    if args.supervisor_mode == "human_cli":
        print("\n" + "=" * 70)
        print("OUTSIDE SUPERVISOR (human_cli, real-time, no pause)")
        print("  In a SECOND terminal, run:")
        print(
            f'  python "{SUPERVISOR_CLI}" --env-uuid {env_uuid} '
            f'--redis-url {args.redis_url}'
        )
        print(f"  Session info: {session_info_path}")
        print("=" * 70 + "\n")
    elif args.supervisor_mode == "oracle":
        print("\n" + "=" * 70)
        print(f"OUTSIDE SUPERVISOR (oracle, max_interventions={args.max_interventions})")
        print("=" * 70 + "\n")

    runner = Runner(result_dir=result_dir, redis_url=args.redis_url)

    def _cleanup():
        if oracle_proc is not None and oracle_proc.poll() is None:
            try:
                oracle_proc.terminate()
                oracle_proc.wait(timeout=5)
            except Exception:
                try:
                    oracle_proc.kill()
                except Exception:
                    pass
        runner.cleanup_subprocesses()

    def _handle_exit(signum, frame):
        logger.info("Signal %s received; cleaning up.", signum)
        _cleanup()
        sys.exit(0)

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    saved_cwd = os.getcwd()
    os.chdir(COGYM_DIR)
    try:
        team_member_config = _toml.load(resolved_team_config)
        runner.start_session(
            session_uuid=session_uuid,
            env_config_path=env_config_path,
            members=[TeamMemberConfig(**m) for m in team_member_config["team_member"]],
            max_steps=args.max_steps,
            disable_collaboration=False,
            add_tick=True,
        )

        if args.supervisor_mode == "oracle":
            # Start after agents so Redis subscriptions are live.
            time.sleep(2)
            oracle_cmd = [
                sys.executable,
                ORACLE_PATH,
                "--env-uuid", env_uuid,
                "--redis-url", args.redis_url,
                "--idx", str(args.idx),
                "--max-interventions", str(args.max_interventions),
                "--agent-node-name", "hint_agent",
            ]
            oracle_proc = subprocess.Popen(oracle_cmd)
            logger.info("Started oracle supervisor pid=%s", oracle_proc.pid)

        started = time.time()
        for proc in runner.subprocesses:
            proc.wait()
        elapsed = (time.time() - started) / 60.0
        logger.info("Session %s finished in %.2f minutes.", session_uuid, elapsed)
    finally:
        os.chdir(saved_cwd)
        if oracle_proc is not None and oracle_proc.poll() is None:
            try:
                oracle_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
