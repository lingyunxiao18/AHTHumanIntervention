#!/usr/bin/env python3
"""
HINT-Agent for Collaborative Gym (Co-Gym).

Implements the AgentNode contract (start / get_action / end) so it can be
launched by `collaborative_gym.nodes.agent_interface.AgentNode` exactly like
the built-in `OneStageCollaborativeAgent` and `CollaborativeAgent`.

Architecture mirrors the Overcooked and CrowdNav HINT-Agent variants:
- Planner: `AdvancedLLMInterpreter` (CoT + Memory) proposes one Co-Gym action.
- Verifier: regex fullmatch against the runtime action space; if the LLM
  emits an invalid action, we retry once and then fall back to
  WAIT_TEAMMATE_CONTINUE().
- Event detector: task-agnostic signals (lack_of_progress, repeated_action,
  stalled_teammate, chat_pending, pending_confirmation) drive proactive
  ICL retrieval of past successful interventions.
- Human intervention pipeline: new user messages in `chat_history` (since the
  last processed timestamp) are consumed as `HumanMessage` objects and fed
  through the same intervention-pattern-learning loop used in Overcooked.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Path setup -------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# The intervention module sits next door; add its directory for both
# package-style and script-style imports.
_intervention_path = os.path.join(PROJECT_ROOT, "hintagent", "src", "human_intervention")
if _intervention_path not in sys.path:
    sys.path.append(_intervention_path)

try:
    from ..human_intervention.advanced_llm_intervention_cogym import (  # type: ignore
        AdvancedLLMInterpreter,
        AgentMemory,
        HumanMessage,
        LLMClient,
        Plan,
        action_space_to_description,
    )
except ImportError:
    from advanced_llm_intervention_cogym import (  # type: ignore
        AdvancedLLMInterpreter,
        AgentMemory,
        HumanMessage,
        LLMClient,
        Plan,
        action_space_to_description,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _obs_signature(obs: Dict[str, Any]) -> str:
    """Content hash of the observation (task-agnostic)."""
    try:
        serialised = json.dumps(obs, sort_keys=True, default=str)
    except Exception:
        serialised = repr(obs)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()[:16]


def _summarise_field(value: Any, max_chars: int = 400) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        s = str(value).strip()
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            s = str(value)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


def _summarise_observation(obs: Dict[str, Any]) -> str:
    """Compact, task-agnostic textual summary of a Co-Gym observation dict."""
    if not isinstance(obs, dict):
        return _summarise_field(obs, max_chars=800)
    parts: List[str] = []
    for key, value in obs.items():
        parts.append(f"- {key}: {_summarise_field(value, max_chars=600)}")
    return "\n".join(parts) if parts else "(empty observation)"


def _new_user_messages(
    chat_history: List[Dict[str, Any]],
    self_role: Optional[str],
    since_len: int,
) -> List[Dict[str, Any]]:
    """Return chat entries added since the last processed observation."""
    if not chat_history:
        return []
    new_entries = chat_history[since_len:]
    return [
        turn for turn in new_entries
        if turn.get("role") and turn.get("role") != self_role
    ]


# ---------------------------------------------------------------------
# Scratchpad (task-agnostic bounded notes)
# ---------------------------------------------------------------------


@dataclass
class SimpleScratchpad:
    """A bounded free-form scratchpad. The interpreter proposes updates as text;
    we cap total size and keep a simple append/replace log."""

    entries: List[str] = field(default_factory=list)
    max_entries: int = 20
    max_chars: int = 4096

    def apply(self, update: str) -> None:
        update = (update or "").strip()
        if not update or update.lower() in {"none", "n/a", "no change"}:
            return
        self.entries.append(update)
        # Enforce caps
        while len(self.entries) > self.max_entries:
            self.entries.pop(0)
        while sum(len(e) for e in self.entries) > self.max_chars and self.entries:
            self.entries.pop(0)

    def to_str(self) -> str:
        if not self.entries:
            return "(empty)"
        return "\n".join(f"[{i+1}] {e}" for i, e in enumerate(self.entries))


# ---------------------------------------------------------------------
# HINTAgentCoGym
# ---------------------------------------------------------------------


COLLAB_ACTIONS = {
    "SEND_TEAMMATE_MESSAGE": {
        "pattern": re.compile(r"^SEND_TEAMMATE_MESSAGE\(message=(.*)\)$", re.DOTALL),
        "template": 'SEND_TEAMMATE_MESSAGE(message="{message}")',
    },
    "WAIT_TEAMMATE_CONTINUE": {
        "pattern": re.compile(r"^WAIT_TEAMMATE_CONTINUE\(\)$", re.DOTALL),
        "template": "WAIT_TEAMMATE_CONTINUE()",
    },
    "REQUEST_TEAMMATE_CONFIRM": {
        "pattern": re.compile(
            r"^REQUEST_TEAMMATE_CONFIRM\(request_id=(.*), pending_action=(.*)\)$",
            re.DOTALL,
        ),
        "template": 'REQUEST_TEAMMATE_CONFIRM(request_id="{request_id}", pending_action="{pending_action}")',
    },
}


class HINTAgentCoGym:
    """
    HINT-Agent for the Collaborative Gym `AgentNode` interface.

    Args:
        model: OpenAI model name (default: gpt-5-mini). The interpreter
            uses OpenAI's JSON mode.
        history_horizon: How many prior psi_t snapshots to include per prompt.
        enable_cot / enable_memory: Ablation switches (mirror Overcooked
            variant).
        retrieval_strategy: 'semantic' | 'random' | 'prepend_all'.
        no_progress_window: How many consecutive obs-unchanged agent steps
            before firing `lack_of_progress`.
        cycle_action_window: Repeat count of exact action string that fires
            `repeated_action`.
        teammate_idle_window: How many consecutive agent steps with no user
            action / user message before firing `stalled_teammate`.
        max_action_retries: How many times to re-prompt the LLM if the
            emitted action doesn't fullmatch any pattern.
        enhance_user_control: If True, wrap EDITOR_UPDATE-like shared-state
            actions in REQUEST_TEAMMATE_CONFIRM when available.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        history_horizon: int = 3,
        enable_cot: bool = True,
        enable_memory: bool = True,
        retrieval_strategy: str = "semantic",
        retrieval_seed: int = 0,
        no_progress_window: int = 4,
        cycle_action_window: int = 2,
        teammate_idle_window: int = 4,
        max_action_retries: int = 1,
        enhance_user_control: bool = False,
        # Paper setting: outside supervisor intervenes on ego; teammate is simulated.
        # Teammate chat is context only; supervisor inbox is the HumanMessage channel.
        treat_teammate_chat_as_intervention: bool = False,
        # Ego should not chat with the simulated teammate (agents don't talk among themselves).
        disable_send_teammate_message: bool = True,
        # Cross-episode intervention memory (JSON path); load at init, merge-save at end.
        memory_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.history_horizon = history_horizon
        self.enable_cot = enable_cot
        self.enable_memory = enable_memory
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_seed = retrieval_seed
        self.no_progress_window = no_progress_window
        self.cycle_action_window = cycle_action_window
        self.teammate_idle_window = teammate_idle_window
        self.max_action_retries = max_action_retries
        self.enhance_user_control = enhance_user_control
        self.treat_teammate_chat_as_intervention = treat_teammate_chat_as_intervention
        self.disable_send_teammate_message = disable_send_teammate_message
        self.memory_path = memory_path
        self.verbose = verbose

        # Populated by `start()`
        self.name: Optional[str] = None
        self.team_members: List[str] = []
        self.task_description: str = ""
        self.action_space: List[Dict[str, Any]] = []
        self.action_patterns: List[Tuple[str, re.Pattern]] = []
        self.action_space_description: str = ""
        self.example_question: str = ""
        self.example_trajectory: List = []

        # LLM setup: OpenAI key comes from environment. Falls back to secrets
        # (CoGym's convention) when running under the AgentNode subprocess.
        self.openai_api_keys: List[str] = []
        self._load_openai_keys()
        from openai import OpenAI

        self.llm_client = LLMClient(
            openai_client=OpenAI(api_key=self._openai_api_key()), model=self.model
        )
        self.memory = AgentMemory()
        if self.memory_path:
            self.load_intervention_memory(self.memory_path)
        self.interpreter = AdvancedLLMInterpreter(
            self.llm_client,
            self.memory,
            history_horizon=history_horizon,
            enable_cot=enable_cot,
            enable_memory=enable_memory,
            retrieval_strategy=retrieval_strategy,
            retrieval_seed=retrieval_seed,
        )

        # State
        self.scratchpad = SimpleScratchpad()
        self._recent_psi: deque = deque(maxlen=history_horizon * 4)
        self._prev_obs_sig: Optional[str] = None
        self._no_progress_count: int = 0
        self._recent_actions: deque = deque(maxlen=cycle_action_window * 2)
        self._teammate_idle_count: int = 0
        self._last_processed_chat_len: int = 0
        self._current_timestep: int = 0

        # Intervention outcome tracking
        self._pending_intervention: Optional[Dict[str, Any]] = None
        self._intervention_success_window: int = max(3, teammate_idle_window)

        # Debug / logging
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_events: List[str] = []
        self.last_prompt_payload: Optional[Dict[str, Any]] = None
        self._action_log: List[Dict[str, Any]] = []
        self._intervention_history: List[str] = []

        # Outside supervisor inbox (thread-safe). Filled by Redis subscriber in the
        # launcher; consumed in get_action without pausing the CoGym session.
        self._supervisor_inbox: List[str] = []
        self._supervisor_lock = threading.Lock()
        self._last_supervisor_text: Optional[str] = None

    # -------- OpenAI key handling --------

    def _load_openai_keys(self) -> None:
        # 1. Env var
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            self.openai_api_keys = [key.strip()]
            return
        # 2. Fall back to CoGym secrets files (secrets.toml or secrets.example.toml)
        try:
            import toml  # optional dep, ships with CoGym
            here = os.getcwd()
            candidates = []
            for base in (here, os.path.join(here, ".."), os.path.dirname(__file__)):
                candidates.extend([
                    os.path.join(base, "secrets.toml"),
                    os.path.join(base, "secrets.example.toml"),
                ])
            # Also try the vendored CoGym clone relative to this package
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            candidates.extend([
                os.path.join(repo_root, "third_party", "collaborative-gym", "secrets.toml"),
                os.path.join(repo_root, "third_party", "collaborative-gym", "secrets.example.toml"),
            ])
            for candidate in candidates:
                if os.path.exists(candidate):
                    data = toml.load(candidate)
                    if data.get("OPENAI_API_KEY"):
                        self.openai_api_keys = [str(data["OPENAI_API_KEY"]).strip()]
                        return
        except Exception:
            pass
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set the env var or put it in "
            "third_party/collaborative-gym/secrets.example.toml (or secrets.toml)."
        )

    def _openai_api_key(self) -> str:
        return self.openai_api_keys[0]

    # ==================== AgentNode interface ====================

    def start(
        self,
        name: str,
        team_members: List[str],
        task_description: str,
        action_space: List[Dict[str, Any]],
        example_question: str,
        example_trajectory: List,
    ) -> None:
        self.name = name
        self.team_members = list(team_members)
        self.task_description = task_description
        self.action_space = list(action_space or [])
        self.action_patterns = []
        for space in self.action_space:
            pat = space.get("pattern")
            if not pat:
                continue
            try:
                self.action_patterns.append(
                    (space.get("machine_readable_identifier", "?"), re.compile(pat, re.DOTALL))
                )
            except re.error:
                logger.warning("Invalid regex in action_space: %r", pat)
        self.action_space_description = action_space_to_description(self.action_space)
        self.example_question = example_question or ""
        self.example_trajectory = list(example_trajectory or [])
        logger.info(
            "HINTAgentCoGym started as '%s' with %d action patterns (CoT=%s, Memory=%s).",
            self.name,
            len(self.action_patterns),
            self.enable_cot,
            self.enable_memory,
        )

    def get_action(self, observation: Dict[str, Any], chat_history: List[Dict[str, Any]]) -> str:
        self._current_timestep += 1

        # 1) Detect events; teammate chat is observational unless explicitly enabled.
        events, teammate_chat = self._detect_events_and_human(observation, chat_history)
        self.last_events = list(events)

        # 2) Outside supervisor interventions take priority (paper setting).
        supervisor_text = self._pop_supervisor_intervention()
        if supervisor_text:
            if "supervisor_intervention" not in events:
                events.append("supervisor_intervention")
            human_msg_text = supervisor_text
            self._last_supervisor_text = supervisor_text
        elif self.treat_teammate_chat_as_intervention and teammate_chat:
            human_msg_text = teammate_chat
        else:
            human_msg_text = None

        # 3) Build psi_t (includes teammate chat as context, even if not an intervention).
        psi_text = self._build_psi(observation, chat_history)
        if supervisor_text:
            psi_text += f"\nOUTSIDE_SUPERVISOR: {supervisor_text}"
        self._recent_psi.append(psi_text)

        # 4) Intervention pattern bookkeeping (supervisor / optional teammate-as-intervention).
        if human_msg_text and self.enable_memory:
            self._pending_intervention = {
                "timestamp": self._current_timestep,
                "start_t": self._current_timestep,
                "text": human_msg_text,
                "source": "supervisor" if supervisor_text else "teammate_chat",
                "events": list(events),
                "start_events": set(events),
                "start_obs_sig": self._prev_obs_sig,
            }
            self._intervention_history.append(human_msg_text)

        human_msg = HumanMessage(t=self._current_timestep, text=human_msg_text or "")

        # 5) CoT+Memory → alternative action accommodating the supervisor if present.
        plan, action = self._plan_with_verification(
            psi_text=psi_text,
            events=events,
            human_msg=human_msg,
        )
        self.last_plan = plan.to_dict() if plan else None

        if plan and plan.scratchpad_update:
            self.scratchpad.apply(plan.scratchpad_update)

        if self.enhance_user_control:
            action = self._maybe_wrap_in_confirmation(action)

        # Paper setting: ego does not message the simulated teammate.
        if self.disable_send_teammate_message and action.startswith("SEND_TEAMMATE_MESSAGE"):
            logger.info(
                "HINTAgentCoGym suppressed SEND_TEAMMATE_MESSAGE (agents do not chat); "
                "falling back to WAIT_TEAMMATE_CONTINUE()."
            )
            action = "WAIT_TEAMMATE_CONTINUE()"

        self._recent_actions.append(action)
        self._last_processed_chat_len = len(chat_history or [])
        self._update_intervention_outcome(events, observation)

        self._action_log.append({
            "t": self._current_timestep,
            "events": list(events),
            "supervisor_message": supervisor_text or "",
            "teammate_chat": teammate_chat or "",
            "human_message": human_msg_text or "",
            "action": action,
            "plan": self.last_plan,
        })

        logger.info(
            "HINTAgentCoGym[%s] t=%d events=%s supervisor=%r teammate_chat=%r -> %s",
            self.name,
            self._current_timestep,
            events,
            supervisor_text or "",
            (teammate_chat or "")[:80],
            action,
        )
        return action

    # ==================== Outside supervisor API ====================

    def apply_human_intervention(self, text: str) -> None:
        """
        Queue an outside-supervisor intervention (thread-safe, non-pausing).

        The CoGym session keeps running. The text is consumed on the next
        ``get_action`` and routed through CoT+Memory as a HumanMessage.
        """
        if text and text.strip():
            with self._supervisor_lock:
                self._supervisor_inbox.append(text.strip())
            logger.info("Supervisor intervention queued: %r", text.strip()[:160])

    def process_human_intervention(self, text: str) -> bool:
        try:
            self.apply_human_intervention(text)
            return True
        except Exception:
            return False

    def _pop_supervisor_intervention(self) -> Optional[str]:
        with self._supervisor_lock:
            if not self._supervisor_inbox:
                return None
            # Drain all pending lines into one intervention blob (FIFO).
            parts = list(self._supervisor_inbox)
            self._supervisor_inbox.clear()
        return "\n".join(parts)

    def load_intervention_memory(self, path: str) -> int:
        """Load intervention_patterns from a prior episode JSON file. Returns count loaded."""
        if not path or not os.path.exists(path):
            return 0
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                patterns = data.get("intervention_patterns") or data.get("patterns") or []
            elif isinstance(data, list):
                patterns = data
            else:
                patterns = []
            existing = self.memory.semantic.setdefault("intervention_patterns", [])
            # Dedup by human_message + corrected_action when present
            seen = {
                (p.get("human_message"), p.get("corrected_action") or p.get("action"))
                for p in existing
                if isinstance(p, dict)
            }
            added = 0
            for p in patterns:
                if not isinstance(p, dict):
                    continue
                key = (p.get("human_message"), p.get("corrected_action") or p.get("action"))
                if key in seen:
                    continue
                existing.append(p)
                seen.add(key)
                added += 1
            logger.info("Loaded %d intervention patterns from %s", added, path)
            return added
        except Exception as e:
            logger.warning("Failed to load intervention memory from %s: %s", path, e)
            return 0

    def save_intervention_memory(self, path: Optional[str] = None) -> None:
        """Persist intervention_patterns for cross-episode transfer."""
        path = path or self.memory_path
        if not path or not self.enable_memory:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            patterns = [
                {k: v for k, v in p.items() if k != "embedding_ctx"}
                for p in self.memory.semantic.get("intervention_patterns", [])
            ]
            # Merge with any on-disk patterns from parallel runs
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        prior = json.load(f)
                    prior_list = prior if isinstance(prior, list) else prior.get("intervention_patterns", [])
                    seen = {
                        (p.get("human_message"), p.get("corrected_action") or p.get("action"))
                        for p in patterns
                    }
                    for p in prior_list or []:
                        if not isinstance(p, dict):
                            continue
                        key = (p.get("human_message"), p.get("corrected_action") or p.get("action"))
                        if key not in seen:
                            patterns.append({k: v for k, v in p.items() if k != "embedding_ctx"})
                            seen.add(key)
                except Exception:
                    pass
            with open(path, "w") as f:
                json.dump(patterns, f, indent=2, default=str)
            logger.info("Saved %d intervention patterns to %s", len(patterns), path)
        except Exception as e:
            logger.warning("Failed to save intervention memory to %s: %s", path, e)

    def end(self, result_dir: str) -> None:
        try:
            out_dir = os.path.join(result_dir, self.name or "hint_agent_cogym")
            os.makedirs(out_dir, exist_ok=True)
            info = {
                "agent": "HINTAgentCoGym",
                "model": self.model,
                "enable_cot": self.enable_cot,
                "enable_memory": self.enable_memory,
                "retrieval_strategy": self.retrieval_strategy,
                "memory_path": self.memory_path,
                "total_steps": self._current_timestep,
                "total_interventions": len(self._intervention_history),
            }
            with open(os.path.join(out_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=4)
            with open(os.path.join(out_dir, "scratchpad.txt"), "w") as f:
                f.write(self.scratchpad.to_str())
            with open(os.path.join(out_dir, "action_log.jsonl"), "w") as f:
                for entry in self._action_log:
                    f.write(json.dumps(entry, default=str) + "\n")
            if self.enable_memory:
                patterns = [
                    {k: v for k, v in p.items() if k != "embedding_ctx"}
                    for p in self.memory.semantic.get("intervention_patterns", [])
                ]
                with open(os.path.join(out_dir, "intervention_patterns.json"), "w") as f:
                    json.dump(patterns, f, indent=2, default=str)
                self.save_intervention_memory(self.memory_path)
        except Exception as e:
            logger.warning("HINTAgentCoGym.end failed to write logs: %s", e)

    # ==================== Internal: psi / events / planning ====================

    def _build_psi(
        self, observation: Dict[str, Any], chat_history: List[Dict[str, Any]]
    ) -> str:
        obs_summary = _summarise_observation(observation)
        recent_chat = ""
        if chat_history:
            tail = chat_history[-4:]
            recent_chat = "\n".join(
                f"{turn.get('role', '?')}: {_summarise_field(turn.get('message', ''), 200)}"
                for turn in tail
            )
        parts = [
            f"TASK: {_summarise_field(self.task_description, 400)}",
            f"OBSERVATION:\n{obs_summary}",
        ]
        if recent_chat:
            parts.append(f"RECENT CHAT:\n{recent_chat}")
        if self.scratchpad.entries:
            parts.append(f"SCRATCHPAD:\n{self.scratchpad.to_str()}")
        return "\n".join(parts)

    def _detect_events_and_human(
        self,
        observation: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> Tuple[List[str], Optional[str]]:
        events: List[str] = []
        # (a) obs delta
        sig = _obs_signature(observation)
        if self._prev_obs_sig is not None and sig == self._prev_obs_sig:
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0
        self._prev_obs_sig = sig
        if self._no_progress_count >= self.no_progress_window:
            events.append("lack_of_progress")

        # (b) repeated action
        if len(self._recent_actions) >= self.cycle_action_window:
            recent_slice = list(self._recent_actions)[-self.cycle_action_window:]
            if len(set(recent_slice)) == 1 and recent_slice[0]:
                events.append("repeated_action")

        # (c) new user messages
        new_user = _new_user_messages(chat_history or [], self.name, self._last_processed_chat_len)
        human_msg_text: Optional[str] = None
        if new_user:
            human_msg_text = "\n".join(
                _summarise_field(t.get("message", ""), 800) for t in new_user
            )
            events.append("chat_pending")
            self._teammate_idle_count = 0
        else:
            self._teammate_idle_count += 1
            if self._teammate_idle_count >= self.teammate_idle_window:
                events.append("stalled_teammate")

        # (d) pending confirmations - inspect observation for confirmation markers
        try:
            pending_conf = observation.get("pending_confirmations") if isinstance(observation, dict) else None
            if pending_conf:
                events.append("pending_confirmation")
        except Exception:
            pass

        return events, human_msg_text

    def _plan_with_verification(
        self,
        psi_text: str,
        events: List[str],
        human_msg: HumanMessage,
    ) -> Tuple[Optional[Plan], str]:
        """Ask interpreter for a plan; retry once with an explicit invalidity hint."""
        recent_history = list(self._recent_psi)[-self.history_horizon:]
        plan: Optional[Plan] = None
        action: Optional[str] = None

        for attempt in range(self.max_action_retries + 1):
            plan = self.interpreter.propose_plan(
                psi_text=psi_text,
                recent_history=recent_history,
                human_msg=human_msg,
                action_space_desc=self.action_space_description,
                task_description=self.task_description,
                events=events,
                scratchpad_text=self.scratchpad.to_str(),
            )
            candidate = plan.steps[0].strip() if plan and plan.steps else ""
            candidate = self._sanitise_action(candidate)
            if self._is_valid_action(candidate):
                action = candidate
                # Overwrite the sanitised action back into the plan for logging
                plan.steps[0] = candidate
                break

            if attempt < self.max_action_retries:
                psi_text = (
                    psi_text
                    + "\n\nATTENTION: your previous emitted action did not fullmatch any "
                    "regex from action_space_description. Re-read the regex patterns "
                    "and emit a fullmatch-valid action string this time."
                )

        if action is None:
            action = "WAIT_TEAMMATE_CONTINUE()"
        return plan, action

    def _sanitise_action(self, candidate: str) -> str:
        """Strip common LLM decorations (Thought:, code fences, backticks)."""
        s = (candidate or "").strip()
        if not s:
            return s
        # Strip code fences
        if s.startswith("```"):
            s = re.sub(r"^```[\w-]*\n?", "", s)
            s = re.sub(r"\n?```\s*$", "", s)
        # Strip Thought:/Action: prefixes that some models emit even in JSON strings
        if "Action:" in s:
            s = s[s.find("Action:") + len("Action:"):].strip()
        if s.startswith("Thought:"):
            idx = s.find("Action:")
            if idx != -1:
                s = s[idx + len("Action:"):].strip()
        # Cut off any trailing "\nThought:" tail
        t_idx = s.find("\nThought:")
        if t_idx != -1:
            s = s[:t_idx].strip()
        # Normalise escaped parens
        s = s.replace("\\(", "(").replace("\\)", ")")
        return s.strip()

    def _is_valid_action(self, action: str) -> bool:
        if not action:
            return False
        # Task action space
        for _mid, pat in self.action_patterns:
            if pat.fullmatch(action):
                return True
        # Collaboration actions (always allowed when collaboration is enabled)
        for data in COLLAB_ACTIONS.values():
            if data["pattern"].fullmatch(action):
                return True
        return False

    def _maybe_wrap_in_confirmation(self, action: str) -> str:
        """If the action is a broadcast-editing action, wrap it in a confirmation."""
        # Heuristic: any action whose identifier ends with _UPDATE and takes a
        # 'text' parameter is a shared-state mutation.
        if not action:
            return action
        m = re.match(r"^([A-Z][A-Z_]+)\(", action)
        if not m:
            return action
        ident = m.group(1)
        for space in self.action_space:
            if space.get("machine_readable_identifier") != ident:
                continue
            if ident.endswith("_UPDATE") and "text" in (space.get("params") or []):
                # Check that REQUEST_TEAMMATE_CONFIRM is a valid collab action here.
                return COLLAB_ACTIONS["REQUEST_TEAMMATE_CONFIRM"]["template"].format(
                    request_id=ident.lower(),
                    pending_action=action.replace('"', '\\"'),
                )
        return action

    # ==================== Intervention outcome tracking ====================

    def _update_intervention_outcome(
        self, current_events: List[str], observation: Dict[str, Any]
    ) -> None:
        """Commit or discard a pending intervention pattern based on progress signals.

        Success:
            - Any failure event present at intervention start is no longer active.
            - AND we observe an observation change within the success window.
        Failure:
            - Window elapsed without meeting the success condition.
        """
        if not self._pending_intervention or not self.enable_memory:
            return
        pending = self._pending_intervention
        start_t = pending.get("start_t", self._current_timestep)
        steps_since = self._current_timestep - int(start_t)
        window = self._intervention_success_window
        start_events = pending.get("start_events") or set()
        current_events_set = set(current_events)

        obs_changed = (self._prev_obs_sig is not None) and (
            self._prev_obs_sig != pending.get("start_obs_sig")
        )

        failure_events_now = start_events.intersection(
            {"lack_of_progress", "repeated_action", "stalled_teammate"}
        )
        failure_events_now = failure_events_now.intersection(current_events_set)

        if steps_since >= 1 and not failure_events_now and obs_changed:
            self.interpreter.commit_intervention_pattern(pending["timestamp"])
            self._pending_intervention = None
            return

        if steps_since >= window:
            self.interpreter.discard_intervention_pattern(pending["timestamp"])
            self._pending_intervention = None

    # ==================== Public introspection helpers ====================

    def get_intervention_history(self) -> List[str]:
        return list(self._intervention_history)

    def get_intervention_stats(self) -> Dict[str, Any]:
        return {
            "total_interventions": len(self._intervention_history),
            "history_length": len(self._recent_psi),
            "memory_entries": len(getattr(self.memory, "episodic", [])),
            "intervention_patterns": len(
                getattr(self.memory, "semantic", {}).get("intervention_patterns", [])
            ),
            "current_timestep": self._current_timestep,
            "last_events": list(self.last_events),
        }
