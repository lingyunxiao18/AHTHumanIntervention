#!/usr/bin/env python3
"""
CoT + Memory Interpreter for HINT-Agent in Collaborative Gym (Co-Gym).

Mirrors the Overcooked / CrowdNav interpreters:
- AgentMemory: episodic events + semantic intervention_patterns.
- LLMClient: OpenAI JSON-mode wrapper (compatible with gpt-5-mini).
- AdvancedLLMInterpreter: builds the CoT+Memory prompt, retrieves ICL examples
  when failure signals fire without a fresh human message, and returns a Plan
  containing a single fully-formed Co-Gym action string.

Key differences vs. Overcooked/CrowdNav:
- Action space is dynamic and task-specific (regex-parameterized action strings
  passed in at runtime via `start()`), so the JSON schema does NOT enum-lock
  action names. Instead, we validate against the runtime `action_space` via
  regex fullmatch after generation.
- Human intervention arrives via `chat_history` from the environment, not
  through an external inbox call. The agent detects new user messages and
  passes them here as `HumanMessage`.
"""
import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from openai import OpenAI

try:
    from jsonschema import ValidationError, validate
except ImportError:  # jsonschema is optional
    def validate(instance, schema):  # type: ignore[no-redef]
        pass

    class ValidationError(Exception):  # type: ignore[no-redef]
        pass


Category = Literal["policy", "env", "teammate", "general_hint"]
RetrievalStrategy = Literal["semantic", "random", "prepend_all"]


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclass
class HumanMessage:
    t: int
    text: str


@dataclass
class Plan:
    steps: List[str]  # exactly one full Co-Gym action string
    chain_of_thought: str
    category: Category
    teammate_behavior: str = ""
    scratchpad_update: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "chain_of_thought": self.chain_of_thought,
            "category": self.category,
            "teammate_behavior": self.teammate_behavior,
            "scratchpad_update": self.scratchpad_update,
        }


# ---------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------


class AgentMemory:
    """
    Co-Gym persistent memory. Same shape as the Overcooked variant so the
    HINT-Agent code path stays consistent.
    - semantic.intervention_patterns: successful (or discarded) intervention
      exemplars, retrievable by state+teammate embedding.
    - episodic: recent plans / human interventions / obs summaries used for
      the "Recent" one-line summary in the prompt.
    """

    def __init__(self, episodic_cap: int = 500):
        self.semantic: Dict[str, Any] = {"intervention_patterns": []}
        self.episodic: List[Dict[str, Any]] = []
        self._cap = episodic_cap

    def write_events(self, events: List[Dict[str, Any]]) -> None:
        self.episodic.extend(events)
        if len(self.episodic) > self._cap:
            self.episodic = self.episodic[-self._cap:]

    def summarize_recent(self, horizon: int = 16) -> str:
        ev = self.episodic[-horizon:]
        notes: List[str] = []
        for e in ev:
            t = e.get("type")
            if t == "plan":
                steps = e.get("steps", [])
                action_head = (steps[0].split("(")[0] if steps else "?")
                notes.append(f"plan[{action_head}]")
            elif t == "human_intervention":
                corrected = e.get("corrected_action") or "pending"
                head = corrected.split("(")[0] if isinstance(corrected, str) else "pending"
                notes.append(f"intervention[{head}]")
            elif t == "obs":
                notes.append(f"obs[{e.get('summary', '?')[:32]}]")
        if not notes:
            return "No recent notable events."
        return "Recent: " + "; ".join(notes[:8])

    def prompt_view(self) -> Dict[str, Any]:
        return {
            "summary": self.summarize_recent(),
            "intervention_patterns": self.semantic.get("intervention_patterns", []),
        }


# ---------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------


class LLMClient:
    """OpenAI JSON-mode adapter (compatible with gpt-5-mini / gpt-4o)."""

    def __init__(self, openai_client: Optional[OpenAI] = None, model: str = "gpt-5-mini"):
        self.client = openai_client or OpenAI()
        self.model = model
        self.verbose = False

    def respond_json(
        self,
        schema: Dict[str, Any],
        system: str,
        user: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        def _call(system_prompt, user_payload):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )
            if not response or not response.choices:
                raise ValueError("Empty response from API (no choices).")
            return response.choices[0].message.content

        def _fallback(reason: str):
            return {
                "steps": ["WAIT_TEAMMATE_CONTINUE()"],
                "chain_of_thought": f"Safe fallback: {reason}",
                "category": "general_hint",
                "teammate_behavior": "",
                "scratchpad_update": "",
            }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                raw = _call(system, user)
                if not raw or not raw.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise ValueError("Empty response after retries")

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {e}. Raw: {raw!r}")

                allowed_properties = set(schema.get("properties", {}).keys())
                parsed = {k: v for k, v in parsed.items() if k in allowed_properties}

                category = parsed.get("category", "general_hint")
                if not category or category not in ["policy", "env", "teammate", "general_hint"]:
                    parsed["category"] = "general_hint"

                parsed.setdefault("steps", ["WAIT_TEAMMATE_CONTINUE()"])
                if "chain_of_thought" in allowed_properties:
                    parsed.setdefault("chain_of_thought", "No chain of thought provided")
                parsed.setdefault("teammate_behavior", "")
                parsed.setdefault("scratchpad_update", "")

                if isinstance(parsed.get("chain_of_thought"), str):
                    parsed["chain_of_thought"] = parsed["chain_of_thought"][:2048]
                if isinstance(parsed.get("teammate_behavior"), str):
                    parsed["teammate_behavior"] = parsed["teammate_behavior"][:220]
                if isinstance(parsed.get("scratchpad_update"), str):
                    parsed["scratchpad_update"] = parsed["scratchpad_update"][:1024]

                try:
                    validate(instance=parsed, schema=schema)
                except ValidationError as e:
                    try:
                        repair_system = system + "\nFix the JSON to match the schema exactly. Return ONLY JSON."
                        repair_user = dict(user)
                        repair_user["__validation_error"] = e.message
                        repair_user["__invalid_output"] = raw
                        repaired_raw = _call(repair_system, repair_user)
                        repaired = json.loads(repaired_raw)
                        repaired = {k: v for k, v in repaired.items() if k in allowed_properties}
                        validate(instance=repaired, schema=schema)
                        return repaired
                    except Exception:
                        return _fallback(f"schema validation failed: {e.message}")

                return parsed
            except ValueError as e:
                last_error = e
                if attempt < max_retries - 1 and "Empty response" in str(e):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return _fallback(str(e))
            except Exception as e:
                raise RuntimeError(f"LLMClient.respond_json failed: {e}")

        raise RuntimeError(
            f"LLMClient.respond_json failed after {max_retries} attempts: {last_error}"
        )


# ---------------------------------------------------------------------
# Prompt / schema construction
# ---------------------------------------------------------------------


PLAN_JSON_SCHEMA_BASE: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["steps", "category", "teammate_behavior"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {"type": "string", "maxLength": 8192},
        },
        "chain_of_thought": {"type": "string", "maxLength": 2048},
        "category": {
            "type": "string",
            "enum": ["policy", "env", "teammate", "general_hint"],
        },
        "teammate_behavior": {"type": "string", "maxLength": 220},
        "scratchpad_update": {"type": "string", "maxLength": 1024},
    },
}


def _get_plan_json_schema(enable_cot: bool = True) -> Dict[str, Any]:
    schema = copy.deepcopy(PLAN_JSON_SCHEMA_BASE)
    required = ["steps", "category", "teammate_behavior"]
    if enable_cot:
        required.append("chain_of_thought")
    schema["required"] = required
    return schema


def _load_system_rules() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rules_file = os.path.join(script_dir, "advanced_llm_system_rules_cogym.txt")
    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "You are a Collaborative Gym agent. Emit exactly one JSON object with keys "
            "steps (list of one action string), chain_of_thought, category, teammate_behavior, "
            "scratchpad_update. The action string must match one of the runtime action patterns."
        )


SYSTEM_RULES = _load_system_rules()


def _get_system_rules(enable_cot: bool = True, enable_memory: bool = True) -> str:
    result = SYSTEM_RULES
    if not enable_cot:
        result = re.sub(
            r'"chain_of_thought":\s*"[^"]*",?\s*//[^\n]*\n?', "", result
        )
        result += (
            "\n\nNOTE: chain_of_thought reasoning is DISABLED. Leave 'chain_of_thought' empty."
        )
    if not enable_memory:
        result += (
            "\n\nNOTE: Memory is DISABLED. Ignore any 'memory' or 'icl_examples' hints; "
            "reason only from the current state, history, and human_message."
        )
    return result


def action_space_to_description(action_space: List[Dict[str, Any]]) -> str:
    """Compact task-agnostic description of the runtime action space."""
    lines: List[str] = []
    for space in action_space:
        name = space.get("human_readable_name", space.get("machine_readable_identifier", "?"))
        params = space.get("params", [])
        desc = space.get("human_readable_description", "").strip()
        pattern = space.get("pattern", "")
        line = f"- {name} (params: {params})"
        if desc:
            line += f"\n    desc: {desc}"
        line += f"\n    regex: {pattern}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------


class AdvancedLLMInterpreter:
    def __init__(
        self,
        llm: LLMClient,
        memory: AgentMemory,
        history_horizon: int = 3,
        enable_cot: bool = True,
        enable_memory: bool = True,
        retrieval_strategy: RetrievalStrategy = "semantic",
        retrieval_seed: int = 0,
    ):
        self.llm = llm
        self.memory = memory
        self.history_horizon = history_horizon
        self.enable_cot = enable_cot
        self.enable_memory = enable_memory
        self.retrieval_strategy: RetrievalStrategy = (
            retrieval_strategy if retrieval_strategy in {"semantic", "random", "prepend_all"} else "semantic"
        )
        self._retrieval_rng = random.Random(retrieval_seed)
        self.verbose = False

        self._last_teammate_behavior: str = ""
        self._pending_patterns: Dict[int, Dict[str, Any]] = {}

        # Debug hooks
        self.last_use_retrieval: bool = False
        self.last_icl_query_text: str = ""
        self.last_icl_examples: List[Dict[str, Any]] = []
        self.last_icl_scores: List[Dict[str, Any]] = []

    # ------------------------- helpers -------------------------

    def _embed_text(self, text: str) -> Optional[List[float]]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            resp = self.llm.client.embeddings.create(
                model="text-embedding-3-small",
                input=[text],
            )
            if not resp or not resp.data:
                return None
            return resp.data[0].embedding
        except Exception:
            return None

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def _select_icl_examples(
        self,
        psi_text: str,
        events: Optional[List[str]] = None,
        k: int = 3,
        teammate_descriptor: str = "",
    ) -> List[Dict[str, Any]]:
        patterns = [
            p for p in self.memory.semantic.get("intervention_patterns", [])
            if p.get("outcome") == "success"
        ]
        self.last_icl_scores = []
        if not patterns:
            return []

        if self.retrieval_strategy == "prepend_all":
            selected = sorted(patterns, key=lambda p: p.get("timestamp", 0), reverse=True)
        elif self.retrieval_strategy == "random":
            n = min(k, len(patterns))
            selected = self._retrieval_rng.sample(patterns, n) if n else []
        else:  # semantic
            td = (teammate_descriptor or "").strip()
            query_text = psi_text.strip() if not td else (psi_text.strip() + f"\nTeammateDescriptor: {td}")
            query_emb = self._embed_text(query_text) if query_text else None

            if query_emb:
                def score(p):
                    emb = p.get("embedding_ctx")
                    if emb is None:
                        p_text = (p.get("psi_text") or "").strip()
                        p_tb = (p.get("teammate_behavior") or "").strip()
                        ctx = p_text if not p_tb else (p_text + f"\nTeammateBehavior: {p_tb}")
                        emb = self._embed_text(ctx) if ctx else None
                        p["embedding_ctx"] = emb
                    sim = self._cosine_sim(query_emb, emb) if emb else -1.0
                    return (sim, p.get("timestamp", 0))
                selected = sorted(patterns, key=score, reverse=True)[:k]
                self.last_icl_scores = [
                    {
                        "timestamp": p.get("timestamp"),
                        "score": float(self._cosine_sim(query_emb, p.get("embedding_ctx"))) if p.get("embedding_ctx") else -1.0,
                        "strategy": "semantic",
                    }
                    for p in selected
                ]
            elif events:
                def score(p):
                    pe = set(p.get("detected_failures") or [])
                    ce = set(events)
                    return (len(pe.intersection(ce)), p.get("timestamp", 0))
                selected = sorted(patterns, key=score, reverse=True)[:k]
            else:
                selected = patterns[-k:]

        examples = []
        for p in selected[:k] if self.retrieval_strategy != "prepend_all" else selected:
            examples.append({
                "timestamp": p.get("timestamp"),
                "human_message": p.get("human_message", ""),
                "detected_failures": p.get("detected_failures", []),
                "state_abstraction": p.get("psi_text", ""),
                "action_taken": p.get("action_taken", ""),
                "category": p.get("category", None),
                "teammate_behavior": p.get("teammate_behavior", ""),
                "outcome": p.get("outcome", "unknown"),
            })
        return examples

    # ------------------------- pattern lifecycle -------------------------

    def _store_intervention_pattern(
        self,
        plan: Plan,
        human_msg: HumanMessage,
        psi_text: str,
        events: Optional[List[str]] = None,
    ) -> None:
        if not self.enable_memory:
            return
        try:
            action_taken = plan.steps[0] if plan.steps else ""
            tb = (plan.teammate_behavior or "").strip()
            ctx_text = psi_text if not tb else (psi_text + f"\nTeammateBehavior: {tb}")
            pattern = {
                "timestamp": human_msg.t,
                "human_message": human_msg.text,
                "detected_failures": list(events or []),
                "action_taken": action_taken,
                "category": getattr(plan, "category", None),
                "teammate_behavior": tb,
                "psi_text": psi_text,
                "embedding_ctx": self._embed_text(ctx_text) if ctx_text else None,
                "outcome": "pending",
            }
            self._pending_patterns[human_msg.t] = pattern
        except Exception:
            return

    def commit_intervention_pattern(self, timestamp: int) -> None:
        pattern = self._pending_patterns.pop(timestamp, None)
        if not pattern:
            return
        pattern["outcome"] = "success"
        self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
        max_patterns = 10
        patterns = self.memory.semantic["intervention_patterns"]
        if len(patterns) > max_patterns:
            self.memory.semantic["intervention_patterns"] = patterns[-max_patterns:]

    def discard_intervention_pattern(self, timestamp: int) -> None:
        pattern = self._pending_patterns.pop(timestamp, None)
        if not pattern:
            return
        pattern["outcome"] = "failure"
        self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
        max_patterns = 10
        patterns = self.memory.semantic["intervention_patterns"]
        if len(patterns) > max_patterns:
            self.memory.semantic["intervention_patterns"] = patterns[-max_patterns:]

    # ------------------------- main entry -------------------------

    def propose_plan(
        self,
        psi_text: str,
        recent_history: List[str],
        human_msg: HumanMessage,
        action_space_desc: str,
        task_description: str = "",
        events: Optional[List[str]] = None,
        scratchpad_text: str = "",
    ) -> Plan:
        """Return a Plan whose steps[0] is a Co-Gym action string."""
        events = list(events or [])

        # Retrieval-time record: if human is speaking, remember it in episodic mem.
        if self.enable_memory and human_msg.text.strip():
            self.memory.write_events([{
                "t": human_msg.t,
                "type": "human_intervention",
                "text": human_msg.text,
                "events": events,
                "corrected_action": None,
            }])

        user_payload: Dict[str, Any] = {
            "task_description": task_description,
            "state": psi_text,
            "history": recent_history[-self.history_horizon:],
            "action_space_description": action_space_desc,
            "human_message": {"t": human_msg.t, "text": human_msg.text},
            "detected_failures": events,
            "previous_teammate_descriptor": self._last_teammate_behavior,
            "scratchpad": scratchpad_text,
        }

        if self.enable_memory:
            mem_view = self.memory.prompt_view()
            failure_events = {"lack_of_progress", "repeated_action", "stalled_teammate"}
            use_retrieval = (not human_msg.text.strip()) and any(e in failure_events for e in events)
            icl_examples = self._select_icl_examples(
                psi_text=psi_text,
                events=events,
                k=3,
                teammate_descriptor=self._last_teammate_behavior,
            ) if use_retrieval else []

            self.last_use_retrieval = bool(use_retrieval)
            self.last_icl_examples = list(icl_examples)
            self.last_icl_query_text = psi_text
            mem_view["intervention_patterns"] = []
            user_payload["memory"] = mem_view
            user_payload["icl_examples"] = icl_examples

        system_rules = _get_system_rules(
            enable_cot=self.enable_cot, enable_memory=self.enable_memory
        )
        schema = _get_plan_json_schema(enable_cot=self.enable_cot)

        try:
            raw = self.llm.respond_json(schema, system_rules, user_payload)
        except Exception:
            return Plan(
                steps=["WAIT_TEAMMATE_CONTINUE()"],
                chain_of_thought="Safe fallback due to LLM failure.",
                category="general_hint",
                teammate_behavior=self._last_teammate_behavior,
                scratchpad_update="",
            )

        steps = raw.get("steps", []) or []
        if not steps or not isinstance(steps[0], str):
            steps = ["WAIT_TEAMMATE_CONTINUE()"]

        category = raw.get("category", "general_hint")
        if category not in ["policy", "env", "teammate", "general_hint"]:
            category = "general_hint"

        chain_of_thought = str(raw.get("chain_of_thought", "")) if self.enable_cot else ""
        chain_of_thought = chain_of_thought[:2048]

        plan = Plan(
            steps=[steps[0]],
            chain_of_thought=chain_of_thought,
            category=category,
            teammate_behavior=str(raw.get("teammate_behavior", ""))[:220],
            scratchpad_update=str(raw.get("scratchpad_update", ""))[:1024],
        )
        if plan.teammate_behavior:
            self._last_teammate_behavior = plan.teammate_behavior

        # Log the plan / intervention outcome in episodic memory
        if self.enable_memory:
            if not human_msg.text.strip():
                self.memory.write_events([{
                    "t": human_msg.t,
                    "type": "plan",
                    "steps": plan.steps,
                }])
            else:
                # Update latest pending intervention with the corrected action
                try:
                    for i in range(len(self.memory.episodic) - 1, -1, -1):
                        ev = self.memory.episodic[i]
                        if ev.get("type") == "human_intervention" and ev.get("corrected_action") is None:
                            ev["corrected_action"] = plan.steps[0]
                            break
                except Exception:
                    pass
                self.memory.write_events([{
                    "t": human_msg.t,
                    "type": "plan",
                    "steps": plan.steps,
                    "category": plan.category,
                }])
                self._store_intervention_pattern(plan, human_msg, psi_text, events=events)

        return plan
