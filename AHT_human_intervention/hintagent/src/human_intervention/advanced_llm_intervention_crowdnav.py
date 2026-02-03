#!/usr/bin/env python3
import os
import json
import time
import re
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Literal, Tuple, Union
from openai import OpenAI

try:
	from jsonschema import validate, ValidationError
except ImportError:
	# Fallback if jsonschema is not available
	def validate(instance, schema):
		pass
	class ValidationError(Exception):
		pass  

# Allowed waypoint actions (0-7)
CROWDNAV_ALLOWED_WAYPOINT_ACTIONS = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]

Category = Literal["policy", "env", "teammate", "general_hint", "null"]

# ---------------------------------------------------------------------
# CoT + Memory Interpreter for Human Interventions in CrowdNav-AHT
# Inputs: text observation, human message, recent K-step history
# Outputs: structured Plan JSON, one-sentence rationale, message category
# ---------------------------------------------------------------------

@dataclass
class Plan:
	steps: List[int]  # List of waypoint action indices (0-7)
	chain_of_thought: str                # Explicit CoT reasoning with full reasoning process
	category: Category                    # model's classification of the human msg
	teammate_behavior: str = ""           # Teammate navigation pattern description
	low_level_override: Optional[Dict[str, float]] = None  # Direct velocity override (vx, vy)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,
			"chain_of_thought": self.chain_of_thought,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,
			"low_level_override": self.low_level_override,
		}

class AgentMemory:
	"""
	CrowdNav-specific persistent memory focused on intervention examples.
	- semantic: successful intervention patterns only
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 500):
		self.semantic: Dict[str, Any] = {
			# Learned intervention patterns to avoid future human corrections
			"intervention_patterns": [],
		}
		self.episodic: List[Dict[str, Any]] = []
		self._cap = episodic_cap

	def write_events(self, events: List[Dict[str, Any]]) -> None:
		self.episodic.extend(events)
		if len(self.episodic) > self._cap:
			self.episodic = self.episodic[-self._cap:]

	def summarize_recent(self, horizon: int = 16) -> str:
		"""
		Turn the last few events into a single neutral sentence for the prompt.
		Keep it short & factual (no hidden CoT here).
		"""
		ev = self.episodic[-horizon:]
		notes: List[str] = []
		for e in ev:
			if e.get("type") == "plan":
				notes.append(f"plan[{','.join(map(str, e.get('steps', [])))}]")
			elif e.get("type") == "result":
				notes.append(f"result[waypoint:{e.get('waypoint', '?')}:{'ok' if e.get('ok') else 'fail'}]")
			elif e.get("type") == "human_intervention":
				corrected_action = e.get('corrected_waypoint', 'pending')
				notes.append(f"intervention[{corrected_action}]")
			elif e.get("type") == "obs":
				notes.append(f"mate_obs[{e.get('mate','?')}]")
		if not notes:
			return "No recent notable events."
		return "Recent: " + "; ".join(notes[:8])

	def update_intervention_outcome(self, timestamp: int, outcome: str) -> None:
		"""Update the stored intervention pattern outcome by timestamp."""
		try:
			patterns = self.semantic.get("intervention_patterns", [])
			for i in range(len(patterns) - 1, -1, -1):
				if patterns[i].get("timestamp") == timestamp:
					patterns[i]["outcome"] = outcome
					return
		except Exception:
			return

	def prompt_view(self) -> Dict[str, Any]:
		"""Compact view for the model prompt."""
		sem = self.semantic
		return {
			"summary": self.summarize_recent(),
			"intervention_patterns": sem.get("intervention_patterns", []),
		}


@dataclass
class HumanMessage:
	t: int
	text: str


def _describe_situation_for_memory(state_prompt: str, events=None, psi_text: Optional[str] = None) -> str:
	"""Compact textual summary for memory retrieval."""
	base = (psi_text or state_prompt or "").strip()
	if len(base) > 400:
		base = base[:400] + "..."
	if events:
		return f"{base} | events: {', '.join(events)}"
	return base

def _extract_first_json_object(text: str) -> Tuple[dict | None, str | None]:
    """
    Fallback extractor in case model returns extra text.
    Handles truncated JSON by finding balanced braces and attempting repair.
    """
    import re
    if not text:
        return None, "Empty text."
    try:
        return json.loads(text), None
    except Exception:
        pass
    # Try fenced JSON
    m = re.search(r"```(?:json)?\s*({.*})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), None
        except Exception as e:
            return None, f"JSON decode error: {e}"
    # Fallback: find first {...} block accounting for string boundaries
    start, depth = None, 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                block = text[start:i+1]
                try:
                    return json.loads(block), None
                except json.JSONDecodeError:
                    pass
    return None, "No JSON object found."


class LLMClient:
    """
    Adapter around the OpenAI API using gpt-5-mini with enforced JSON mode.
    Falls back to heuristic extraction if response_format fails.
    """

    def __init__(self, openai_client: Optional[OpenAI] = None, model: str = "gpt-5-mini"):
        self.client = openai_client or OpenAI()
        self.model = model
        self.verbose = False

    def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request structured JSON output from GPT-5-mini.
        Returns parsed dict or raises ValueError with raw LLM text on failure.
        """
        def _call(system_prompt, user_payload):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=2048,
            )
            return response

        def _fallback(reason: str):
            return {
                "steps": [0],
                "chain_of_thought": f"Safe fallback: {reason}",
                "category": "general_hint",
                "teammate_behavior": "",
                "low_level_override": None,
            }

        try:
            response = _call(system, user)

            try:
                text_out = response.choices[0].message.content
            except Exception:
                text_out = getattr(response, "choices", [{}])[0].get("message", {}).get("content", None) or str(response)

            # Handle empty response (e.g., hit token limit)
            if not text_out or not text_out.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", None) if response.choices else None
                if finish_reason == "length":
                    if self.verbose:
                        print(f"[LLM-WARNING] Response hit token limit ({response.usage.completion_tokens if hasattr(response, 'usage') else 'unknown'} tokens), using fallback")
                else:
                    if self.verbose:
                        print(f"[LLM-WARNING] Empty response (finish_reason: {finish_reason}), using fallback")
                # Return fallback plan
                obj = {
                    "steps": [0],
                    "chain_of_thought": "Response was truncated or empty, using safe fallback",
                    "category": "general_hint",
                    "teammate_behavior": "",
                    "low_level_override": None,
                }
                # Skip to validation (will pass since we set all required fields)
                text_out = None  # Signal that we already have obj

            # Parse JSON if we haven't already created a fallback obj
            if text_out is not None:
                try:
                    obj = json.loads(text_out)
                except json.JSONDecodeError as e:
                    obj, err = _extract_first_json_object(text_out)
                    if obj is None:
                        if "Unterminated string" in str(e) or "No JSON object found" in str(err):
                            if self.verbose:
                                print(f"[LLM-WARNING] Response appears truncated, using fallback")
                                obj = {
                                    "steps": [0],
                                    "chain_of_thought": "Response was truncated, using safe fallback",
                                    "category": "general_hint",
                                    "teammate_behavior": "",
                                    "low_level_override": None,
                                }
                            else:
                                raise ValueError(f"Could not parse JSON: {err}\nRaw: {text_out}")
            
            
            # Remove any properties not in the schema
            # This is necessary because the schema has additionalProperties: False
            allowed_properties = set(schema.get("properties", {}).keys())
            obj = {k: v for k, v in obj.items() if k in allowed_properties}
            
            # Repair common issues before validation
            # Ensure required fields exist FIRST (before validation)
            obj.setdefault("steps", [0])
            if not isinstance(obj.get("steps"), list) or len(obj["steps"]) == 0:
                obj["steps"] = [0]
            elif len(obj["steps"]) > 1:
                # Keep strict maxItems=1 by truncating extra waypoints
                obj["steps"] = obj["steps"][:1]
            
            # Fix category field - ensure it's always a valid enum value
            category = obj.get("category", "general_hint")
            if not category or category.strip() == "" or category not in ["policy", "env", "teammate", "general_hint", "null"]:
                obj["category"] = "general_hint"
            if "chain_of_thought" in allowed_properties:
                obj.setdefault("chain_of_thought", "No chain of thought provided")
            obj.setdefault("teammate_behavior", "")
            if "low_level_override" in allowed_properties:
                obj.setdefault("low_level_override", None)
            
            # Truncate fields that have maxLength constraints before validation
            if "chain_of_thought" in obj and isinstance(obj["chain_of_thought"], str):
                obj["chain_of_thought"] = obj["chain_of_thought"][:512]  # Match schema maxLength
            if "teammate_behavior" in obj and isinstance(obj["teammate_behavior"], str):
                obj["teammate_behavior"] = obj["teammate_behavior"][:220]  # Match schema maxLength
            
            # Validate against schema (one repair attempt)
            try:
                validate(instance=obj, schema=schema)
            except ValidationError as e:
                try:
                    repair_system = system + "\nFix the JSON to match the schema exactly. Return ONLY JSON."
                    repair_user = dict(user)
                    repair_user["__validation_error"] = e.message
                    repair_user["__invalid_output"] = obj
                    repaired_resp = _call(repair_system, repair_user)
                    repaired_text = repaired_resp.choices[0].message.content
                    repaired = json.loads(repaired_text)
                    repaired = {k: v for k, v in repaired.items() if k in allowed_properties}
                    validate(instance=repaired, schema=schema)
                    return repaired
                except Exception:
                    return _fallback(f"schema validation failed: {e.message}")
            
            return obj

        except Exception as e:
            return _fallback(f"LLMClient.respond_json failed: {e}")


# JSON Schema for Plan output
PLAN_JSON_SCHEMA = {
	"$schema": "http://json-schema.org/draft-07/schema#",
	"type": "object",
	"additionalProperties": False,
	"required": ["steps", "chain_of_thought", "category", "teammate_behavior"],
	"properties": {
		"steps": {
			"type": "array",
			"items": {"type": "integer", "minimum": 0, "maximum": 7},
			"minItems": 1,
		"maxItems": 1,
		},
		"chain_of_thought": {"type": "string", "maxLength": 512}, 
		"category": {"type": "string", "enum": ["policy", "env", "teammate", "general_hint", "null"]},
		"teammate_behavior": {"type": "string", "maxLength": 220},
		"low_level_override": {
			"oneOf": [
				{
					"type": "object",
					"additionalProperties": False,
					"required": ["vx", "vy"],
					"properties": {
						"vx": {"type": "number"},
						"vy": {"type": "number"}
					}
				},
				{"type": "null"}
			],
			"default": None
		},
	},
}


def _get_system_rules(enable_cot: bool = True, enable_memory: bool = True) -> str:
	"""Return system rules with optional CoT/memory sections removed."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	rules_file = os.path.join(script_dir, "advanced_llm_system_rules_crowdnav.txt")
	try:
		with open(rules_file, 'r', encoding='utf-8') as f:
			base_rules = f.read()
	except Exception as e:
		# Keep stdout quiet
		base_rules = "You are a navigation assistant for a robot in a crowd."

	if enable_cot and enable_memory:
		return base_rules

	result = base_rules

	if not enable_memory:
		# Remove memory line from INPUT CONTEXT
		result = re.sub(r'-\s*memory:.*\n', '', result, flags=re.IGNORECASE)
		# Remove Memory Use step from CHAIN OF THOUGHT
		result = re.sub(r'5\.\s*Memory Use:.*\n', '', result, flags=re.IGNORECASE)

	if not enable_cot:
		# Remove CHAIN OF THOUGHT block
		result = re.sub(r'CHAIN OF THOUGHT.*?\n\n', '', result, flags=re.DOTALL | re.IGNORECASE)
		# Set chain_of_thought field to empty in output description
		result = re.sub(r'"chain_of_thought":\s*".*?"', '"chain_of_thought": ""', result)

	return result


class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter for human interventions in CrowdNav-AHT.
	Uses LLM to generate structured plans from text observations and human messages.
	"""
	
	def __init__(self, llm_client: LLMClient, memory: AgentMemory, history_horizon: int = 2,
	             enable_cot: bool = True, enable_memory: bool = True):
		self.llm = llm_client
		self.memory = memory
		self.history_horizon = history_horizon
		self.enable_cot = enable_cot
		self.enable_memory = enable_memory
		self.verbose = False
		self._last_teammate_behavior = ""
		self._pending_patterns: Dict[int, Dict[str, Any]] = {}
		self.system_rules = _get_system_rules(enable_cot=enable_cot, enable_memory=enable_memory)

	def _embed_text(self, text: str) -> Optional[List[float]]:
		"""Embed text using a sentence encoder (OpenAI embeddings)."""
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
		na = np.linalg.norm(a)
		nb = np.linalg.norm(b)
		if na == 0 or nb == 0:
			return -1.0
		return float(np.dot(a, b) / (na * nb))

	def _select_icl_examples(self, state_prompt: str, events=None, k: int = 3, psi_text: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Select a small set of ICL examples from intervention_patterns."""
		patterns = [
			p for p in self.memory.semantic.get("intervention_patterns", [])
			if p.get("outcome") == "success"
		]
		if not patterns:
			return []
		query_text = psi_text or _describe_situation_for_memory(state_prompt, events, psi_text=psi_text)
		query_emb = self._embed_text(query_text) if query_text else None
		
		if query_emb:
			def score(p):
				emb = p.get("embedding")
				if emb is None:
					p_text = p.get("psi_text") or ""
					emb = self._embed_text(p_text) if p_text else None
				sim = self._cosine_sim(query_emb, emb) if emb else -1.0
				return (sim, p.get("timestamp", 0))
			sorted_p = sorted(patterns, key=score, reverse=True)
		elif events:
			def score(p):
				pe = set(p.get("detected_failures") or [])
				ce = set(events)
				overlap = len(pe.intersection(ce))
				return (overlap, p.get("timestamp", 0))
			sorted_p = sorted(patterns, key=score, reverse=True)
		else:
			sorted_p = patterns[-k:]
		
		selected = sorted_p[:k]
		examples = []
		for p in selected:
			examples.append({
				"timestamp": p.get("timestamp"),
				"detected_failures": p.get("detected_failures", []),
				"state_abstraction": p.get("psi_text", ""),
				"outcome": p.get("outcome", "unknown"),
				"teammate_behavior": p.get("teammate_behavior", ""),
				"category": p.get("category", None),
				"skill": p.get("skill"),
				"low_level_override": p.get("low_level_override"),
			})
		return examples

	def _store_intervention_pattern(self, plan: Plan, human_message: str, state_prompt: str, events=None, t: int = 0, psi_text: Optional[str] = None) -> None:
		"""Queue intervention pattern; commit only on success."""
		try:
			if not human_message.strip():
				return
			psi_snapshot = psi_text or ""
			pattern = {
				"timestamp": t,
				"human_message": human_message,
				"detected_failures": events or [],
				"skill": plan.steps[0] if plan.steps else None,
				"low_level_override": plan.low_level_override,
				"category": getattr(plan, "category", None),
				"teammate_behavior": getattr(plan, "teammate_behavior", ""),
				"psi_text": psi_snapshot,
				"embedding": self._embed_text(psi_snapshot) if psi_snapshot else None,
				"outcome": "pending",
			}
			self._pending_patterns[t] = pattern
		except Exception:
			return

	def commit_intervention_pattern(self, timestamp: int) -> None:
		pattern = self._pending_patterns.pop(timestamp, None)
		if not pattern:
			return
		pattern["outcome"] = "success"
		self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
		if len(self.memory.semantic["intervention_patterns"]) > 10:
			self.memory.semantic["intervention_patterns"] = self.memory.semantic["intervention_patterns"][-10:]

	def discard_intervention_pattern(self, timestamp: int) -> None:
		pattern = self._pending_patterns.pop(timestamp, None)
		if not pattern:
			return
		pattern["outcome"] = "failure"
		self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
		if len(self.memory.semantic["intervention_patterns"]) > 10:
			self.memory.semantic["intervention_patterns"] = self.memory.semantic["intervention_patterns"][-10:]

	def interpret(
		self,
		state_prompt: str,
		recent_history: List[Dict[str, Any]],
		human_message: Optional[str] = None,
		events: Optional[List[str]] = None,
		timestep: Optional[int] = None,
		psi_text: Optional[str] = None,
	) -> Plan:
		"""
		Generate a plan from current state, history, and optional human intervention.
		
		Args:
			state_prompt: Text description of current environment state
			recent_history: List of recent waypoint actions and outcomes
			human_message: Optional human intervention text
		
		Returns:
			Plan object with waypoint actions and reasoning
		"""
		if self.enable_memory:
			memory_view = self.memory.prompt_view()
			failure_events = {"lack_of_progress", "cyclic_behavior"}
			use_retrieval = (not human_message or not human_message.strip()) and events and any(
				e in failure_events for e in events
			)
			icl_examples = self._select_icl_examples(state_prompt, events=events, k=3, psi_text=psi_text) if use_retrieval else []
			# Do not duplicate ICL examples inside memory_view
			memory_view["intervention_patterns"] = []
		else:
			memory_view = {}
			icl_examples = []
		
		# Build user prompt
		user_prompt = {
			"state": state_prompt,
			"history": recent_history[-self.history_horizon:],
			"memory": memory_view,
			"detected_failures": events or [],
			"icl_examples": icl_examples,
			"previous_teammate_descriptor": self._last_teammate_behavior,
		}
		
		if human_message:
			user_prompt["human_message"] = human_message
		
		try:
			raw_plan = self.llm.respond_json(PLAN_JSON_SCHEMA, self.system_rules, user_prompt)
			try:
				if self.verbose:
					print(f"[LLM] Raw plan: {json.dumps(raw_plan, ensure_ascii=True)}")
					if self.enable_cot:
						print(f"[COT] {raw_plan.get('chain_of_thought', '')}")
					low_level_override = raw_plan.get("low_level_override")
					if low_level_override is not None:
						print(f"[LLM] Low-level override: {low_level_override}")
			except Exception:
				pass
		except Exception:
			return Plan(
				steps=[0],
				chain_of_thought="Safe fallback due to verification failure",
				category="general_hint",
				teammate_behavior="",
				low_level_override=None,
			)
		# Record in episodic memory
		if self.enable_memory:
			self.memory.write_events([{
				"type": "plan",
				"t": len(self.memory.episodic),
				"steps": raw_plan.get("steps", []),
				"category": raw_plan.get("category", ""),
				"human_message": human_message,
			}])
		
		# Convert to Plan dataclass
		plan = Plan(
			steps=raw_plan.get("steps", [0]),
			chain_of_thought=raw_plan.get("chain_of_thought", "") if self.enable_cot else "",
			category=raw_plan.get("category", "general_hint"),
			teammate_behavior=raw_plan.get("teammate_behavior", ""),
			low_level_override=raw_plan.get("low_level_override"),
		)
		if plan.teammate_behavior:
			self._last_teammate_behavior = plan.teammate_behavior
		
		# Store intervention pattern for retrieval (if human intervened)
		if self.enable_memory and human_message and human_message.strip():
			self._store_intervention_pattern(plan, human_message, psi_text or state_prompt, events=events, t=int(timestep or 0), psi_text=psi_text)
		
		return plan
