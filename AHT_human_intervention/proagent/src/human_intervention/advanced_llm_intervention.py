#!/usr/bin/env python3
import os
import json
import time
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Literal, Tuple
from openai import OpenAI  

PROAGENT_ALLOWED_ML_ACTIONS = {
	"pickup_onion",
	"pickup_dish", 
	"put_onion_in_pot",
	"fill_dish_with_soup",
	"deliver_soup",
	"place_obj_on_counter"
}

# All valid wait actions
WAIT_ACTIONS = [f"wait({i})" for i in range(1, 21)]

# Combined allowed actions for schema validation
ALL_VALID_ML_ACTIONS = list(PROAGENT_ALLOWED_ML_ACTIONS) + WAIT_ACTIONS

Category = Literal["policy", "env", "teammate", "vague"]

# ---------------------------------------------------------------------
# CoT + Memory Interpreter for Human Interventions in Overcooked-AI
# Inputs: observable state, human message, recent K-step history
# Outputs: structured Plan JSON, one-sentence rationale, message category
# ---------------------------------------------------------------------

@dataclass
class Plan:
	steps: List[str]  # List of ML_action strings instead of MacroStep objects
	confidence: float
	rationale_public: str                 # <= 1 sentence, no CoT
	category: Category                    # model's classification of the human msg
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,  # Direct list of ML_action strings
			"confidence": self.confidence,
			"rationale_public": self.rationale_public,
			"category": self.category,
			"memory_writes": self.memory_writes,
		}

class AgentMemory:
	"""
	Overcooked-specific persistent memory with structured world state tracking.
	- semantic: long-lived facts, rules, and dynamic world state
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 120, mdp=None):
		self.semantic: Dict[str, Any] = {
			# Static layout & traffic
			"layout_facts": {
				"pots": [],               # [{"pos": [x,y]}, ...]
				"counters": [],           # [{"pos": [x,y]}]
				"onion_sources": [],      # [{"pos": [x,y]}]
				"dish_sources": [],       # [{"pos": [x,y]}]
				"serving_windows": [],    # [{"pos": [x,y]}]
				"blocked_cells": [],      # [(x,y), ...] (updated by env messages)
			},
			# Dynamic, frequently updated world state (LLM/env messages can correct this)
			"world_state": {
				"pot_status": [],         # [{"pos":[x,y],"contents":n_onions,"ready":bool,"burning":bool}]
				"onion_caches": [],       # [{"pos":[x,y],"count":k,"t":ts}]
				"dish_availability": 0,   # rough count if tracked
			},
			# Teammate hypothesis & coordination contract
			"teammate_model": {
				"current_role": "unknown",
				"role_confidence": 0.0,
				"since_t": None,
				"behavior_description": "",
				"last_updated": None,
				# Optional: reliability score influences how much we defer
				"reliability": 0.5
			},
			# Human/operator preferences or "house rules"
			"human_prefs": {
				# e.g., "prefer_left_pot": True, "avoid_crossing_center": True
			},
			# Compact "if…then…" contracts derived from interventions or LLM
			"playbook": [
				# {"if": "mate_role==server and pot_ready", "then": ["pickup_onion", "..."], "note":"..."}
			],
			# Safety and throttles
			"afford_safety": {"max_wait_on_pass": 2},
			# Choke points or priority zones useful for pathing/avoidance
			"hotspots": [],
			# Things to clarify when vague messages show up
			"open_questions": [],
		}
		self.episodic: List[Dict[str, Any]] = []
		self._cap = episodic_cap
		
		# Initialize layout facts if MDP is provided
		if mdp is not None:
			self.initialize_layout_facts(mdp)

	def initialize_layout_facts(self, mdp) -> None:
		"""
		Initialize layout_facts from the MDP using the same logic as generate_layout_prompt.
		This populates the memory with structured layout information.
		"""
		layout_facts = self.semantic["layout_facts"]
		
		# Map MDP method names to our layout_facts keys
		layout_mapping = {
			"onion_dispenser": "onion_sources",
			"dish_dispenser": "dish_sources", 
			"serving": "serving_windows",
			"pot": "pots",
		}
		
		# Extract layout information from MDP
		for obj_type, facts_key in layout_mapping.items():
			try:
				locations = getattr(mdp, f"get_{obj_type}_locations")()
				layout_facts[facts_key] = [{"pos": list(pos), "id": idx} for idx, pos in enumerate(locations)]
			except AttributeError:
				# Some layouts might not have all object types
				layout_facts[facts_key] = []
		
		# Initialize counters (typically all non-blocked positions)
		try:
			# Get the terrain map to identify counter positions
			terrain = mdp.terrain_mtx
			counters = []
			for y in range(terrain.shape[0]):
				for x in range(terrain.shape[1]):
					# Counter positions are typically ' ' (space) in the terrain
					if terrain[y, x] == ' ':
						counters.append({"pos": [x, y]})
			layout_facts["counters"] = counters
		except AttributeError:
			layout_facts["counters"] = []

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
				notes.append(f"plan[{','.join(e.get('steps', []))}]")
			elif e.get("type") == "result":
				notes.append(f"result[{e.get('macro')}:{'ok' if e.get('ok') else 'fail'}]")
			elif e.get("type") == "human_msg":
				notes.append(f"human[{e.get('kind')}]")
			elif e.get("type") == "obs":
				notes.append(f"mate_obs[{e.get('mate','?')}]")
		if not notes:
			return "No recent notable events."
		return "Recent: " + "; ".join(notes[:8])

	def debug_print_memory(self) -> None:
		"""Print detailed memory contents for debugging."""
		print("=" * 50)
		print("MEMORY DEBUG - FULL CONTENTS")
		print("=" * 50)
		
		print(f"\nEPISODIC MEMORY ({len(self.episodic)} entries):")
		for i, entry in enumerate(self.episodic[-10:]):  # Last 10 entries
			print(f"  {i}: {entry}")
		
		print(f"\nSEMANTIC MEMORY:")
		for key, value in self.semantic.items():
			print(f"  {key}: {value}")
		
		print("=" * 50)

	def upsert_semantic(self, patch: Dict[str, Any]) -> None:
		"""
		Conservative merge of small dict patches from the model (after gating).
		Only whitelisted top-level keys are merged.
		"""
		WHITELIST = {"layout_facts", "world_state", "teammate_model", "human_prefs",
					 "playbook", "afford_safety", "hotspots", "open_questions"}
		print(f"[MEMORY] upsert_semantic called with keys: {list(patch.keys())}")
		for k, v in patch.items():
			if k not in WHITELIST:
				print(f"[MEMORY] Skipping non-whitelisted key: {k}")
				continue
			print(f"[MEMORY] Applying patch for key: {k}")
			if isinstance(v, list):
				base = self.semantic.get(k, [])
				base.extend(v)
				if k == "playbook":
					seen = set()
					dedup = []
					for item in base:
						sig = item.get("if", json.dumps(item, sort_keys=True))
						if sig in seen:
							continue
						seen.add(sig)
						dedup.append(item)
					base = dedup[-16:]
				self.semantic[k] = base
			elif isinstance(v, dict) and isinstance(self.semantic.get(k, {}), dict):
				# Special handling for teammate_model to preserve behavior_description
				if k == "teammate_model" and "behavior_description" in v:
					# Update behavior description with timestamp
					v["last_updated"] = time.time()
				# Deep merge for nested dicts like layout_facts and world_state
				self._deep_merge(self.semantic[k], v)
			else:
				self.semantic[k] = v

	def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
		"""Deep merge update_dict into base_dict."""
		for key, value in update_dict.items():
			if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
				self._deep_merge(base_dict[key], value)
			else:
				base_dict[key] = value

	def prompt_view(self) -> Dict[str, Any]:
		"""Compact view for the model prompt."""
		sem = self.semantic
		tm = sem["teammate_model"]
		ws = sem.get("world_state", {})
		
		# Debug: Print memory contents
		print(f"[MEMORY-DEBUG] Episodic memory: {len(self.episodic)} entries")
		if self.episodic:
			print(f"[MEMORY-DEBUG] Last 3 episodic entries: {self.episodic[-3:]}")
		print(f"[MEMORY-DEBUG] Semantic memory keys: {list(sem.keys())}")
		print(f"[MEMORY-DEBUG] Teammate model: {tm}")
		print(f"[MEMORY-DEBUG] World state: {ws}")

		# Keep short: last 1-2 key pot statuses, last 1-2 onion caches
		key_pots = ws.get("pot_status", [])[-2:]
		onion_caches = ws.get("onion_caches", [])[-2:]

		return {
			"layout_facts": {
				"pots": sem["layout_facts"].get("pots", []),
				"serving_windows": sem["layout_facts"].get("serving_windows", []),
			},
			"world_state": {
				"pot_status": key_pots,
				"onion_caches": onion_caches,
				"dish_availability": ws.get("dish_availability", 0),
			},
			"teammate_model": {
				"role": tm.get("current_role"),
				"role_confidence": tm.get("role_confidence"),
				"behavior_description": tm.get("behavior_description"),
				"reliability": tm.get("reliability"),
				"last_updated": tm.get("last_updated"),
			},
			"human_prefs": sem.get("human_prefs", {}),
			"playbook": sem.get("playbook", [])[-4:],  # small slice
			"afford_safety": sem.get("afford_safety", {}),
			"hotspots": sem.get("hotspots", [])[-2:],
			"summary": self.summarize_recent(),
		}

@dataclass
class HumanMessage:
	t: int
	text: str

def _extract_first_json_object(text: str) -> Tuple[dict | None, str | None]:
    """
    Fallback extractor in case model returns extra text.
    """
    import re
    if not text:
        return None, "Empty text."
    try:
        return json.loads(text), None
    except Exception:
        pass
    # Try fenced JSON
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), None
        except Exception as e:
            return None, f"JSON decode error: {e}"
    # Fallback: find first {...} block
    start, depth = None, 0
    for i, ch in enumerate(text):
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
                except Exception as e:
                    return None, f"JSON decode error: {e}"
    return None, "No JSON object found."


class LLMClient:
    """
    Adapter around the OpenAI API using gpt-4.1-mini with enforced JSON mode.
    Falls back to heuristic extraction if response_format fails.
    """

    def __init__(self, openai_client: Optional[OpenAI] = None, model: str = "gpt-4.1-mini"):
        self.client = openai_client or OpenAI()
        self.model = model

    def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request structured JSON output from GPT-4.1-mini.
        Returns parsed dict or raises ValueError with raw LLM text on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512,
            )

            # Standard chat completions API
            try:
                text_out = response.choices[0].message.content
            except Exception:
                # Fallback for SDK variation
                text_out = getattr(response, "choices", [{}])[0].get("message", {}).get("content", None) or str(response)

            if not text_out:
                raise ValueError(f"No text content in response: {response}")

            # Attempt to parse cleanly
            try:
                return json.loads(text_out)
            except json.JSONDecodeError:
                # Extract first JSON block if model returned explanation text
                obj, err = _extract_first_json_object(text_out)
                if obj is None:
                    raise ValueError(f"Could not parse JSON: {err}\nRaw: {text_out}")
                return obj

        except Exception as e:
            raise RuntimeError(f"LLMClient.respond_json failed: {e}")

PLAN_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["steps", "confidence", "rationale_public", "category", "teammate_behavior"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "enum": ALL_VALID_ML_ACTIONS
            }
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale_public": {"type": "string", "maxLength": 180},
        "category": {"type": "string", "enum": ["policy", "env", "teammate", "vague"]},
        "teammate_behavior": {"type": "string", "maxLength": 220},
        "memory_writes": {
            "type": "array",
            "items": {"type": "object"},
            "default": []
        }
    }
}

def _load_system_rules() -> str:
	"""Load system rules from external file."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	rules_file = os.path.join(script_dir, "advanced_llm_system_rules.txt")
	try:
		with open(rules_file, 'r', encoding='utf-8') as f:
			return f.read()
	except FileNotFoundError:
		print(f"[WARNING] System rules file not found at {rules_file}, using fallback")
		return (
			"You control Player0 in Overcooked-AI. Goal: deliver soups quickly and safely.\n"
			"Use ONLY these ML actions: " + ", ".join(ALL_VALID_ML_ACTIONS) + "\n"
			"Return JSON with steps, confidence, rationale_public, category, and memory_writes."
		)

SYSTEM_RULES = _load_system_rules()

class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter.
	- compose the prompt from: state snapshot, short history, memory.view, human msg (+ optional heuristic category)
	- call LLM in JSON mode
	- return Plan + apply memory_writes (gated) to AgentMemory
	"""
	def __init__(self, llm: LLMClient, memory: AgentMemory, history_horizon: int = 8):
		self.llm = llm
		self.memory = memory
		self.history_horizon = history_horizon

	def propose_plan(
		self,
		state_prompt: str,
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
	) -> Plan:
		user_payload = {
			"state_prompt": state_prompt,
			"recent_history": recent_history[-self.history_horizon:],
			"memory_view": self.memory.prompt_view(),
			"human_message": {
				"t": human_msg.t,
				"text": human_msg.text,
			}
		}

		raw = self.llm.respond_json(PLAN_JSON_SCHEMA, SYSTEM_RULES, user_payload)
		
		# Debug: Check if LLM returned memory_writes
		if raw.get("memory_writes"):
			print(f"[MEMORY] LLM returned {len(raw['memory_writes'])} memory writes: {raw['memory_writes']}")
		else:
			print(f"[MEMORY] LLM returned no memory_writes")

		# Validate ML actions are in allowed set
		steps = raw.get("steps", [])
		validated_steps = []
		for step in steps:
			if step in ALL_VALID_ML_ACTIONS:
				validated_steps.append(step)
			else:
				print(f"[WARNING] Invalid ML action '{step}', skipping")
		
		# Determine category: empty string if no human intervention, otherwise use LLM response or default
		category = ""
		if human_msg.text.strip():  # If there's actual human intervention text
			category = raw.get("category", "vague")
		
		plan = Plan(
			steps=validated_steps,
			confidence=float(raw.get("confidence", 0.5)),
			rationale_public=str(raw.get("rationale_public", ""))[:180],
			category=category,
			memory_writes=raw.get("memory_writes", []),
		)

		# Plan hygiene checks
		# Hard cap actions to <= 3 and ensure at least one safe step
		if len(plan.steps) > 3:
			plan.steps = plan.steps[:3]
		if not plan.steps:
			plan.steps = ["wait(1)"]

		# Clamp confidence to [0,1]
		plan.confidence = max(0.0, min(1.0, plan.confidence))

		# NEW: ingest LLM-authored teammate behavior (if present)
		tb = (raw.get("teammate_behavior") or "").strip()
		if tb:
			plan.memory_writes.append({
				"teammate_model": {
					"behavior_description": tb,
					"last_updated": time.time()
				}
			})
		
		safe_patch = self._gate_memory_writes(plan.memory_writes, plan.category)
		if plan.memory_writes:
			print(f"[MEMORY] LLM generated {len(plan.memory_writes)} memory writes")
		if safe_patch:
			print(f"[MEMORY] Applying safe patch: {list(safe_patch.keys())}")
			self.memory.upsert_semantic(safe_patch)
		else:
			print(f"[MEMORY] No safe patch generated from {len(plan.memory_writes)} writes")
		self.memory.write_events([
			{"t": human_msg.t, "type": "human_msg", "kind": plan.category, "text": human_msg.text},
			{"t": human_msg.t, "type": "plan", "steps": plan.steps, "conf": plan.confidence}
		])

		return plan

	def _gate_memory_writes(self, writes: List[Dict[str, Any]], category: str = "vague") -> Dict[str, Any]:
		"""
		Convert a list of 'memory_writes' entries into a single semantic patch.
		Keep it conservative: allow updates only in whitelisted regions and small payloads.
		Category-based gating for different intervention types.
		Expected patterns (examples):
		  {"teammate_model": {"current_role":"plate_runner","role_confidence":0.7,"since_t":123}}
		  {"human_prefs": {"prefer_left_pot": true}}
		  {"world_state": {"pot_status": [{"pos":[1,2],"ready":true}]}}
		  {"playbook": [{"if":"...","then":[...],"note":"..."}]}
		"""
		patch: Dict[str, Any] = {}
		if not writes:
			print(f"[MEMORY] No writes to gate")
			return patch
		print(f"[MEMORY] Gating {len(writes)} writes with category: {category}")
			
		# Category-based whitelist - env messages can update world_state more freely
		ALLOWED_KEYS = {"teammate_model", "human_prefs", "playbook", "afford_safety", "hotspots", "open_questions", "layout_facts"}
		if category == "env":
			ALLOWED_KEYS.add("world_state")
		
		for i, w in enumerate(writes):
			print(f"[MEMORY] Processing write {i}: {list(w.keys())}")
			for k, v in w.items():
				if k not in ALLOWED_KEYS:
					print(f"[MEMORY] Rejecting key '{k}' (not in ALLOWED_KEYS)")
					continue
				if isinstance(v, (dict, list)) and len(json.dumps(v)) > 2000:
					print(f"[MEMORY] Rejecting key '{k}' (payload too large)")
					continue
				print(f"[MEMORY] Accepting key '{k}'")
				if k not in patch:
					patch[k] = v
				else:
					if isinstance(v, list):
						base = patch[k] if isinstance(patch[k], list) else []
						patch[k] = base + v
					elif isinstance(v, dict):
						base = patch[k] if isinstance(patch[k], dict) else {}
						base.update(v)
						patch[k] = base
		print(f"[MEMORY] Final patch keys: {list(patch.keys())}")
		return patch
