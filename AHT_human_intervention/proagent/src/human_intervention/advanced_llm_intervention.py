#!/usr/bin/env python3
import os
import json
import time
import re
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Literal, Tuple
from openai import OpenAI

try:
	from jsonschema import validate, ValidationError
except ImportError:
	# Fallback if jsonschema is not available
	def validate(instance, schema):
		pass
	class ValidationError(Exception):
		pass  

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
	teammate_behavior: str = ""           # Add this field
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)
	low_level_override: Optional[str] = None  # Direct low-level action override (e.g., "move_north", "wait", "interact")
	intervention_reason: str = ""         # LLM's inference of why human provided this intervention
	chain_of_thought: str = ""           # Explicit CoT reasoning (≤6 lines)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,  # Direct list of ML_action strings
			"confidence": self.confidence,
			"rationale_public": self.rationale_public,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,  # Include it
			"memory_writes": self.memory_writes,
			"low_level_override": self.low_level_override,
			"intervention_reason": self.intervention_reason,
			"chain_of_thought": self.chain_of_thought,
		}

class AgentMemory:
	"""
	Overcooked-specific persistent memory with structured world state tracking.
	- semantic: long-lived facts, rules, and dynamic world state
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 500, mdp=None):
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
				"since_t": None,
				"behavior_description": "",
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
			# Learned intervention patterns to avoid future human corrections
			"intervention_patterns": [],
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
		
		# Initialize counters (typically 'X' positions in Overcooked-AI)
		try:
			# Get the terrain map to identify counter positions
			terrain = mdp.terrain_mtx
			counters = []
			for y in range(terrain.shape[0]):
				for x in range(terrain.shape[1]):
					# Counter positions are typically 'X' in Overcooked-AI layouts
					if terrain[y, x] == 'X':
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
			elif e.get("type") == "human_intervention":
				corrected_action = e.get('corrected_ml_action', 'pending')
				notes.append(f"intervention[{corrected_action}]")
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
					# Update behavior description with since_t timestamp
					pass  # since_t is already set in the write
				# Deep merge for nested dicts like layout_facts and world_state
				self._deep_merge(self.semantic[k], v)
			else:
				self.semantic[k] = v

	def _merge_list_by_pos(self, dest: List[Dict[str, Any]], src: List[Dict[str, Any]], key="pos", cap=32):
		"""Merge lists by position to avoid duplicates."""
		idx = {tuple(it.get(key, [])): i for i, it in enumerate(dest) if key in it}
		for it in src:
			p = tuple(it.get(key, []))
			if p in idx:
				dest[idx[p]].update(it)
			else:
				dest.append(it)
		if len(dest) > cap:
			del dest[:-cap]

	def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
		"""Deep merge update_dict into base_dict."""
		for key, value in update_dict.items():
			if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
				self._deep_merge(base_dict[key], value)
			elif key in {"pot_status", "onion_caches"} and isinstance(value, list):
				base = base_dict.get(key, [])
				self._merge_list_by_pos(base, value)
				base_dict[key] = base
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
				"since_t": tm.get("since_t"),
				"behavior_description": tm.get("behavior_description"),
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

# Removed hand-coded parsing function - LLM will detect low-level commands through reasoning

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
                obj = json.loads(text_out)
            except json.JSONDecodeError:
                # Extract first JSON block if model returned explanation text
                obj, err = _extract_first_json_object(text_out)
                if obj is None:
                    raise ValueError(f"Could not parse JSON: {err}\nRaw: {text_out}")
            
            # Debug: log the raw LLM response
            print(f"[LLM-DEBUG] Raw LLM response: {obj}")
            
            # Pre-validate repair: if any low-level token leaked into steps, move it to low_level_override
            try:
                low_level_enums = [v for v in PLAN_JSON_SCHEMA["properties"]["low_level_override"]["enum"] if v is not None]
                steps_list = obj.get("steps", []) if isinstance(obj.get("steps", []), list) else []
                leaked = [s for s in steps_list if isinstance(s, str) and s in low_level_enums]
                if leaked:
                    # Prefer explicit low_level_override from the model; otherwise use the first leaked token
                    if not obj.get("low_level_override"):
                        obj["low_level_override"] = leaked[0]
                    # Replace steps with a valid placeholder ML action to satisfy schema
                    obj["steps"] = ["wait(1)"]
                    # Strengthen confidence for direct overrides if not provided
                    obj.setdefault("confidence", 1.0)
                    obj.setdefault("rationale_public", f"Direct low-level command: {obj['low_level_override']}")
            except Exception:
                pass
            
            # Validate against schema and repair if needed
            try:
                validate(instance=obj, schema=schema)
            except ValidationError as ve:
                print(f"[LLM-DEBUG] Validation error: {ve}")
                print(f"[LLM-DEBUG] Problematic object: {obj}")
                # Minimal repair for required fields
                obj.setdefault("steps", ["wait(1)"])
                obj.setdefault("confidence", 0.5)
                obj.setdefault("rationale_public", "")
                
                # Fix category field - ensure it's always a valid enum value
                category = obj.get("category", "vague")
                if not category or category not in ["policy", "env", "teammate", "vague"]:
                    obj["category"] = "vague"
                
                obj.setdefault("teammate_behavior", "")
                obj.setdefault("memory_writes", [])
                obj.setdefault("low_level_override", None)
                obj.setdefault("intervention_reason", "")
                
                # Try validation again - if it still fails, let it raise
                validate(instance=obj, schema=schema)
            
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
        "chain_of_thought": {"type": "string", "maxLength": 500},
        "memory_writes": {
            "type": "array",
            "items": {"type": "object"},
            "default": []
        },
        "low_level_override": {
            "type": ["string", "null"],
            "enum": ["move_north", "move_south", "move_east", "move_west", "wait", "interact", "stay", None],
            "default": None
        },
        "intervention_reason": {
            "type": "string",
            "maxLength": 300,
            "default": ""
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
			"Return JSON with steps, confidence, rationale_public, category, teammate_behavior, and memory_writes."
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
	
	def _record_human_intervention(self, human_msg: HumanMessage, state, agent_index: int):
		"""Record human intervention with complete state information."""
		try:
			# Capture state information
			ego_pos = None
			mate_pos = None
			ego_holding = None
			mate_holding = None
			
			players = getattr(state, "players", [])
			if len(players) > agent_index:
				ego = players[agent_index]
				ego_pos = [int(ego.position[0]), int(ego.position[1])]
				ego_holding = str(ego.held_object.name) if ego.held_object else "nothing"
			if len(players) > (1 - agent_index):
				mate = players[1 - agent_index]
				mate_pos = [int(mate.position[0]), int(mate.position[1])]
				mate_holding = str(mate.held_object.name) if mate.held_object else "nothing"
			
			# Record the intervention with complete state information
			self.memory.write_events([
				{"t": human_msg.t,
				 "type": "human_intervention",
				 "text": human_msg.text,
				 "state": {
					 "ego_pos": ego_pos,
					 "ego_holding": ego_holding,
					 "mate_pos": mate_pos,
					 "mate_holding": mate_holding
				 },
				 "previous_ml_action": None,  # Will be filled by caller
				 "corrected_ml_action": None  # Will be filled when action is generated
				 }
			])
			print(f"📝 Recorded intervention with state: ego_pos={ego_pos}, ego_holding={ego_holding}, mate_pos={mate_pos}, mate_holding={mate_holding}")
		except Exception as e:
			print(f"⚠️ Error recording intervention: {e}")
	
	def update_intervention_with_corrected_action(self, corrected_ml_action: str):
		"""Update the most recent human intervention record with the corrected ML action."""
		try:
			# Find the most recent human_intervention event and update it
			for i in range(len(self.memory.episodic) - 1, -1, -1):
				event = self.memory.episodic[i]
				if event.get("type") == "human_intervention" and event.get("corrected_ml_action") is None:
					event["corrected_ml_action"] = corrected_ml_action
					print(f"✅ Updated intervention record with corrected action: {corrected_ml_action}")
					break
		except Exception as e:
			print(f"⚠️ Error updating intervention record: {e}")

	def _store_intervention_pattern(self, plan: Plan, human_msg: HumanMessage, state=None) -> None:
		"""Store intervention pattern in memory for future learning."""
		if not plan.intervention_reason or state is None:
			return
		
		try:
			# Extract key state features for pattern matching
			ego_pos = None
			mate_pos = None
			ego_holding = None
			mate_holding = None
			
			players = getattr(state, "players", [])
			if len(players) > 0:
				ego = players[0]  # Assuming agent_index=0 for ego
				ego_pos = [int(ego.position[0]), int(ego.position[1])]
				ego_holding = str(ego.held_object.name) if ego.held_object else "nothing"
			if len(players) > 1:
				mate = players[1]
				mate_pos = [int(mate.position[0]), int(mate.position[1])]
				mate_holding = str(mate.held_object.name) if mate.held_object else "nothing"
			
			pattern = {
				"timestamp": human_msg.t,
				"human_message": human_msg.text,
				"intervention_reason": plan.intervention_reason,
				"context": {
					"ego_pos": ego_pos,
					"ego_holding": ego_holding,
					"mate_pos": mate_pos,
					"mate_holding": mate_holding,
					"terrain_shape": state.terrain_mtx.shape if hasattr(state, 'terrain_mtx') else None,
				},
				"action_taken": plan.steps[0] if plan.steps else None,
				"low_level_override": plan.low_level_override,
			}
			
			# Store in memory
			if "intervention_patterns" not in self.memory.semantic:
				self.memory.semantic["intervention_patterns"] = []
			
			self.memory.semantic["intervention_patterns"].append(pattern)
			
			# Keep only recent patterns (last 50)
			if len(self.memory.semantic["intervention_patterns"]) > 50:
				self.memory.semantic["intervention_patterns"] = self.memory.semantic["intervention_patterns"][-50:]
			
			print(f"[MEMORY] Stored intervention pattern: {plan.intervention_reason}")
			
		except Exception as e:
			print(f"⚠️ Error storing intervention pattern: {e}")

	def propose_plan(
		self,
		state_prompt: str,
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
		state=None,  # Add state parameter for intervention recording
		agent_index=None,  # Add agent_index for intervention recording
	) -> Plan:
		# Record intervention with complete state information if human message exists
		if human_msg.text.strip() and state is not None and agent_index is not None:
			self._record_human_intervention(human_msg, state, agent_index)
		
		user_payload = {
			"state_prompt": state_prompt,
			"recent_history": recent_history[-self.history_horizon:],
			"memory_view": self.memory.prompt_view(),
			"human_message": {
				"t": human_msg.t,
				"text": human_msg.text,
			}
		}

		# Use LLM for all interventions - it will detect low-level commands through reasoning
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
		
		# Always use a valid category (never empty string)
		category = raw.get("category", "vague")
		if not category or category.strip() == "":
			category = "vague"
		
		plan = Plan(
			steps=validated_steps,
			confidence=float(raw.get("confidence", 0.5)),
			rationale_public=str(raw.get("rationale_public", ""))[:180],
			category=category,
			teammate_behavior=str(raw.get("teammate_behavior", ""))[:220],
			memory_writes=raw.get("memory_writes", []),
			low_level_override=raw.get("low_level_override"),
			intervention_reason=str(raw.get("intervention_reason", ""))[:300],
			chain_of_thought=str(raw.get("chain_of_thought", ""))[:500],
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
					"since_t": human_msg.t
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
		# Only write plan event if there's no human intervention (human_intervention type is written by apply_human_intervention)
		if not human_msg.text.strip():
			self.memory.write_events([
				{"t": human_msg.t, "type": "plan", "steps": plan.steps, "conf": plan.confidence}
			])
		else:
			# For human interventions, update the intervention with corrected action and record plan
			self.update_intervention_with_corrected_action(plan.steps[0] if plan.steps else "none")
			self.memory.write_events([
				{"t": human_msg.t, "type": "plan", "steps": plan.steps, "conf": plan.confidence, "category": plan.category}
			])
			
			# Store intervention pattern for learning
			self._store_intervention_pattern(plan, human_msg, state)

		return plan

	def _gate_memory_writes(self, writes: List[Dict[str, Any]], category: str = "vague") -> Dict[str, Any]:
		"""
		Convert a list of 'memory_writes' entries into a single semantic patch.
		Keep it conservative: allow updates only in whitelisted regions and small payloads.
		Allow small world_state patches for any category.
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
		
		# Allow small world_state patches for any category
		ALLOWED_KEYS = {"teammate_model", "human_prefs", "playbook", "afford_safety", "hotspots", "open_questions", "layout_facts", "world_state", "intervention_patterns"}
		
		MAX_BYTES_PER_WRITE = 2000
		MAX_WS_ITEMS = 6
		
		def _is_small_world_state(v: Any) -> bool:
			if not isinstance(v, dict):
				return True
			ps = v.get("pot_status", [])
			oc = v.get("onion_caches", [])
			return len(ps) <= MAX_WS_ITEMS and len(oc) <= MAX_WS_ITEMS
		
		for i, w in enumerate(writes):
			print(f"[MEMORY] Processing write {i}: {list(w.keys())}")
			for k, v in w.items():
				if k not in ALLOWED_KEYS:
					print(f"[MEMORY] Rejecting key '{k}' (not in ALLOWED_KEYS)")
					continue
				if isinstance(v, (dict, list)) and len(json.dumps(v)) > MAX_BYTES_PER_WRITE:
					print(f"[MEMORY] Rejecting key '{k}' (payload too large)")
					continue
				if k == "world_state" and not _is_small_world_state(v):
					print(f"[MEMORY] Rejecting world_state (too large)")
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
