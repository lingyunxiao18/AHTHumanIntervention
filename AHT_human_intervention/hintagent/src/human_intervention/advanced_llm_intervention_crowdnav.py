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

# Allowed waypoint actions (0-7)
CROWDNAV_ALLOWED_WAYPOINT_ACTIONS = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]

Category = Literal["policy", "env", "teammate", "vague"]

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
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)
	low_level_override: Optional[int] = None  # Direct waypoint override (0-7)
	intervention_reason: str = ""         # LLM's inference of why human provided this intervention

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,
			"chain_of_thought": self.chain_of_thought,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,
			"memory_writes": self.memory_writes,
			"low_level_override": self.low_level_override,
			"intervention_reason": self.intervention_reason,
		}

class AgentMemory:
	"""
	CrowdNav-specific persistent memory with structured navigation state tracking.
	- semantic: long-lived facts, rules, and dynamic navigation state
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 500):
		self.semantic: Dict[str, Any] = {
			# Navigation patterns and preferences
			"navigation_patterns": {
				"preferred_directions": [],      # [0, 3, 0] - preferred waypoint directions
				"avoid_directions": [],          # [1, 2] - directions to avoid
				"successful_paths": [],          # [{"waypoints": [0,3,0], "outcome": "success"}]
			},
			# Obstacle and crowd memory
			"obstacle_memory": [],               # [{"pos": [x,y], "type": "crowd"|"static", "timestamp": t}]
			# Teammate hypothesis & coordination contract
			"teammate_model": {
				"since_t": None,
				"behavior_description": "",
			},
			# Human/operator preferences or "house rules"
			"human_prefs": {
				# e.g., "avoid_crowds": True, "prefer_direct_path": True
			},
			# Compact "if…then…" contracts derived from interventions or LLM
			"playbook": [
				# {"if": "crowd_ahead AND clear_alternative", "then": [2, 0, 3], "note": "detour around crowd"}
			],
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

	def upsert_semantic(self, patch: Dict[str, Any]) -> None:
		"""
		Conservative merge of small dict patches from the model (after gating).
		Only whitelisted top-level keys are merged.
		"""
		WHITELIST = {"navigation_patterns", "obstacle_memory", "teammate_model", "human_prefs",
					 "playbook", "intervention_patterns"}
		for k, v in patch.items():
			if k not in WHITELIST:
				continue
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
				# Deep merge for nested dicts like navigation_patterns
				self._deep_merge(self.semantic[k], v)
			else:
				self.semantic[k] = v

	def _merge_list_by_pos(self, dest: List[Dict[str, Any]], src: List[Dict[str, Any]], key="pos", cap=32):
		"""Merge lists by position to avoid duplicates."""
		# Build index mapping position tuples to indices
		idx = {}
		for i, it in enumerate(dest):
			if key in it:
				val = it.get(key, [])
				# Only create tuple if val is a list/tuple that can be converted
				if isinstance(val, (list, tuple)):
					try:
						idx[tuple(val)] = i
					except TypeError:
						# Skip if value can't be converted to tuple (e.g., contains unhashable types)
						pass
		
		# Merge source items
		for it in src:
			if key in it:
				val = it.get(key, [])
				if isinstance(val, (list, tuple)):
					try:
						p = tuple(val)
						if p in idx:
							dest[idx[p]].update(it)
						else:
							dest.append(it)
					except TypeError:
						# Skip if value can't be converted to tuple
						dest.append(it)
				else:
					dest.append(it)
			else:
				dest.append(it)
		
		# Cap the list size
		if len(dest) > cap:
			del dest[:-cap]

	def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
		"""Deep merge update_dict into base_dict."""
		for key, value in update_dict.items():
			if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
				self._deep_merge(base_dict[key], value)
			elif key == "obstacle_memory" and isinstance(value, list):
				base = base_dict.get(key, [])
				self._merge_list_by_pos(base, value)
				base_dict[key] = base
			else:
				base_dict[key] = value

	def prompt_view(self) -> Dict[str, Any]:
		"""Compact view for the model prompt."""
		sem = self.semantic
		tm = sem["teammate_model"]
		np = sem.get("navigation_patterns", {})
		
		# Keep short: last 2-3 key obstacle memories
		key_obstacles = sem.get("obstacle_memory", [])[-3:]
		
		return {
			"navigation_patterns": {
				"preferred_directions": np.get("preferred_directions", [])[-5:],
				"avoid_directions": np.get("avoid_directions", []),
			},
			"obstacle_memory": key_obstacles,
			"teammate_model": {
				"since_t": tm.get("since_t"),
				"behavior_description": tm.get("behavior_description"),
			},
			"playbook": sem.get("playbook", [])[-4:],  # Keep last 4 entries
			"summary": self.summarize_recent(),
			"intervention_patterns": sem.get("intervention_patterns", [])[-4:],  # Keep last 4 entries
		}


@dataclass
class HumanMessage:
	t: int
	text: str

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

    def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request structured JSON output from GPT-5-mini.
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
                max_completion_tokens=2048,  
				reasoning_effort="low",
            )

            try:
                text_out = response.choices[0].message.content
            except Exception:
                text_out = getattr(response, "choices", [{}])[0].get("message", {}).get("content", None) or str(response)

            # Handle empty response (e.g., hit token limit)
            if not text_out or not text_out.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", None) if response.choices else None
                if finish_reason == "length":
                    print(f"[LLM-WARNING] Response hit token limit ({response.usage.completion_tokens if hasattr(response, 'usage') else 'unknown'} tokens), using fallback")
                else:
                    print(f"[LLM-WARNING] Empty response (finish_reason: {finish_reason}), using fallback")
                # Return fallback plan
                obj = {
                    "steps": [0],
                    "chain_of_thought": "Response was truncated or empty, using safe fallback",
                    "category": "vague",
                    "teammate_behavior": "",
                    "memory_writes": [],
                    "low_level_override": None,
                    "intervention_reason": ""
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
                            print(f"[LLM-WARNING] Response appears truncated, using fallback")
                            obj = {
                                "steps": [0],
                                "chain_of_thought": "Response was truncated, using safe fallback",
                                "category": "vague",
                                "teammate_behavior": "",
                                "memory_writes": [],
                                "low_level_override": None,
                                "intervention_reason": ""
                            }
                        else:
                            raise ValueError(f"Could not parse JSON: {err}\nRaw: {text_out}")
            
            print(f"[LLM-DEBUG] Raw LLM response: {obj}")
            
            # Remove any properties not in the schema
            # This is necessary because the schema has additionalProperties: False
            allowed_properties = set(schema.get("properties", {}).keys())
            obj = {k: v for k, v in obj.items() if k in allowed_properties}
            
            # Repair common issues before validation
            # Ensure required fields exist FIRST (before validation)
            obj.setdefault("steps", [0])
            
            # Fix category field - ensure it's always a valid enum value
            category = obj.get("category", "vague")
            if not category or category.strip() == "" or category not in ["policy", "env", "teammate", "vague"]:
                obj["category"] = "vague"
            if "chain_of_thought" in allowed_properties:
                obj.setdefault("chain_of_thought", "No chain of thought provided")
            obj.setdefault("teammate_behavior", "")
            if "memory_writes" in allowed_properties:
                obj.setdefault("memory_writes", [])
            if "low_level_override" in allowed_properties:
                obj.setdefault("low_level_override", None)
            if "intervention_reason" in allowed_properties:
                obj.setdefault("intervention_reason", "")
            
            # Truncate fields that have maxLength constraints before validation
            if "chain_of_thought" in obj and isinstance(obj["chain_of_thought"], str):
                obj["chain_of_thought"] = obj["chain_of_thought"][:512]  # Match schema maxLength
            if "teammate_behavior" in obj and isinstance(obj["teammate_behavior"], str):
                obj["teammate_behavior"] = obj["teammate_behavior"][:220]  # Match schema maxLength
            if "intervention_reason" in obj and isinstance(obj["intervention_reason"], str):
                obj["intervention_reason"] = obj["intervention_reason"][:300]  # Match schema maxLength
            
            # Validate against schema
            try:
                validate(instance=obj, schema=schema)
            except ValidationError as e:
                raise ValueError(f"JSON did not match schema: {e.message}. Parsed: {obj!r}")
            
            print(f"[LLM-DEBUG] Parsed: {obj}")
            return obj

        except Exception as e:
            raise RuntimeError(f"LLMClient.respond_json failed: {e}")


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
			"maxItems": 3,
		},
		"chain_of_thought": {"type": "string", "maxLength": 512}, 
		"category": {"type": "string", "enum": ["policy", "env", "teammate", "vague"]},
		"teammate_behavior": {"type": "string", "maxLength": 220},
		"memory_writes": {
			"type": "array",
			"items": {"type": "object"},
			"default": []
		},
		"low_level_override": {
			"type": ["integer", "null"],
			"enum": [0, 1, 2, 3, 4, 5, 6, 7, None],
			"default": None
		},
		"intervention_reason": {
			"type": "string",
			"maxLength": 300,
			"default": ""
		},
	},
}


class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter for human interventions in CrowdNav-AHT.
	Uses LLM to generate structured plans from text observations and human messages.
	"""
	
	def __init__(self, llm_client: LLMClient, memory: AgentMemory, history_horizon: int = 6):
		self.llm = llm_client
		self.memory = memory
		self.history_horizon = history_horizon
		
		# Load system rules
		script_dir = os.path.dirname(os.path.abspath(__file__))
		rules_file = os.path.join(script_dir, "advanced_llm_system_rules_crowdnav.txt")
		try:
			with open(rules_file, 'r', encoding='utf-8') as f:
				self.system_rules = f.read()
		except Exception as e:
			print(f"[WARNING] Could not load system rules: {e}")
			self.system_rules = "You are a navigation assistant for a robot in a crowd."

	def interpret(
		self,
		state_prompt: str,
		recent_history: List[Dict[str, Any]],
		human_message: Optional[str] = None,
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
		memory_view = self.memory.prompt_view()
		
		# Build user prompt
		user_prompt = {
			"state_prompt": state_prompt,
			"recent_history": recent_history[-self.history_horizon:],
			"memory_view": memory_view,
		}
		
		if human_message:
			user_prompt["human_message"] = human_message
		
		raw_plan = self.llm.respond_json(PLAN_JSON_SCHEMA, self.system_rules, user_prompt)
		# Apply memory writes
		for write in raw_plan.get("memory_writes", []):
			self.memory.upsert_semantic(write)
		
		# Record in episodic memory
		self.memory.write_events([{
			"type": "plan",
			"t": len(self.memory.episodic),
			"steps": raw_plan.get("steps", []),
			"category": raw_plan.get("category", ""),
			"human_message": human_message,
		}])
		
		# Convert to Plan dataclass
		return Plan(
			steps=raw_plan.get("steps", [0]),
			chain_of_thought=raw_plan.get("chain_of_thought", ""),
			category=raw_plan.get("category", "vague"),
			teammate_behavior=raw_plan.get("teammate_behavior", ""),
			memory_writes=raw_plan.get("memory_writes", []),
			low_level_override=raw_plan.get("low_level_override"),
			intervention_reason=raw_plan.get("intervention_reason", ""),
		)

