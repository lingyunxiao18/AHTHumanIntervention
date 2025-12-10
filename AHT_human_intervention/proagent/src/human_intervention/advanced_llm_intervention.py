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
	chain_of_thought: str                # Explicit CoT reasoning with full reasoning process
	category: Category                    # model's classification of the human msg
	teammate_behavior: str = ""           # Add this field
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)
	low_level_override: Optional[str] = None  # Direct low-level action override (e.g., "move_north", "wait", "interact")
	intervention_reason: str = ""         # LLM's inference of why human provided this intervention

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,  # Direct list of ML_action strings
			"chain_of_thought": self.chain_of_thought,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,  # Include it
			"memory_writes": self.memory_writes,
			"low_level_override": self.low_level_override,
			"intervention_reason": self.intervention_reason,
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

	def upsert_semantic(self, patch: Dict[str, Any]) -> None:
		"""
		Conservative merge of small dict patches from the model (after gating).
		Only whitelisted top-level keys are merged.
		"""
		WHITELIST = {"layout_facts", "world_state", "teammate_model", "human_prefs",
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
				# Deep merge for nested dicts like layout_facts and world_state
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
			"playbook": sem.get("playbook", [])[-4:],  # Keep last 4 entries
			"summary": self.summarize_recent(),
			"intervention_patterns": sem.get("intervention_patterns", [])[-4:],  # Keep last 4 entries
		}

@dataclass
class HumanMessage:
	t: int
	text: str

class LLMClient:
    """
    Adapter around the OpenAI API using gpt-5-mini with enforced JSON mode.
    Falls back to heuristic extraction if response_format fails.
    """

    def __init__(self, openai_client: Optional[OpenAI] = None, model: str = "gpt-5-mini"):
        self.client = openai_client or OpenAI()
        self.model = model

    def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Request structured JSON output from GPT-5-mini.
        Returns parsed dict or raises ValueError with raw LLM text on failure.
        
        Args:
            max_retries: Maximum number of retry attempts for empty responses
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=1,
                    max_completion_tokens=2048,
                )
                
                # Check if response is valid
                if not response or not response.choices:
                    raise ValueError(f"Empty response from API (no choices). Attempt {attempt + 1}/{max_retries}")
                
                # Extract the text content from the first choice
                raw = response.choices[0].message.content
                
                # Check if content is empty or None
                if not raw or not raw.strip():
                    if attempt < max_retries - 1:
                        print(f"⚠️ Empty response from LLM (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise ValueError(f"Empty response from API after {max_retries} attempts. Response object: {response}")
                
                # Parse JSON
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {e}. Raw LLM output: {raw!r}")
                
                # Remove any properties not in the schema (e.g., memory_writes when memory is disabled)
                # This is necessary because the schema has additionalProperties: False
                allowed_properties = set(schema.get("properties", {}).keys())
                parsed = {k: v for k, v in parsed.items() if k in allowed_properties}
                
                # Repair common issues before validation
                # Fix category field - ensure it's always a valid enum value
                category = parsed.get("category", "vague")
                if not category or category.strip() == "" or category not in ["policy", "env", "teammate", "vague"]:
                    parsed["category"] = "vague"
                
                # Ensure required fields exist (only if they're in the schema)
                parsed.setdefault("steps", ["wait(1)"])
                if "chain_of_thought" in allowed_properties:
                    parsed.setdefault("chain_of_thought", "No chain of thought provided")
                parsed.setdefault("teammate_behavior", "")
                if "memory_writes" in allowed_properties:
                    parsed.setdefault("memory_writes", [])
                if "low_level_override" in allowed_properties:
                    parsed.setdefault("low_level_override", None)
                if "intervention_reason" in allowed_properties:
                    parsed.setdefault("intervention_reason", "")
                
                # Truncate fields that have maxLength constraints
                if "chain_of_thought" in parsed and isinstance(parsed["chain_of_thought"], str):
                    parsed["chain_of_thought"] = parsed["chain_of_thought"][:2048]
                if "teammate_behavior" in parsed and isinstance(parsed["teammate_behavior"], str):
                    parsed["teammate_behavior"] = parsed["teammate_behavior"][:220]
                if "intervention_reason" in parsed and isinstance(parsed["intervention_reason"], str):
                    parsed["intervention_reason"] = parsed["intervention_reason"][:300]
                
                # Validate against schema
                try:
                    validate(instance=parsed, schema=schema)
                except ValidationError as e:
                    raise ValueError(f"JSON did not match schema: {e.message}. Parsed: {parsed!r}")
                
                print(f"Parsed: {parsed}")
                return parsed
                
            except ValueError as e:
                # If it's a ValueError (empty response or JSON parse error), retry if possible
                last_error = e
                if attempt < max_retries - 1 and "Empty response" in str(e):
                    print(f"⚠️ Retrying due to error: {e}")
                    import time
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    # If it's a JSON parse error or we've exhausted retries, raise immediately
                    raise
            except Exception as e:
                # For other exceptions, don't retry
                raise RuntimeError(f"LLMClient.respond_json failed: {e}")
        
        # If we exhausted all retries, raise the last error
        raise RuntimeError(f"LLMClient.respond_json failed after {max_retries} attempts: {last_error}")

PLAN_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["steps", "chain_of_thought", "category", "teammate_behavior"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "enum": ALL_VALID_ML_ACTIONS
            }
        },
        "chain_of_thought": {"type": "string", "maxLength": 2048},
        "category": {"type": "string", "enum": ["policy", "env", "teammate", "vague"]},
        "teammate_behavior": {"type": "string", "maxLength": 220},
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

def _get_plan_json_schema(enable_cot: bool = True, enable_memory: bool = True) -> Dict[str, Any]:
	"""
	Generate JSON schema based on enabled features.
	
	Args:
		enable_cot: Whether to require chain_of_thought
		enable_memory: Whether to include memory_writes
		
	Returns:
		Modified JSON schema dictionary
	"""
	import copy
	schema = copy.deepcopy(PLAN_JSON_SCHEMA)
	required_fields = ["steps", "category", "teammate_behavior"]
	
	# Add chain_of_thought to required only if CoT is enabled
	if enable_cot:
		required_fields.append("chain_of_thought")
	
	schema["required"] = required_fields
	
	# Remove memory_writes from properties if memory is disabled
	if not enable_memory:
		schema["properties"].pop("memory_writes", None)
	
	# Note: intervention_reason is always kept in schema - it's empty when no intervention, filled when human intervenes
	
	return schema

def _load_system_rules() -> str:
	"""Load system rules from external file."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	rules_file = os.path.join(script_dir, "advanced_llm_system_rules.txt")
	try:
		with open(rules_file, 'r', encoding='utf-8') as f:
			return f.read()
	except FileNotFoundError:
		return (
			"You control Player0 in Overcooked-AI. Goal: deliver soups quickly and safely.\n"
			"Use ONLY these ML actions: " + ", ".join(ALL_VALID_ML_ACTIONS) + "\n"
			"Return JSON with steps, chain_of_thought, category, teammate_behavior, and memory_writes."
		)

SYSTEM_RULES = _load_system_rules()

def _get_system_rules(enable_cot: bool = True, enable_memory: bool = True) -> str:
	"""
	Generate system rules based on enabled features.
	
	Args:
		enable_cot: Whether to include chain-of-thought instructions
		enable_memory: Whether to include memory-related instructions
		
	Returns:
		Modified system rules string
	"""
	base_rules = SYSTEM_RULES
	
	# If both are enabled, return full rules
	if enable_cot and enable_memory:
		return base_rules
	
	result = base_rules
	
	# Remove CoT section if disabled
	if not enable_cot:
		# Remove the entire THINK (CHAIN OF THOUGHT) section
		import re
		# Match from "THINK (CHAIN OF THOUGHT" to and including "OUTPUT:", replace with just "OUTPUT:"
		cot_pattern = r'THINK \(CHAIN OF THOUGHT.*?\nOUTPUT:'
		result = re.sub(cot_pattern, 'OUTPUT:', result, flags=re.DOTALL)
		
		# Update schema description to indicate CoT is empty
		result = re.sub(r'"chain_of_thought":\s*"[^"]*"', '"chain_of_thought": ""', result)
		result = result.replace(
			'"chain_of_thought": "Your complete reasoning from THINK section above (be explicit and structured)",',
			'"chain_of_thought": "",  // Empty when CoT is disabled'
		)
	
	# Remove memory-related sections if disabled
	if not enable_memory:
		# Remove memory_view from INPUTS
		result = re.sub(r'\(c\)\s*memory_view[^\n]*\n?', '', result)
		
		# Remove EVENT-BASED INTERVENTION REUSE section
		event_pattern = r'────────────────────────────────────────────\nEVENT-BASED INTERVENTION REUSE.*?(?=────────────────────────────────────────────|CONSTRAINTS)'
		result = re.sub(event_pattern, '', result, flags=re.DOTALL)
		
		# Remove MEMORY AND APPLYING SIMILAR INTERVENTIONS section
		memory_pattern = r'────────────────────────────────────────────\nMEMORY AND APPLYING SIMILAR INTERVENTIONS.*?(?=────────────────────────────────────────────|CONSTRAINTS)'
		result = re.sub(memory_pattern, '', result, flags=re.DOTALL)
		
		# Remove memory_writes from schema
		result = re.sub(r'"memory_writes":\s*\[[^\]]*\],?\s*//.*?\n?', '', result)
		result = result.replace('"memory_writes": [ ... ],                                    // optional structured updates', '')
		
		# Update teammate_behavior description to not mention memory_view
		result = result.replace(
			'summarizing the teammate\'s recent role and pattern using `memory_view` and `recent_history`.',
			'summarizing the teammate\'s recent role and pattern using `recent_history`.'
		)
	
	# Note: intervention_reason is always kept in system prompt - it's empty when no intervention, filled when human intervenes
	
	return result

class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter.
	- compose the prompt from: state snapshot, short history, memory.view, human msg (+ optional heuristic category)
	- call LLM in JSON mode
	- return Plan + apply memory_writes (gated) to AgentMemory
	"""
	def __init__(self, llm: LLMClient, memory: AgentMemory, history_horizon: int = 8, 
	             enable_cot: bool = True, enable_memory: bool = True):
		self.llm = llm
		self.memory = memory
		self.history_horizon = history_horizon
		self.enable_cot = enable_cot
		self.enable_memory = enable_memory
	
	def _record_human_intervention(self, human_msg: HumanMessage, state, agent_index: int, events=None):
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
				 "events": events or [],  # Store events that triggered this intervention
				 "previous_ml_action": None,  # Will be filled by caller
				 "corrected_ml_action": None  # Will be filled when action is generated
				 }
			])
		except Exception as e:
			pass
	
	def update_intervention_with_corrected_action(self, corrected_ml_action: str):
		"""Update the most recent human intervention record with the corrected ML action."""
		try:
			# Find the most recent human_intervention event and update it
			for i in range(len(self.memory.episodic) - 1, -1, -1):
				event = self.memory.episodic[i]
				if event.get("type") == "human_intervention" and event.get("corrected_ml_action") is None:
					event["corrected_ml_action"] = corrected_ml_action
					break
		except Exception as e:
			pass

	def _store_intervention_pattern(self, plan: Plan, human_msg: HumanMessage, state=None, events=None) -> None:
		"""Store intervention pattern in memory for future learning."""
		if not self.enable_memory or not plan.intervention_reason or state is None:
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
				"events_triggered": events or [],  # Store events that triggered this intervention
				"action_taken": plan.steps[0] if plan.steps else None,
				"low_level_override": plan.low_level_override,
			}
			
			# Store in memory
			if "intervention_patterns" not in self.memory.semantic:
				self.memory.semantic["intervention_patterns"] = []
			
			self.memory.semantic["intervention_patterns"].append(pattern)
			print("Storing pattern:", pattern)
			
		except Exception as e:
			pass

	def propose_plan(
		self,
		state_prompt: str,
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
		state=None,  # Add state parameter for intervention recording
		agent_index=None,  # Add agent_index for intervention recording
		events=None,  # Add events parameter for event-based triggers
	) -> Plan:
		# Record intervention with complete state information if human message exists and memory is enabled
		if self.enable_memory and human_msg.text.strip() and state is not None and agent_index is not None:
			self._record_human_intervention(human_msg, state, agent_index, events=events)
		
		# Extract events from state_prompt if not provided directly
		if events is None:
			# Try to extract from state_prompt (format: "EVENTS: [...]")
			import re
			events_match = re.search(r'EVENTS:\s*\[(.*?)\]', state_prompt)
			if events_match:
				events_str = events_match.group(1)
				events = [e.strip().strip('"\'') for e in events_str.split(',') if e.strip()]
			else:
				events = []
		
		user_payload = {
			"state_prompt": state_prompt,
			"recent_history": recent_history[-self.history_horizon:],
			"human_message": {
				"t": human_msg.t,
				"text": human_msg.text,
			},
			"user_events": events,  # Add events to payload
		}
		
		# Conditionally include memory_view if memory is enabled
		if self.enable_memory:
			user_payload["memory_view"] = self.memory.prompt_view()

		# Optional: Automatic pattern matching for event-based triggers
		# Pre-fill low_level_override if a matching pattern is found (gives LLM a hint)
		# Only works if memory is enabled
		matching_pattern = None
		if self.enable_memory and events:
			patterns = self.memory.semantic.get("intervention_patterns", [])
			matching = []
			for p in patterns:
				p_events = p.get("events_triggered", [])
				# Check if all pattern events are in current events (subset match)
				if p_events and all(e in events for e in p_events):
					matching.append(p)
			if matching:
				# Use most recent matching pattern
				matching_pattern = matching[-1]
				if matching_pattern.get("low_level_override"):
					# Pre-fill the override as a hint (LLM can still override if needed)
					print(f"🔍 Auto-matched pattern: {matching_pattern.get('events_triggered')} → {matching_pattern.get('low_level_override')}")

		# Get system rules based on enabled features
		system_rules = _get_system_rules(enable_cot=self.enable_cot, enable_memory=self.enable_memory)
		
		# Get JSON schema based on enabled features
		schema = _get_plan_json_schema(enable_cot=self.enable_cot, enable_memory=self.enable_memory)
		
		# Use LLM for all interventions - it will detect low-level commands through reasoning
		raw = self.llm.respond_json(schema, system_rules, user_payload)
		
		# If no human intervention and we have a matching pattern, apply the override automatically
		if not human_msg.text.strip() and matching_pattern and matching_pattern.get("low_level_override"):
			# Only auto-apply if LLM didn't already provide an override
			if not raw.get("low_level_override"):
				raw["low_level_override"] = matching_pattern["low_level_override"]
				print(f"✅ Auto-applied pattern override: {raw['low_level_override']}")
		
		# Validate ML actions are in allowed set
		steps = raw.get("steps", [])
		validated_steps = []
		for step in steps:
			if step in ALL_VALID_ML_ACTIONS:
				validated_steps.append(step)
		
		# Always use a valid category (never empty string)
		category = raw.get("category", "vague")
		if not category or category.strip() == "":
			category = "vague"
		
		# Conditionally set chain_of_thought based on enable_cot
		if self.enable_cot:
			chain_of_thought_raw = str(raw.get("chain_of_thought", ""))
			chain_of_thought = chain_of_thought_raw[:2048] if chain_of_thought_raw else "No chain of thought provided"
		else:
			chain_of_thought = ""  # Empty CoT when disabled
		
		# Conditionally include memory_writes based on enable_memory
		memory_writes = raw.get("memory_writes", []) if self.enable_memory else []
		
		# Always extract intervention_reason (it's always in schema)
		# It will be empty string when no human intervention, filled when human intervenes
		intervention_reason = str(raw.get("intervention_reason", ""))[:300]
		
		plan = Plan(
			steps=validated_steps,
			chain_of_thought=chain_of_thought,
			category=category,
			teammate_behavior=str(raw.get("teammate_behavior", ""))[:220],
			memory_writes=memory_writes,
			low_level_override=raw.get("low_level_override"),
			intervention_reason=intervention_reason,
		)

		# Plan hygiene checks
		# Hard cap actions to <= 3 and ensure at least one safe step
		if len(plan.steps) > 3:
			plan.steps = plan.steps[:3]
		if not plan.steps:
			plan.steps = ["wait(1)"]

		# NEW: ingest LLM-authored teammate behavior (if present and memory enabled)
		if self.enable_memory:
			tb = (raw.get("teammate_behavior") or "").strip()
			if tb:
				plan.memory_writes.append({
					"teammate_model": {
						"behavior_description": tb,
						"since_t": human_msg.t
					}
				})
			
			safe_patch = self._gate_memory_writes(plan.memory_writes, plan.category)
			if safe_patch:
				self.memory.upsert_semantic(safe_patch)
			
			# Only write plan event if there's no human intervention (human_intervention type is written by apply_human_intervention)
			if not human_msg.text.strip():
				self.memory.write_events([
					{"t": human_msg.t, "type": "plan", "steps": plan.steps}
				])
			else:
				# For human interventions, update the intervention with corrected action and record plan
				self.update_intervention_with_corrected_action(plan.steps[0] if plan.steps else "none")
				self.memory.write_events([
					{"t": human_msg.t, "type": "plan", "steps": plan.steps, "category": plan.category}
				])
				
				# Store intervention pattern for learning
				self._store_intervention_pattern(plan, human_msg, state, events=events)

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
			return patch
		
		# Allow small world_state patches for any category
		ALLOWED_KEYS = {"teammate_model", "human_prefs", "playbook", "layout_facts", "world_state", "intervention_patterns"}
		
		MAX_BYTES_PER_WRITE = 2000
		MAX_WS_ITEMS = 6
		
		def _is_small_world_state(v: Any) -> bool:
			if not isinstance(v, dict):
				return True
			ps = v.get("pot_status", [])
			oc = v.get("onion_caches", [])
			return len(ps) <= MAX_WS_ITEMS and len(oc) <= MAX_WS_ITEMS
		
		for i, w in enumerate(writes):
			for k, v in w.items():
				if k not in ALLOWED_KEYS:
					continue
				if isinstance(v, (dict, list)) and len(json.dumps(v)) > MAX_BYTES_PER_WRITE:
					continue
				if k == "world_state" and not _is_small_world_state(v):
					continue
				
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
		return patch
