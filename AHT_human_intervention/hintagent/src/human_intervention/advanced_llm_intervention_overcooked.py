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

Category = Literal["policy", "env", "teammate", "general_hint"]

# ---------------------------------------------------------------------
# CoT + Memory Interpreter for Human Interventions in Overcooked-AI
# Inputs: observable state, human message, recent K-step history
# Outputs: structured Plan JSON, one-sentence rationale, message category
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# ICL Helper Functions for Memory Case Building
# ---------------------------------------------------------------------

def _describe_situation_for_memory(state, events=None, psi_text: Optional[str] = None) -> str:
	"""Short natural-language description of the local situation for ICL."""
	if psi_text:
		base = psi_text.strip()
	else:
		try:
			players = getattr(state, "players", [])
			terrain_shape = getattr(state, "terrain_mtx", None)
			ego = players[0] if len(players) > 0 else None
			mate = players[1] if len(players) > 1 else None

			ego_pos = list(ego.position) if ego is not None else None
			mate_pos = list(mate.position) if mate is not None else None
			ego_hold = str(ego.held_object.name) if ego is not None and ego.held_object else "nothing"
			mate_hold = str(mate.held_object.name) if mate is not None and mate.held_object else "nothing"

			parts = []
			if terrain_shape is not None:
				parts.append(f"Layout size {terrain_shape[1]}x{terrain_shape[0]}")
			if ego_pos is not None:
				parts.append(f"ego at {ego_pos} holding {ego_hold}")
			if mate_pos is not None:
				parts.append(f"teammate at {mate_pos} holding {mate_hold}")
			base = "; ".join(parts) if parts else "situation unknown"
		except Exception:
			base = "situation unknown"
	if events:
		return base + " | events: " + ", ".join(events)
	return base


def _describe_intervention_for_memory(plan: "Plan", human_msg: "HumanMessage") -> str:
	"""Short text description of what the human asked and what the agent did."""
	steps = plan.steps or []
	steps_txt = ", ".join(steps) if steps else "no planned ML steps"
	ll_override = plan.low_level_override or "none"
	msg = human_msg.text.strip() or "no explicit human text"
	return (
		f'human said: "{msg}". '
		f"Interpreter chose ML steps: [{steps_txt}], "
		f"low_level_override: {ll_override}."
	)

@dataclass
class Plan:
	steps: List[str]  # List of ML_action strings instead of MacroStep objects
	chain_of_thought: str                # Explicit CoT reasoning with full reasoning process
	category: Category                    # model's classification of the human msg
	teammate_behavior: str = ""           # Add this field
	low_level_override: Optional[str] = None  # Direct low-level action override (e.g., "move_north", "wait", "interact")

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,  # Direct list of ML_action strings
			"chain_of_thought": self.chain_of_thought,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,  # Include it
			"low_level_override": self.low_level_override,
		}

class AgentMemory:
	"""
	Overcooked-specific persistent memory focused on intervention examples.
	- semantic: successful intervention patterns only
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 500, mdp=None):
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
        def _call(system_prompt, user_payload):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )
            if not response or not response.choices:
                raise ValueError("Empty response from API (no choices).")
            return response.choices[0].message.content

        def _fallback(reason: str):
            return {
                "steps": ["wait(1)"],
                "chain_of_thought": f"Safe fallback: {reason}",
                "category": "general_hint",
                "teammate_behavior": "",
                "low_level_override": None,
            }
        for attempt in range(max_retries):
            try:
                raw = _call(system, user)
                
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
                
                # Remove any properties not in the schema
                # This is necessary because the schema has additionalProperties: False
                allowed_properties = set(schema.get("properties", {}).keys())
                parsed = {k: v for k, v in parsed.items() if k in allowed_properties}
                
                # Repair common issues before validation
                # Fix category field - ensure it's always a valid enum value
                category = parsed.get("category", "general_hint")
                if not category or category.strip() == "" or category not in ["policy", "env", "teammate", "general_hint"]:
                    parsed["category"] = "general_hint"
                
                # Ensure required fields exist (only if they're in the schema)
                parsed.setdefault("steps", ["wait(1)"])
                if "chain_of_thought" in allowed_properties:
                    parsed.setdefault("chain_of_thought", "No chain of thought provided")
                parsed.setdefault("teammate_behavior", "")
                if "low_level_override" in allowed_properties:
                    parsed.setdefault("low_level_override", None)
                
                # Truncate fields that have maxLength constraints
                if "chain_of_thought" in parsed and isinstance(parsed["chain_of_thought"], str):
                    parsed["chain_of_thought"] = parsed["chain_of_thought"][:2048]
                if "teammate_behavior" in parsed and isinstance(parsed["teammate_behavior"], str):
                    parsed["teammate_behavior"] = parsed["teammate_behavior"][:220]
                
                # Validate against schema (one repair attempt)
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
                    return _fallback(str(e))
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
        "category": {"type": "string", "enum": ["policy", "env", "teammate", "general_hint"]},
        "teammate_behavior": {"type": "string", "maxLength": 220},
        "low_level_override": {
            "type": ["string", "null"],
            "enum": ["move_north", "move_south", "move_east", "move_west", "wait", "interact", "stay", None],
            "default": None
        },
    }
}

def _get_plan_json_schema(enable_cot: bool = True, enable_memory: bool = True) -> Dict[str, Any]:
	"""
	Generate JSON schema based on enabled features.
	
	Args:
		enable_cot: Whether to require chain_of_thought
		enable_memory: Whether to include memory-related instructions (unused here)
		
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
	return schema

def _load_system_rules() -> str:
	"""Load system rules from external file."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	rules_file = os.path.join(script_dir, "advanced_llm_system_rules_overcooked.txt")
	try:
		with open(rules_file, 'r', encoding='utf-8') as f:
			return f.read()
	except FileNotFoundError:
		return (
			"You control Player0 in Overcooked-AI. Goal: deliver soups quickly and safely.\n"
			"Use ONLY these ML actions: " + ", ".join(ALL_VALID_ML_ACTIONS) + "\n"
			"Return JSON with steps, chain_of_thought, category, teammate_behavior, and low_level_override."
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
		# Remove Memory line from INPUTS
		result = re.sub(r'-\s*Memory:[^\n]*\n?', '', result)
		
		# Remove EVENT-BASED INTERVENTION REUSE section
		event_pattern = r'────────────────────────────────────────────\nEVENT-BASED INTERVENTION REUSE.*?(?=────────────────────────────────────────────|CONSTRAINTS)'
		result = re.sub(event_pattern, '', result, flags=re.DOTALL)
		
		# Remove MEMORY AND APPLYING SIMILAR INTERVENTIONS section
		memory_pattern = r'────────────────────────────────────────────\nMEMORY AND APPLYING SIMILAR INTERVENTIONS.*?(?=────────────────────────────────────────────|CONSTRAINTS)'
		result = re.sub(memory_pattern, '', result, flags=re.DOTALL)
		
		# Update teammate_behavior description to not mention memory_view
		result = result.replace(
			'summarizing the teammate\'s recent role and pattern using `memory_view` and `recent_history`.',
			'summarizing the teammate\'s recent role and pattern using `recent_history`.'
		)
	
	return result

class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter.
	- compose the prompt from: state snapshot, short history, memory.view, human msg (+ optional heuristic category)
	- call LLM in JSON mode
	- return Plan
	"""
	def __init__(self, llm: LLMClient, memory: AgentMemory, history_horizon: int = 8, 
	             enable_cot: bool = True, enable_memory: bool = True):
		self.llm = llm
		self.memory = memory
		self.history_horizon = history_horizon
		self.enable_cot = enable_cot
		self.enable_memory = enable_memory
		self._last_teammate_behavior = ""
		self._pending_patterns: Dict[int, Dict[str, Any]] = {}
	
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

	def _select_icl_examples(self, state=None, events=None, k: int = 4, psi_text: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Select a small set of ICL examples from intervention_patterns."""
		patterns = [
			p for p in self.memory.semantic.get("intervention_patterns", [])
			if p.get("outcome") == "success"
		]
		if not patterns:
			return []
		
		query_text = psi_text or _describe_situation_for_memory(state, events, psi_text=psi_text) if (state is not None or psi_text) else ""
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
				# Break ties by recency (timestamp)
				return (overlap, p.get("timestamp", 0))
			sorted_p = sorted(patterns, key=score, reverse=True)
		else:
			# Fallback: just most recent patterns
			sorted_p = patterns[-k:]
		
		selected = sorted_p[:k]
		
		# Trim each example to the fields the LLM really needs (keep prompt small)
		examples = []
		for p in selected:
			examples.append({
				"timestamp": p.get("timestamp"),
				"detected_failures": p.get("detected_failures", []),
				"situation": p.get("psi_text", ""),
				"outcome": p.get("outcome", "unknown"),
				# minimal raw info in case model wants concrete action
				"action_taken": p.get("action_taken"),
				"low_level_override": p.get("low_level_override"),
			})
		return examples

	def _store_intervention_pattern(self, plan: Plan, human_msg: HumanMessage, state=None, events=None, psi_text: Optional[str] = None) -> None:
		"""Queue intervention pattern; commit only on success."""
		if not self.enable_memory or state is None:
			return
		
		try:
			psi_snapshot = psi_text or ""
			pattern = {
				"timestamp": human_msg.t,
				"human_message": human_msg.text,
				"detected_failures": events or [],
				"action_taken": plan.steps[0] if plan.steps else None,
				"low_level_override": plan.low_level_override,
				"psi_text": psi_snapshot,
				"embedding": self._embed_text(psi_snapshot) if psi_snapshot else None,
				"outcome": "pending",
			}
			self._pending_patterns[human_msg.t] = pattern
			
			print(f"[ICL] Stored intervention pattern (t={human_msg.t}): {psi_snapshot[:60]}...")
			
		except Exception as e:
			# Be conservative: never crash the interpreter because of logging
			print(f"[WARN] _store_intervention_pattern failed: {e}")
			return

	def commit_intervention_pattern(self, timestamp: int) -> None:
		pattern = self._pending_patterns.pop(timestamp, None)
		if not pattern:
			return
		pattern["outcome"] = "success"
		self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
		max_patterns = 32
		if len(self.memory.semantic["intervention_patterns"]) > max_patterns:
			self.memory.semantic["intervention_patterns"] = \
				self.memory.semantic["intervention_patterns"][-max_patterns:]

	def discard_intervention_pattern(self, timestamp: int) -> None:
		pattern = self._pending_patterns.pop(timestamp, None)
		if not pattern:
			return
		pattern["outcome"] = "failure"
		self.memory.semantic.setdefault("intervention_patterns", []).append(pattern)
		max_patterns = 15
		if len(self.memory.semantic["intervention_patterns"]) > max_patterns:
			self.memory.semantic["intervention_patterns"] = \
				self.memory.semantic["intervention_patterns"][-max_patterns:]

	def propose_plan(
		self,
		state_prompt: str,
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
		state=None,  # Add state parameter for intervention recording
		agent_index=None,  # Add agent_index for intervention recording
		events=None,  # Add events parameter for event-based triggers
		psi_text: Optional[str] = None,
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
			"state": state_prompt,
			"history": recent_history[-self.history_horizon:],
			"human_message": {
				"t": human_msg.t,
				"text": human_msg.text,
			},
			"detected_failures": events,  # Add events to payload
			"previous_teammate_descriptor": self._last_teammate_behavior,
		}
		
		# Conditionally include memory_view if memory is enabled
		if self.enable_memory:
			mem_view = self.memory.prompt_view()
			failure_events = {"lack_of_progress", "cyclic_behavior"}
			use_retrieval = (not human_msg.text.strip()) and events and any(e in failure_events for e in events)
			icl_examples = self._select_icl_examples(state=state, events=events, k=3, psi_text=psi_text) if use_retrieval else []
			# Do not duplicate ICL examples inside memory_view
			mem_view["intervention_patterns"] = []
			user_payload["memory"] = mem_view
			# Also expose them explicitly for clarity
			user_payload["icl_examples"] = icl_examples
			
			# Debug: print ICL examples being used
			if icl_examples:
				print(f"[ICL] Using {len(icl_examples)} examples:")
				for i, ex in enumerate(icl_examples):
					situation = ex.get("situation", "")[:50]
					events_str = ", ".join(ex.get("detected_failures", []))
					print(f"  [{i+1}] t={ex.get('timestamp')} events=[{events_str}] situation={situation}...")

		# Get system rules based on enabled features
		system_rules = _get_system_rules(enable_cot=self.enable_cot, enable_memory=self.enable_memory)
		
		# Get JSON schema based on enabled features
		schema = _get_plan_json_schema(enable_cot=self.enable_cot, enable_memory=self.enable_memory)
		
		# Use LLM for all interventions - it will detect low-level commands through reasoning
		try:
			raw = self.llm.respond_json(schema, system_rules, user_payload)
		except Exception:
			return Plan(
				steps=["wait(1)"],
				chain_of_thought="Safe fallback due to verification failure",
				category="general_hint",
				teammate_behavior="",
				low_level_override=None,
			)
		
		# Validate ML actions are in allowed set
		steps = raw.get("steps", [])
		validated_steps = []
		for step in steps:
			if step in ALL_VALID_ML_ACTIONS:
				validated_steps.append(step)
		
		# Always use a valid category (never empty string)
		category = raw.get("category", "general_hint")
		if not category or category.strip() == "":
			category = "general_hint"
		
		# Conditionally set chain_of_thought based on enable_cot
		if self.enable_cot:
			chain_of_thought_raw = str(raw.get("chain_of_thought", ""))
			chain_of_thought = chain_of_thought_raw[:2048] if chain_of_thought_raw else "No chain of thought provided"
		else:
			chain_of_thought = ""  # Empty CoT when disabled
		
		plan = Plan(
			steps=validated_steps,
			chain_of_thought=chain_of_thought,
			category=category,
			teammate_behavior=str(raw.get("teammate_behavior", ""))[:220],
			low_level_override=raw.get("low_level_override"),
		)
		if plan.teammate_behavior:
			self._last_teammate_behavior = plan.teammate_behavior

		# Plan hygiene checks
		# Hard cap actions to <= 3 and ensure at least one safe step
		if len(plan.steps) > 3:
			plan.steps = plan.steps[:3]
		if not plan.steps:
			plan.steps = ["wait(1)"]

		if self.enable_memory:
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
				self._store_intervention_pattern(plan, human_msg, state, events=events, psi_text=psi_text)

		return plan

