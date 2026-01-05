#!/usr/bin/env python3
"""
Baseline LLM Intervention System - No CoT, No Memory
Simplified version for baseline performance evaluation.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from openai import OpenAI

try:
	from jsonschema import validate, ValidationError
except ImportError:
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

@dataclass
class Plan:
	"""Simplified Plan without CoT or memory operations."""
	steps: List[str]  # List of ML_action strings
	category: Category
	teammate_behavior: str = ""
	chain_of_thought: str = ""  # Always empty in baseline
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)  # Always empty in baseline
	low_level_override: Optional[str] = None
	intervention_reason: str = ""

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": self.steps,
			"category": self.category,
			"teammate_behavior": self.teammate_behavior,
			"chain_of_thought": "",
			"memory_writes": [],
			"low_level_override": self.low_level_override,
			"intervention_reason": ""
		}

@dataclass
class HumanMessage:
	t: int
	text: str

def _extract_first_json_object(text: str):
	"""Fallback extractor in case model returns extra text."""
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
	# Fallback: find first {...} block
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
	"""Adapter around the OpenAI API with enforced JSON mode."""

	def __init__(self, openai_client: Optional[OpenAI] = None, model: str = "gpt-4o-mini"):
		self.client = openai_client or OpenAI()
		self.model = model

	def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any]) -> Dict[str, Any]:
		"""Request structured JSON output from the LLM."""
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=[
					{"role": "system", "content": system},
					{"role": "user", "content": json.dumps(user, ensure_ascii=False)}
				],
				response_format={"type": "json_object"},
				temperature=0.1,
				max_completion_tokens=1024,  # Reduced for baseline (no CoT needed)
			)

			try:
				text_out = response.choices[0].message.content
			except Exception:
				text_out = getattr(response, "choices", [{}])[0].get("message", {}).get("content", None) or str(response)

			if not text_out:
				raise ValueError(f"No text content in response: {response}")

			# Attempt to parse cleanly
			try:
				obj = json.loads(text_out)
			except json.JSONDecodeError as e:
				obj, err = _extract_first_json_object(text_out)
				if obj is None:
					print(f"[LLM-WARNING] Response appears truncated, using fallback")
					obj = {
						"steps": ["wait(1)"],
						"category": "vague",
						"teammate_behavior": "",
						"chain_of_thought": "",
						"memory_writes": [],
						"low_level_override": None,
						"intervention_reason": ""
					}
				else:
					raise ValueError(f"Could not parse JSON: {err}\nRaw: {text_out}")
			
			# Validate against schema
			try:
				validate(instance=obj, schema=schema)
			except ValidationError as ve:
				print(f"[LLM-DEBUG] Validation error: {ve}")
				# Minimal repair for required fields
				obj.setdefault("steps", ["wait(1)"])
				obj.setdefault("category", "vague")
				obj.setdefault("teammate_behavior", "")
				obj.setdefault("chain_of_thought", "")
				obj.setdefault("memory_writes", [])
				obj.setdefault("low_level_override", None)
				obj.setdefault("intervention_reason", "")
				
				# Ensure category is valid
				if obj.get("category") not in ["policy", "env", "teammate", "vague"]:
					obj["category"] = "vague"
				
				# Try validation again
				validate(instance=obj, schema=schema)
			
			# Force empty CoT and memory_writes for baseline
			obj["chain_of_thought"] = ""
			obj["memory_writes"] = []
			
			return obj

		except Exception as e:
			raise RuntimeError(f"LLMClient.respond_json failed: {e}")

PLAN_JSON_SCHEMA_BASELINE: Dict[str, Any] = {
	"$schema": "http://json-schema.org/draft-07/schema#",
	"type": "object",
	"additionalProperties": False,
	"required": ["steps", "category", "teammate_behavior"],
	"properties": {
		"steps": {
			"type": "array",
			"minItems": 1,
			"items": {
				"type": "string",
				"enum": ALL_VALID_ML_ACTIONS
			}
		},
		"category": {
			"type": "string",
			"enum": ["policy", "env", "teammate", "vague"],
			"default": "vague"
		},
		"teammate_behavior": {
			"type": "string",
			"maxLength": 220,
			"default": ""
		},
		"chain_of_thought": {
			"type": "string",
			"default": ""
		},
		"memory_writes": {
			"type": "array",
			"default": []
		},
		"low_level_override": {
			"type": ["string", "null"],
			"enum": ["move_north", "move_south", "move_east", "move_west", "wait", "interact", "stay", None],
			"default": None
		},
		"intervention_reason": {
			"type": "string",
			"default": ""
		}
	}
}

def _load_system_rules_baseline() -> str:
	"""Load baseline system rules from external file."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	rules_file = os.path.join(script_dir, "advanced_llm_system_rules_baseline.txt")
	try:
		with open(rules_file, 'r', encoding='utf-8') as f:
			return f.read()
	except FileNotFoundError:
		print(f"[WARNING] Baseline system rules file not found at {rules_file}, using fallback")
		return (
			"You control Player0 in Overcooked-AI. Goal: deliver soups quickly and safely.\n"
			"Use ONLY these ML actions: " + ", ".join(ALL_VALID_ML_ACTIONS) + "\n"
			"Return JSON with steps, category, and teammate_behavior. Do not provide chain_of_thought or memory_writes."
		)

SYSTEM_RULES_BASELINE = _load_system_rules_baseline()

class BaselineLLMInterpreter:
	"""
	Simplified LLM interpreter without CoT or memory operations.
	Only generates medium-level action steps.
	"""
	
	def __init__(self, llm: LLMClient, memory=None, history_horizon: int = 8):
		"""
		Initialize baseline interpreter.
		Note: memory parameter is kept for compatibility but not used.
		"""
		self.llm = llm
		self.memory = memory  # Not used, but kept for compatibility
		self.history_horizon = history_horizon
	
	def propose_plan(
		self,
		state_prompt: str,
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
		state=None,
		agent_index=None,
	) -> Plan:
		"""
		Generate plan without CoT or memory operations.
		Only uses state_prompt and recent_history.
		"""
		# Use empty human message for baseline (no interventions)
		human_msg = HumanMessage(t=human_msg.t if hasattr(human_msg, 't') else 0, text="")
		
		user_payload = {
			"state_prompt": state_prompt,
			"recent_history": recent_history[-self.history_horizon:],
			"human_message": {
				"t": human_msg.t,
				"text": "",
			}
		}

		# Call LLM with baseline prompt (no memory_view)
		raw = self.llm.respond_json(PLAN_JSON_SCHEMA_BASELINE, SYSTEM_RULES_BASELINE, user_payload)
		
		# Validate ML actions
		steps = raw.get("steps", [])
		validated_steps = []
		for step in steps:
			if step in ALL_VALID_ML_ACTIONS:
				validated_steps.append(step)
			else:
				print(f"[WARNING] Invalid ML action '{step}', skipping")
		
		if not validated_steps:
			validated_steps = ["wait(1)"]
		
		# Cap to 3 steps
		if len(validated_steps) > 3:
			validated_steps = validated_steps[:3]
		
		category = raw.get("category", "vague")
		if not category or category not in ["policy", "env", "teammate", "vague"]:
			category = "vague"
		
		# Create plan with empty CoT and no memory writes
		plan = Plan(
			steps=validated_steps,
			category=category,
			teammate_behavior=str(raw.get("teammate_behavior", ""))[:220],
			chain_of_thought="",  # Always empty in baseline
			memory_writes=[],  # Always empty in baseline
			low_level_override=None,  # No low-level overrides in baseline
			intervention_reason=""
		)
		
		return plan

