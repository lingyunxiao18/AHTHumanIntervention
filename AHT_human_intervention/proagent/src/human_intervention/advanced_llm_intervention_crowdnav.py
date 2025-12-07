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
				"current_pattern": "",           # "direct_to_goal", "detour", "stationary"
				"coordination_style": "",        # "independent", "coordinated", "conflicting"
				"confidence": 0.0,
			},
			# Human/operator preferences or "house rules"
			"human_prefs": {
				# e.g., "avoid_crowds": True, "prefer_direct_path": True
			},
			# Compact "if…then…" contracts derived from interventions or LLM
			"playbook": [
				# {"if": "crowd_ahead AND clear_alternative", "then": [2, 0, 3], "note": "detour around crowd"}
			],
			# Safety and throttles
			"afford_safety": {"max_wait_on_pass": 2},
			# Choke points or priority zones useful for pathing/avoidance
			"hotspots": [],                      # [{"pos": [x,y], "type": "chokepoint", "risk": "high"}]
			# Things to clarify when vague messages show up
			"open_questions": [],
			# Learned intervention patterns to avoid future human corrections
			"intervention_patterns": [],
			# Learned human intervention cards (LLM Match→Apply)
			"playbook_cards": [],   # list[dict] of short, egocentric cards
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
		WHITELIST = {"navigation_patterns", "obstacle_memory", "teammate_model", "human_prefs",
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
				# Deep merge for nested dicts
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
		
		return {
			"navigation_patterns": {
				"preferred_directions": np.get("preferred_directions", [])[-5:],
				"avoid_directions": np.get("avoid_directions", []),
			},
			"obstacle_memory": sem.get("obstacle_memory", [])[-5:],
			"teammate_model": {
				"since_t": tm.get("since_t"),
				"behavior_description": tm.get("behavior_description"),
				"current_pattern": tm.get("current_pattern", ""),
				"coordination_style": tm.get("coordination_style", ""),
			},
			"human_prefs": sem.get("human_prefs", {}),
			"playbook": sem.get("playbook", [])[-4:],  # small slice
			"afford_safety": sem.get("afford_safety", {}),
			"hotspots": sem.get("hotspots", [])[-2:],
			"summary": self.summarize_recent(),
		}

	def add_intervention_card(self, card: Dict[str, Any]) -> None:
		cards = self.semantic.setdefault("playbook_cards", [])
		cards.append(card)

	def topk_cards(self, query_tokens: List[str], k: int = 5) -> List[Dict[str, Any]]:
		"""Very simple lexical prefilter for candidate cards (keeps code tiny)."""
		cards = self.semantic.get("playbook_cards", [])
		scored = []
		qset = {t.lower() for t in query_tokens}
		for c in cards:
			text = (c.get("title","") + " " + c.get("when_text","")).lower()
			score = sum(1 for t in qset if t in text)
			scored.append((score, c))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [c for s, c in scored[:k] if s > 0]

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
                max_completion_tokens=2048,
            )

            try:
                text_out = response.choices[0].message.content
            except Exception:
                text_out = getattr(response, "choices", [{}])[0].get("message", {}).get("content", None) or str(response)

            if not text_out:
                raise ValueError(f"No text content in response: {response}")

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
            
            # Validate against schema and repair if needed
            try:
                validate(instance=obj, schema=schema)
            except ValidationError as ve:
                print(f"[LLM-DEBUG] Validation error: {ve}")
                print(f"[LLM-DEBUG] Problematic object: {obj}")
                # Minimal repair for required fields
                obj.setdefault("steps", [0])
                obj.setdefault("chain_of_thought", "Emergency repair: executing default action")
                
                category = obj.get("category", "vague")
                if not category or category not in ["policy", "env", "teammate", "vague"]:
                    obj["category"] = "vague"
                
                obj.setdefault("teammate_behavior", "")
                obj.setdefault("memory_writes", [])
                obj.setdefault("low_level_override", None)
                obj.setdefault("intervention_reason", "")
                
                obj["chain_of_thought"] = obj.get("chain_of_thought") or obj.get("brief_rationale","")
                
                validate(instance=obj, schema=schema)
            
            obj["chain_of_thought"] = obj.get("chain_of_thought") or obj.get("brief_rationale","")
            
            return obj

        except Exception as e:
            raise RuntimeError(f"LLMClient.respond_json failed: {e}")

    def match_apply(self, system: str, user: str) -> Dict[str, Any]:
        """Run the Match→Apply card matching prompt and return parsed JSON."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rules_file = os.path.join(script_dir, "advanced_llm_system_rules_crowdnav.txt")
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                full_rules = f.read()
            match_section_start = full_rules.find("MATCH→APPLY CONTROLLER (CARD MATCHING)")
            if match_section_start >= 0:
                system_text = full_rules[match_section_start:]
            else:
                system_text = system
        except Exception:
            system_text = system
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=512,
            )
            text_out = response.choices[0].message.content
            obj = json.loads(text_out)
            return obj
        except Exception as e:
            print(f"[LLM-WARNING] Match→Apply failed: {e}")
            return {
                "matched_card_id": None,
                "similarity_reason": "Error in matching",
                "apply": False,
                "low_level_override": None,
                "keep_medium_plan": True,
                "cooldown": 0
            }


# JSON Schema for Plan output
PLAN_JSON_SCHEMA = {
	"type": "object",
	"properties": {
		"steps": {
			"type": "array",
			"items": {"type": "integer", "minimum": 0, "maximum": 7},
			"minItems": 1,
			"maxItems": 3,
		},
		"chain_of_thought": {"type": "string"},
		"category": {"type": "string", "enum": ["policy", "env", "teammate", "vague", ""]},
		"teammate_behavior": {"type": "string"},
		"memory_writes": {
			"type": "array",
			"items": {"type": "object"},
		},
		"low_level_override": {
			"type": ["integer", "null"],
			"enum": [0, 1, 2, 3, 4, 5, 6, 7, None],
		},
		"intervention_reason": {"type": "string"},
	},
	"required": ["steps", "chain_of_thought", "category", "teammate_behavior", "memory_writes", "low_level_override", "intervention_reason"],
	"additionalProperties": False,
}


class AdvancedLLMInterpreter:
	"""
	CoT + Memory interpreter for human interventions in CrowdNav-AHT.
	Uses LLM to generate structured plans from text observations and human messages.
	"""
	
	def __init__(self, llm_client: LLMClient, memory: AgentMemory, history_horizon: int = 8):
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
		
		# Call LLM
		try:
			raw_plan = self.llm.respond_json(PLAN_JSON_SCHEMA, self.system_rules, user_prompt)
		except Exception as e:
			print(f"[INTERPRETER-ERROR] LLM call failed: {e}")
			# Fallback plan
			raw_plan = {
				"steps": [0],
				"chain_of_thought": f"Error in interpretation: {e}",
				"category": "vague",
				"teammate_behavior": "",
				"memory_writes": [],
				"low_level_override": None,
				"intervention_reason": "",
			}
		
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

