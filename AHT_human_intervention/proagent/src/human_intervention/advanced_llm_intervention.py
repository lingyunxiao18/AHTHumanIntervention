#!/usr/bin/env python3
import os
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Literal, Tuple

OPENAI_API_KEY="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"

# Minimal dependency: use OpenAI if available; otherwise operate in rule-based fallback
try:
	from openai import OpenAI  # SDK >=1.0 (Responses API)
	_HAS_OPENAI = True
except Exception:
	OpenAI = None  # type: ignore
	_HAS_OPENAI = False


PROAGENT_ALLOWED_ML_ACTIONS = {
	"pickup_onion",
	"pickup_dish", 
	"put_onion_in_pot",
	"fill_dish_with_soup",
	"deliver_soup",
	"place_obj_on_counter",
	"wait"
}

# ---------------------------------------------------------------------
# CoT + Memory Interpreter for Human Interventions in Overcooked-AI
# Inputs: observable state, human message, recent K-step history
# Outputs: structured Plan JSON, one-sentence rationale, message category
# ---------------------------------------------------------------------

Category = Literal["policy", "env", "teammate", "vague"]

@dataclass
class MacroStep:
	"""
	A single medium-level action step.
	- macro: ML action string (e.g., pickup_onion, put_onion_in_pot, deliver_soup, wait(3))
	- args: free-form dict; keep keys simple (e.g., 'pot_id', 'duration')
	- guard: OPTIONAL string guard (simple boolean in your runtime; feel free to ignore if not used)
	- timeout: OPTIONAL max primitive steps allowed before we bail/replan
	"""
	macro: str
	args: Optional[Dict[str, Any]] = None
	guard: Optional[str] = None
	timeout: Optional[int] = None

@dataclass
class Plan:
	steps: List[MacroStep]
	confidence: float
	rationale_public: str                 # <= 1 sentence, no CoT
	category: Category                    # model's classification of the human msg
	memory_writes: List[Dict[str, Any]] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"steps": [asdict(s) for s in self.steps],
			"confidence": self.confidence,
			"rationale_public": self.rationale_public,
			"category": self.category,
			"memory_writes": self.memory_writes,
		}

class AgentMemory:
	"""
	Lightweight persistent memory (kept in RAM by default).
	- semantic: long-lived facts & rules
	- episodic: recent events for summarization
	"""
	def __init__(self, episodic_cap: int = 120):
		self.semantic: Dict[str, Any] = {
			"layout_facts": {},
			"teammate_model": {
				"current_role": "unknown",
				"role_confidence": 0.0,
				"since_t": None,
				"behavior_notes": "",
			},
			"human_prefs": {},
			"playbook": [],
			"afford_safety": {"max_wait_on_pass": 2},
			"hotspots": [],
			"open_questions": [],
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
			elif e.get("type") == "human_msg":
				notes.append(f"human[{e.get('kind')}]")
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
		WHITELIST = {"layout_facts", "teammate_model", "human_prefs",
					 "playbook", "afford_safety", "hotspots", "open_questions"}
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
				self.semantic[k].update(v)
			else:
				self.semantic[k] = v

	def prompt_view(self) -> Dict[str, Any]:
		"""Compact view for the model prompt."""
		sem = self.semantic
		return {
			"layout_facts": sem.get("layout_facts", {}),
			"teammate_model": {
				k: sem["teammate_model"].get(k)
				for k in ("current_role", "role_confidence", "since_t", "behavior_notes")
			},
			"human_prefs": sem.get("human_prefs", {}),
			"playbook": sem.get("playbook", [])[-6:],
			"afford_safety": sem.get("afford_safety", {}),
			"hotspots": sem.get("hotspots", [])[-3:],
			"summary": self.summarize_recent(),
		}

@dataclass
class HumanMessage:
	t: int
	text: str
	category_hint: Optional[Category] = None

def classify_message_heuristic(text: str) -> Category:
	"""
	Optional lightweight pre-classification;
	the model will still output its own category.
	"""
	low = text.lower()
	if any(k in low for k in ["do ", "don't ", "wait", "go ", "get ", "serve", "deliver", "take", "put "]):
		return "policy"
	if any(k in low for k in ["ready", "cooked", "timer", "on the right", "left pot", "there is", "available"]):
		return "env"
	if any(k in low for k in ["teammate", "they are", "she is", "he is", "your partner"]):
		return "teammate"
	return "vague"

class LLMClient:
	"""
	Adapter around your OpenAI Responses API (or similar).
	You must implement `respond_json(schema, system, user)` to:
	- run a short, hidden CoT,
	- enforce JSON-only output matching `schema`,
	- return parsed Python dict.
	"""
	def __init__(self, openai_client):
		self.client = openai_client

	def respond_json(self, schema: Dict[str, Any], system: str, user: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Return JSON using available OpenAI API path.
		Tries Responses API first; falls back to Chat Completions with json_object.
		"""
		# Try Responses API
		try:
			print("[LLMClient] Using Responses API with json_schema")
			r = self.client.responses.create(
				model="gpt-4.1-nano",
				input=[
					{"role": "system", "content": system},
					{"role": "user", "content": json.dumps(user)}
				],
				response_format={
					"type": "json_schema",
					"json_schema": {"name": "Plan", "schema": schema}
				},
			)
			content = r.output[0].content[0].text
			return json.loads(content)
		except Exception as e:
			print(f"[LLMClient] Responses API failed: {e}")

		# Fallback to Chat Completions JSON
		try:
			print("[LLMClient] Using Chat Completions with json_object")
			res = self.client.chat.completions.create(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": system},
					{"role": "user", "content": json.dumps(user)}
				],
				temperature=0.1,
				max_tokens=500,
				response_format={"type": "json_object"},
			)
			text = res.choices[0].message.content.strip()
			print(f"[LLMClient] Chat JSON text head: {text[:120]}")
			return json.loads(text)
		except Exception as e:
			print(f"[LLMClient] Chat Completions failed: {e}")

		# Last resort: plain chat, try to extract JSON
		print("[LLMClient] Using Chat Completions plain + JSON extraction")
		res = self.client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": system + "\nReturn ONLY JSON."},
				{"role": "user", "content": json.dumps(user)}
			],
			temperature=0.1,
			max_tokens=500,
		)
		text = res.choices[0].message.content.strip()
		if "```json" in text:
			start = text.find("```json") + 7
			end = text.find("```", start)
			text = text[start:end].strip()
		elif "{" in text and "}" in text:
			start = text.find("{")
			end = text.rfind("}") + 1
			text = text[start:end]
		print(f"[LLMClient] Extracted JSON head: {text[:120]}")
		return json.loads(text)

PLAN_JSON_SCHEMA: Dict[str, Any] = {
	"$schema": "http://json-schema.org/draft-07/schema#",
	"type": "object",
	"required": ["steps", "confidence", "rationale_public", "category"],
	"properties": {
		"steps": {
			"type": "array",
			"minItems": 1,
			"items": {
				"type": "object",
				"required": ["macro"],
				"properties": {
					"macro": {"type": "string"},
					"args": {"type": "object"},
					"guard": {"type": "string"},
					"timeout": {"type": "integer", "minimum": 0}
				}
			}
		},
		"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"rationale_public": {"type": "string", "maxLength": 180},
		"category": {"type": "string", "enum": ["policy", "env", "teammate", "vague"]},
		"memory_writes": {"type": "array", "items": {"type": "object"}}
	}
}

SYSTEM_RULES = (
	"You control Player0 in Overcooked-AI. Goal: deliver soups quickly and safely.\n"
	"INPUTS you receive: (a) state_snapshot, (b) recent_history (few lines), (c) memory_view, (d) human_message.\n"
	"Do a SHORT, PRIVATE chain-of-thought (<= 6 lines) to parse the human intent and choose a plan.\n"
	"OUTPUT ONLY one JSON object matching the provided schema.\n"
	"Constraints:\n"
	" - 1–3 ML action steps max.\n"
	" - Use ONLY these ML actions: pickup_onion, pickup_dish, put_onion_in_pot, fill_dish_with_soup, deliver_soup, place_obj_on_counter, wait(N)\n"
	" - Add timeouts when appropriate (e.g., 6–20 primitive steps).\n"
	" - Keep rationale_public to ONE sentence (no hidden reasoning).\n"
	" - Classify the human message into one category: policy | env | teammate | vague.\n"
	" - Prefer conservative, feasible actions if uncertain (e.g., wait(1)).\n"
	" - Respect simple safety heuristics from memory_view.afford_safety.\n"
	"Do not include your thoughts; return JSON only."
)

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
		state_snapshot: Dict[str, Any],
		human_msg: HumanMessage,
		recent_history: List[Dict[str, Any]],
	) -> Plan:
		if human_msg.category_hint is None:
			human_msg.category_hint = classify_message_heuristic(human_msg.text)

		user_payload = {
			"state_snapshot": self._compact_state(state_snapshot),
			"recent_history": recent_history[-self.history_horizon:],
			"memory_view": self.memory.prompt_view(),
			"human_message": {
				"t": human_msg.t,
				"text": human_msg.text,
				"category_hint": human_msg.category_hint,
			}
		}

		raw = self.llm.respond_json(PLAN_JSON_SCHEMA, SYSTEM_RULES, user_payload)

		steps = [MacroStep(**s) for s in raw.get("steps", [])]
		plan = Plan(
			steps=steps,
			confidence=float(raw.get("confidence", 0.5)),
			rationale_public=str(raw.get("rationale_public", ""))[:180],
			category=raw.get("category", "vague"),
			memory_writes=raw.get("memory_writes", []),
		)

		safe_patch = self._gate_memory_writes(plan.memory_writes)
		if safe_patch:
			self.memory.upsert_semantic(safe_patch)
		self.memory.write_events([
			{"t": human_msg.t, "type": "human_msg", "kind": plan.category, "text": human_msg.text},
			{"t": human_msg.t, "type": "plan", "steps": [s.macro for s in plan.steps], "conf": plan.confidence}
		])

		return plan

	def _gate_memory_writes(self, writes: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""
		Convert a list of 'memory_writes' entries into a single semantic patch.
		Keep it conservative: allow updates only in whitelisted regions and small payloads.
		Expected patterns (examples):
		  {"teammate_model": {"current_role":"plate_runner","role_confidence":0.7,"since_t":123}}
		  {"human_prefs": {"prefer_left_pot": true}}
		  {"playbook": [{"if":"...","then":[...],"note":"..."}]}
		"""
		patch: Dict[str, Any] = {}
		if not writes:
			return patch
		for w in writes:
			for k, v in w.items():
				if k not in {"teammate_model", "human_prefs", "playbook", "afford_safety", "hotspots", "open_questions"}:
					continue
				if isinstance(v, (dict, list)) and len(json.dumps(v)) > 2000:
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

	def _compact_state(self, state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Trim the observable state to essentials to keep tokens small.
		Expect keys like:
		 - players -> [{"pos":[x,y],"orient":"N|S|E|W","holding":"onion|dish|soup|nothing"}, ...]
		 - pots -> [{"id":1,"need":1,"cooking":false,"ready_in":null}, ...]
		 - counters -> possibly large; keep only non-empty locations
		 - layout_name, timestep, etc.
		"""
		s = dict(state_snapshot)
		counters = s.get("counters", {})
		if isinstance(counters, dict) and "objects" in counters:
			objs = counters["objects"]
			if isinstance(objs, list) and len(objs) > 12:
				s["counters"]["objects"] = objs[:12]
		return s