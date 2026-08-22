"""
Action-only peer teammate personas for the AHT × HINT Co-Gym paper arc.

These agents implement the CoGym AgentNode contract (start / get_action / end)
and **never** emit SEND_TEAMMATE_MESSAGE. Critique / privileged prefs belong to
the outside supervisor, not the teammate.

Personas
--------
- idle: almost always WAIT_TEAMMATE_CONTINUE
- complementary_searcher: fill missing travel searches from the task query
- greedy_editor: early EDITOR_UPDATE with a thin plan (often hurts quality)
- follower: delayed complementary searches (starts later, lower rate)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PERSONA_NAMES = ("idle", "complementary_searcher", "greedy_editor", "follower")

WAIT = "WAIT_TEAMMATE_CONTINUE()"

_TRIP_RE = re.compile(
    r"(?:from|From)\s+(?P<org>.+?)\s+to\s+(?P<dest>.+?)"
    r"(?:\s+starting on\s+(?P<date>\d{4}-\d{2}-\d{2})|(?=\.)|$)",
    re.IGNORECASE,
)
_DAYS_RE = re.compile(r"(\d+)\s*-?\s*day", re.IGNORECASE)


def _parse_trip(task_description: str) -> Dict[str, str]:
    org, dest, date = "Salt Lake City", "Burbank", "2022-03-12"
    days = 3
    m = _TRIP_RE.search(task_description or "")
    if m:
        org = m.group("org").strip().rstrip(".,")
        dest = m.group("dest").strip().rstrip(".,")
        # If lookahead stopped early, dest may include trailing words — trim at "starting"
        if " starting" in dest.lower():
            dest = re.split(r"\s+starting\b", dest, flags=re.I)[0].strip()
        if m.groupdict().get("date") and m.group("date"):
            date = m.group("date")
    else:
        # Fallback: "from X to Y starting on DATE"
        m2 = re.search(
            r"from\s+(.+?)\s+to\s+(.+?)\s+starting on\s+(\d{4}-\d{2}-\d{2})",
            task_description or "",
            flags=re.I,
        )
        if m2:
            org, dest, date = m2.group(1).strip(), m2.group(2).strip(), m2.group(3)
    dm = _DAYS_RE.search(task_description or "")
    if dm:
        days = int(dm.group(1))
    try:
        y, mo, d = [int(x) for x in date.split("-")]
        ret_d = min(d + max(days - 1, 1), 28)
        ret = f"{y:04d}-{mo:02d}-{ret_d:02d}"
    except Exception:
        ret = date
    return {"org": org, "dest": dest, "date": date, "return_date": ret, "days": str(days)}


def _editor_text(observation: Dict[str, Any]) -> str:
    if not isinstance(observation, dict):
        return ""
    return str(observation.get("travel_plan_editor") or "").strip()


def _has_action_pattern(action_space: List[Any], prefix: str) -> bool:
    for act in action_space or []:
        if isinstance(act, dict):
            name = str(act.get("human_readable_name") or "")
            ident = str(act.get("machine_readable_identifier") or "")
            pattern = str(act.get("regex_pattern") or act.get("pattern") or "")
            blob = f"{name} {ident} {pattern}"
        else:
            blob = str(act)
        if prefix in blob:
            return True
    return True  # if unknown space, assume travel tools exist


class PeerTeammateAgent:
    """Deterministic action-only teammate. Never chats."""

    def __init__(self, persona: str = "complementary_searcher", wait_every: int = 1):
        if persona not in PERSONA_NAMES:
            raise ValueError(f"Unknown persona {persona!r}; choose from {PERSONA_NAMES}")
        self.persona = persona
        self.wait_every = max(1, wait_every)
        self.name: Optional[str] = None
        self.task_description: str = ""
        self.action_space: List[Any] = []
        self.trip: Dict[str, str] = {}
        self._step = 0
        self._done_searches: List[str] = []
        self._action_log: List[Dict[str, Any]] = []

    def start(
        self,
        name: str,
        team_members: List[str],
        task_description: str,
        action_space: List[Any],
        example_question: str = "",
        example_trajectory: Any = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.task_description = task_description or ""
        self.action_space = action_space or []
        self.trip = _parse_trip(self.task_description)
        self._step = 0
        self._done_searches = []
        logger.info(
            "PeerTeammate[%s] started as %r trip=%s",
            self.persona,
            name,
            self.trip,
        )

    def get_action(self, observation: dict, chat_history: list) -> str:
        self._step += 1
        action = self._choose(observation)
        # Hard ban on teammate chat / finish spam (ego owns FINISH unless greedy).
        if action.startswith("SEND_TEAMMATE_MESSAGE"):
            action = WAIT
        self._action_log.append({"t": self._step, "persona": self.persona, "action": action})
        logger.info("PeerTeammate[%s] t=%d -> %s", self.persona, self._step, action[:120])
        return action

    def end(self, result_dir: str) -> None:
        try:
            out_dir = os.path.join(result_dir, self.name or "teammate")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "info.json"), "w") as f:
                json.dump(
                    {
                        "agent": "PeerTeammateAgent",
                        "persona": self.persona,
                        "total_steps": self._step,
                        "trip": self.trip,
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(out_dir, "action_log.jsonl"), "w") as f:
                for entry in self._action_log:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("PeerTeammateAgent.end failed: %s", e)

    # ------------------------------------------------------------------
    def _choose(self, observation: dict) -> str:
        if self.persona == "idle":
            return WAIT

        if self.persona == "greedy_editor":
            return self._greedy_editor(observation)

        if self.persona == "follower":
            # Start later and act every other tick.
            if self._step < 3 or (self._step % 2 == 0):
                return WAIT
            return self._next_search(observation)

        # complementary_searcher
        if self._step % self.wait_every == 0 and self._step > 1:
            # slight yield so ego can interleave
            if self._step % 5 == 0:
                return WAIT
        return self._next_search(observation)

    def _search_plan(self) -> List[Tuple[str, str]]:
        t = self.trip
        return [
            ("flight_out", f"FLIGHT_SEARCH(origin={t['org']}, destination={t['dest']}, date={t['date']})"),
            ("flight_ret", f"FLIGHT_SEARCH(origin={t['dest']}, destination={t['org']}, date={t['return_date']})"),
            ("hotel", f"ACCOMMODATION_SEARCH(city={t['dest']})"),
            ("food", f"RESTAURANT_SEARCH(city={t['dest']})"),
            ("attr", f"ATTRACTION_SEARCH(city={t['dest']})"),
        ]

    def _next_search(self, observation: dict) -> str:
        editor = _editor_text(observation)
        if editor and len(editor) > 80:
            # Ego (or someone) already drafting — idle rather than fight the editor.
            return WAIT
        for key, action in self._search_plan():
            if key not in self._done_searches:
                if key.startswith("flight") and not _has_action_pattern(self.action_space, "FLIGHT_SEARCH"):
                    continue
                self._done_searches.append(key)
                return action
        return WAIT

    def _greedy_editor(self, observation: dict) -> str:
        editor = _editor_text(observation)
        # After a couple of waits / one search, overwrite the plan aggressively.
        if self._step <= 2:
            if "hotel" not in self._done_searches:
                self._done_searches.append("hotel")
                return f"ACCOMMODATION_SEARCH(city={self.trip['dest']})"
            return WAIT
        if editor and "Travel Plan:" in editor and self._step > 6:
            return WAIT
        t = self.trip
        thin = (
            f"Travel Plan:\n"
            f"Day 1:\nCurrent City: from {t['org']} to {t['dest']}\n"
            f"Transportation: Flight Number: F0000001, from {t['org']} to {t['dest']}, "
            f"Departure Time: 09:00, Arrival Time: 11:00\n"
            f"Breakfast: -, {t['dest']}\nAttraction: -, {t['dest']}\n"
            f"Lunch: -, {t['dest']}\nDinner: -, {t['dest']}\n"
            f"Accommodation: Placeholder Hotel, {t['dest']}\n\n"
            f"Day 2:\nCurrent City: {t['dest']}\nTransportation: -\n"
            f"Breakfast: -, {t['dest']}\nAttraction: -, {t['dest']}\n"
            f"Lunch: -, {t['dest']}\nDinner: -, {t['dest']}\n"
            f"Accommodation: Placeholder Hotel, {t['dest']}\n\n"
            f"Day 3:\nCurrent City: from {t['dest']} to {t['org']}\n"
            f"Transportation: Flight Number: F0000002, from {t['dest']} to {t['org']}, "
            f"Departure Time: 18:00, Arrival Time: 20:00\n"
            f"Breakfast: -, {t['dest']}\nAttraction: -\nLunch: -\nDinner: -\nAccommodation: -"
        )
        return f"EDITOR_UPDATE(text={thin})"


def make_peer_teammate(persona: str) -> PeerTeammateAgent:
    return PeerTeammateAgent(persona=persona)
