"""Redis channel helpers for outside-supervisor interventions into HINT-Agent.

The CoGym session (HINT-Agent + simulated teammate) keeps running. A separate
supervisor process publishes free-text interventions; the HINT-Agent launcher
subscribes and calls ``apply_human_intervention`` without pausing the env.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Channel: env_{session}/hint_agent/supervisor_intervention
CHANNEL_SUFFIX = "supervisor_intervention"


def intervention_channel(env_uuid: str, agent_node_name: str = "hint_agent") -> str:
    return f"{env_uuid}/{agent_node_name}/{CHANNEL_SUFFIX}"


def publish_intervention(
    redis_url: str,
    env_uuid: str,
    text: str,
    agent_node_name: str = "hint_agent",
) -> None:
    """Publish one supervisor intervention (non-blocking for the session)."""
    import redis

    text = (text or "").strip()
    if not text:
        return
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    payload = json.dumps({"text": text, "source": "outside_supervisor"})
    channel = intervention_channel(env_uuid, agent_node_name)
    r.publish(channel, payload)
    logger.info("Published supervisor intervention on %s: %r", channel, text[:120])


def start_intervention_subscriber(
    redis_url: str,
    env_uuid: str,
    on_message: Callable[[str], None],
    agent_node_name: str = "hint_agent",
) -> threading.Thread:
    """
    Background thread: subscribe to supervisor interventions and invoke callback.

    Does not pause CoGym. Interventions sit in HINT-Agent's inbox until the next
    ``get_action`` call, then flow through CoT+Memory like Overcooked.
    """
    import redis

    channel = intervention_channel(env_uuid, agent_node_name)

    def _loop() -> None:
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = r.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
        logger.info("HINT-Agent listening for supervisor interventions on %s", channel)
        for msg in pubsub.listen():
            if msg is None or msg.get("type") != "message":
                continue
            data = msg.get("data")
            text: Optional[str] = None
            try:
                if isinstance(data, str):
                    parsed = json.loads(data)
                    text = parsed.get("text") if isinstance(parsed, dict) else data
                else:
                    text = str(data)
            except json.JSONDecodeError:
                text = str(data)
            if text and text.strip():
                try:
                    on_message(text.strip())
                except Exception as e:
                    logger.warning("Failed to apply supervisor intervention: %s", e)

    t = threading.Thread(target=_loop, name="hint-supervisor-sub", daemon=True)
    t.start()
    return t
