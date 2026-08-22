"""
HINT-Agent

Import agents lazily so Co-Gym (or CrowdNav) can load without Overcooked deps.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HINTAgent", "HINTAgentCrowdNav", "HINTAgentCoGym"]


def __getattr__(name: str) -> Any:
    if name == "HINTAgent":
        from .hint_agent_overcooked import HINTAgent

        return HINTAgent
    if name == "HINTAgentCrowdNav":
        from .hint_agent_crowdnav import HINTAgentCrowdNav

        return HINTAgentCrowdNav
    if name == "HINTAgentCoGym":
        from .hint_agent_cogym import HINTAgentCoGym

        return HINTAgentCoGym
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
