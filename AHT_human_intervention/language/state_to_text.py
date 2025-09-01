# state_text.py
# --------------------------------------------
# Deterministic text summaries of an Overcooked state.
# Produces:
#   - CTX tokens: "[CTX] ego:empty mate:onion pots:L(2/3,cook>0) R(ready) dish:avail onions:avail soups_waiting:1"
#   - English:   "You are empty-handed at (3,2), facing N. Your teammate holds onion at (5,1)..."
#
# Drop-in usage:
#   from state_text import build_ctx_text, build_english_text, describe_state
#   ctx = build_ctx_text(env.mdp, state)
#   eng = build_english_text(env.mdp, state)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers over typical Overcooked structures ----------

def _p_orient_str(ori: Tuple[int,int]) -> str:
    return {(0,-1):"N",(0,1):"S",(-1,0):"W",(1,0):"E"}.get(tuple(ori), "?")

def _player_pos(state, idx: int) -> Tuple[int,int]:
    return state.players[idx].position

def _player_hold(state, idx: int) -> Optional[str]:
    p = state.players[idx]
    return p.get_object().name if p.has_object() else None

def _first_or_none(seq):
    return seq[0] if seq else None

def _mdp_positions(mdp, attr: str) -> List[Tuple[int,int]]:
    # attr in: 'get_pot_locations', 'get_onion_dispenser_locations', 'get_dish_dispenser_locations', 'get_serving_locations'
    fn = getattr(mdp, attr, None)
    try:
        xs = fn() if callable(fn) else []
        return list(xs) if xs else []
    except Exception:
        return []

def _soups_from_state(state) -> List[Dict[str,Any]]:
    """Return [{'pos':(x,y), 'n':int, 'cook':int}] for all soup objects we can see."""
    out = []
    objs = getattr(state, "objects", {}) or {}
    for o in objs.values():
        # OvercookedAI typically: name == 'soup', state == (soup_type, num_items, cook_time)
        if getattr(o, "name", None) == "soup":
            try:
                soup_type, n, cook = o.state
                out.append({"pos": o.position, "n": int(n), "cook": int(cook)})
            except Exception:
                # best effort fallback
                out.append({"pos": getattr(o, "position", (0,0)), "n": 0, "cook": 0})
    return out

def _pots_labeled(mdp, soups: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Label pots by left/right when there are exactly 2; otherwise P0,P1,..."""
    pots = _mdp_positions(mdp, "get_pot_locations")
    pots_sorted = sorted(pots, key=lambda xy: (xy[0], xy[1]))  # left->right, then top->bottom
    labels = []
    if len(pots_sorted) == 2:
        labels = ["L","R"]
    else:
        labels = [f"P{i}" for i in range(len(pots_sorted))]

    # map pot tile -> soup state (if any), else n=0,cook=-1 to indicate empty
    def state_at(xy):
        for s in soups:
            if s["pos"] == xy:
                return s
        return {"pos": xy, "n": 0, "cook": -1}  # -1 = no pot soup object found

    return [{"label": lab, **state_at(xy)} for lab,xy in zip(labels, pots_sorted)]

def _ready_count(soups: List[Dict[str,Any]]) -> int:
    return sum(1 for s in soups if s["cook"] == 0 and s["n"] >= 3)

def _counter_counts(state) -> Dict[str,int]:
    """Crude counts of loose items on counters (not in hands)."""
    cnt = {"onion":0, "dish":0, "soup":0}
    objs = getattr(state, "objects", {}) or {}
    for o in objs.values():
        n = getattr(o, "name", "")
        if n in cnt:
            cnt[n] += 1
    return cnt

# ---------- CTX (compact token) description ----------

def build_ctx_text(mdp, state, include_coords: bool = False) -> str:
    """Deterministic affordance summary as a short token string."""
    p0_xy = _player_pos(state, 0); p1_xy = _player_pos(state, 1)
    p0_h  = _player_hold(state, 0) or "empty"
    p1_h  = _player_hold(state, 1) or "empty"
    p0_o  = _p_orient_str(state.players[0].orientation)
    p1_o  = _p_orient_str(state.players[1].orientation)

    soups = _soups_from_state(state)
    pots  = _pots_labeled(mdp, soups)
    dish_src = bool(_mdp_positions(mdp, "get_dish_dispenser_locations"))
    onion_src= bool(_mdp_positions(mdp, "get_onion_dispenser_locations"))
    serve_xy = _first_or_none(_mdp_positions(mdp, "get_serving_locations"))

    counter = _counter_counts(state)
    ready   = _ready_count(soups)

    parts = []
    parts.append(f"[CTX] ego:{p0_h} mate:{p1_h} eori:{p0_o} mori:{p1_o}")
    if include_coords:
        parts.append(f" epos:{p0_xy[0]},{p0_xy[1]} mpos:{p1_xy[0]},{p1_xy[1]}")
    if pots:
        pot_tokens = []
        for p in pots:
            # cook==-1 means "no soup object on pot" → treat like empty idle pot
            status = "empty" if p["cook"] < 0 and p["n"] == 0 else ("ready" if (p["cook"] == 0 and p["n"] >= 3) else (f"{p['n']}/3,cook>0" if p["n"]>0 else "empty"))
            pot_tokens.append(f"{p['label']}({status})")
        parts.append(" pots:" + " ".join(pot_tokens))
    parts.append(f" dish:{'avail' if dish_src else 'none'}")
    parts.append(f" onions:{'avail' if onion_src else 'none'}")
    parts.append(f" soups_ready:{ready}")
    if serve_xy:
        parts.append(f" window:{serve_xy[0]},{serve_xy[1]}")
    if any(counter.values()):
        parts.append(f" counter_items:onion={counter['onion']},dish={counter['dish']},soup={counter['soup']}")
    return " ".join(parts)

# ---------- English (human-readable) description ----------

def build_english_text(mdp, state) -> str:
    p0_xy = _player_pos(state, 0); p1_xy = _player_pos(state, 1)
    p0_h  = _player_hold(state, 0) or "empty"
    p1_h  = _player_hold(state, 1) or "empty"
    p0_o  = _p_orient_str(state.players[0].orientation)
    p1_o  = _p_orient_str(state.players[1].orientation)

    soups = _soups_from_state(state)
    pots  = _pots_labeled(mdp, soups)
    dish_src = bool(_mdp_positions(mdp, "get_dish_dispenser_locations"))
    onion_src= bool(_mdp_positions(mdp, "get_onion_dispenser_locations"))
    counter = _counter_counts(state)

    # Pots sentence
    if not pots:
        pots_s = "No pots detected."
    else:
        parts = []
        for p in pots:
            if p["cook"] == 0 and p["n"] >= 3:
                parts.append(f"{p['label']} pot is READY")
            elif p["n"] > 0 and p["cook"] >= 0:
                parts.append(f"{p['label']} pot: {p['n']}/3 onions, cooking")
            else:
                parts.append(f"{p['label']} pot is empty")
        pots_s = "; ".join(parts) + "."

    items_s = []
    items_s.append("dish dispenser available" if dish_src else "no dish dispenser")
    items_s.append("onion dispenser available" if onion_src else "no onion dispenser")
    if any(counter.values()):
        items_s.append(f"counters: onions={counter['onion']}, dishes={counter['dish']}, soups={counter['soup']}")

    s0 = f"You are at {p0_xy} facing {p0_o}, holding {p0_h}."
    s1 = f"Your teammate is at {p1_xy} facing {p1_o}, holding {p1_h}."
    s2 = pots_s
    s3 = "; ".join(items_s) + "."
    return " ".join([s0, s1, s2, s3])

# ---------- One-call wrapper ----------

def describe_state(mdp, state, kind: str = "both") -> str:
    """
    kind: 'ctx' | 'english' | 'both'
    """
    if kind == "ctx":
        return build_ctx_text(mdp, state, include_coords=False)
    if kind == "english":
        return build_english_text(mdp, state)
    ctx = build_ctx_text(mdp, state, include_coords=False)
    eng = build_english_text(mdp, state)
    return ctx + "\n" + eng
