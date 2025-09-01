# state_to_text.py
# Deterministic, layout-agnostic text descriptions of an Overcooked state.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# ---------- tiny safe-call helpers ----------
def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def _ori_str(ori: Tuple[int,int]) -> str:
    return {(0,-1):"N",(0,1):"S",(-1,0):"W",(1,0):"E"}.get(tuple(ori), "?")

# ---------- map feature extractors ----------
def _terrain_positions(mdp, tile_char: str) -> List[Tuple[int,int]]:
    """Scan mdp.terrain_mtx for a given tile char (e.g., 'X' counters)."""
    mtx = getattr(mdp, "terrain_mtx", None)
    out = []
    if not mtx:
        return out
    H, W = len(mtx), len(mtx[0])
    for y in range(H):
        for x in range(W):
            if mtx[y][x] == tile_char:
                out.append((x, y))
    return out

def _pots(mdp) -> List[Tuple[int,int]]:
    pots = _safe(getattr(mdp, "get_pot_locations", lambda: []) , [])
    return list(pots or [])

def _onion_disp(mdp) -> List[Tuple[int,int]]:
    xs = _safe(getattr(mdp, "get_onion_dispenser_locations", lambda: []), [])
    return list(xs or [])

def _dish_disp(mdp) -> List[Tuple[int,int]]:
    xs = _safe(getattr(mdp, "get_dish_dispenser_locations", lambda: []), [])
    return list(xs or [])

def _serving(mdp) -> List[Tuple[int,int]]:
    xs = _safe(getattr(mdp, "get_serving_locations", lambda: []), [])
    return list(xs or [])

def _counters(mdp) -> List[Tuple[int,int]]:
    # Overcooked uses 'X' for counters in terrain_mtx
    return _terrain_positions(mdp, 'X')

# ---------- object/state feature extractors ----------
def _players(state):
    return getattr(state, "players", [])

def _pos_held(state, idx: int) -> Tuple[Tuple[int,int], Optional[str]]:
    p = state.players[idx]
    pos = p.position
    held = p.get_object().name if p.has_object() else None
    return pos, held

def _soups(state) -> List[Dict[str,Any]]:
    """Return list of soup objects with pos, n(items), cook time."""
    out = []
    objs = getattr(state, "objects", {}) or {}
    for o in objs.values():
        if getattr(o, "name", None) == "soup":
            try:
                soup_type, n, cook = o.state
                out.append({"pos": o.position, "type": soup_type, "n": int(n), "cook": int(cook)})
            except Exception:
                out.append({"pos": getattr(o, "position", (0,0)), "type": "onion", "n": 0, "cook": 0})
    return out

def _loose_items(state, name: str) -> List[Tuple[int,int]]:
    """Positions of items placed on tiles (not in hands)."""
    out = []
    objs = getattr(state, "objects", {}) or {}
    for o in objs.values():
        if getattr(o, "name", None) == name:
            xy = getattr(o, "position", None)
            if xy is not None:
                out.append(xy)
    return out

def _soups_not_in_pots(state, pot_tiles: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    pots_set = set(pot_tiles)
    return [s["pos"] for s in _soups(state) if s["pos"] not in pots_set]

def _pots_with_state(mdp, state) -> List[Dict[str,Any]]:
    """For every pot tile, report items and cook time (0 if ready, -1 if no soup object)."""
    pots = _pots(mdp)
    soups = {tuple(s["pos"]): s for s in _soups(state)}
    out = []
    for xy in pots:
        s = soups.get(tuple(xy))
        if s is None:
            out.append({"pos": xy, "type": "onion", "onions": 0, "cook": -1})
        else:
            out.append({"pos": xy, "type": s["type"], "onions": int(s["n"]), "cook": int(s["cook"])})
    return out



# ---------- TEXT BUILDERS ----------
def build_ctx_text(mdp, state, include_coords: bool = False) -> str:
    """
    Compact token string meant to prepend to the command. Includes everything important,
    but summarized tersely.
    """
    p0_xy, p0_hold = _pos_held(state, 0)
    p1_xy, p1_hold = _pos_held(state, 1)
    p0_o = _ori_str(state.players[0].orientation)
    p1_o = _ori_str(state.players[1].orientation)

    pots_state = _pots_with_state(mdp, state)
    onion_dsps = _onion_disp(mdp)
    dish_dsps  = _dish_disp(mdp)
    serv_locs  = _serving(mdp)
    counters   = _counters(mdp)

    loose_onions = _loose_items(state, "onion")
    loose_dishes = _loose_items(state, "dish")
    soups_not_pot= _soups_not_in_pots(state, [p["pos"] for p in pots_state])

    parts = []
    parts.append(f"[CTX] ego:{p0_hold or 'empty'}@{p0_o} mate:{p1_hold or 'empty'}@{p1_o}")
    if include_coords:
        parts.append(f" epos:{p0_xy[0]},{p0_xy[1]} mpos:{p1_xy[0]},{p1_xy[1]}")
    # Pots summary
    if pots_state:
        ps = []
        for p in sorted(pots_state, key=lambda d: (d["pos"][0], d["pos"][1])):
            status = "ready" if (p["cook"] == 0 and p["onions"] >= 3) else \
                     ("empty" if p["onions"] == 0 and p["cook"] < 0 else f"{p['onions']}/3,cook={p['cook']}")
            ps.append(f"{p['pos'][0]},{p['pos'][1]}({status})")
        parts.append(" pots:" + " ".join(ps))
    # Map inventories
    parts.append(" onion_dsps:" + " ".join(f"{x},{y}" for x,y in onion_dsps) if onion_dsps else " onion_dsps:none")
    parts.append(" dish_dsps:"  + " ".join(f"{x},{y}" for x,y in dish_dsps)  if dish_dsps  else " dish_dsps:none")
    parts.append(" serving:"    + " ".join(f"{x},{y}" for x,y in serv_locs)  if serv_locs  else " serving:none")
    parts.append(" counters:"   + " ".join(f"{x},{y}" for x,y in counters)   if counters   else " counters:none")
    # Loose objects
    parts.append(" onions:" + " ".join(f"{x},{y}" for x,y in loose_onions) if loose_onions else " onions:none")
    parts.append(" dishes:" + " ".join(f"{x},{y}" for x,y in loose_dishes) if loose_dishes else " dishes:none")
    parts.append(" soups:"  + " ".join(f"{x},{y}" for x,y in soups_not_pot) if soups_not_pot else " soups:none")

    return " ".join(parts)

def build_full_text(mdp, state) -> str:
    """
    Verbose, line-oriented description that enumerates every requested element.
    """
    p0_xy, p0_hold = _pos_held(state, 0)
    p1_xy, p1_hold = _pos_held(state, 1)
    p0_o = _ori_str(state.players[0].orientation)
    p1_o = _ori_str(state.players[1].orientation)

    # Map features
    counters   = sorted(_counters(mdp))
    pots       = sorted(_pots(mdp))
    onion_dsps = sorted(_onion_disp(mdp))
    dish_dsps  = sorted(_dish_disp(mdp))
    serves     = sorted(_serving(mdp))

    # Object/state features
    pots_state   = sorted(_pots_with_state(mdp, state), key=lambda d: (d["pos"][0], d["pos"][1]))
    soups_notpot = sorted(_soups_not_in_pots(state, pots))
    onions_loose = sorted(_loose_items(state, "onion"))
    dishes_loose = sorted(_loose_items(state, "dish"))

    lines = []
    lines.append("[PLAYERS]")
    lines.append(f"ego: pos=({p0_xy[0]},{p0_xy[1]}), ori={p0_o}, holding={p0_hold or 'empty'}")
    lines.append(f"mate: pos=({p1_xy[0]},{p1_xy[1]}), ori={p1_o}, holding={p1_hold or 'empty'}")
    lines.append("")
    lines.append("[MAP]")
    lines.append("counters: " + (", ".join(f"({x},{y})" for x,y in counters) if counters else "none"))
    lines.append("pots:     " + (", ".join(f"({x},{y})" for x,y in pots) if pots else "none"))
    lines.append("onion_dispensers: " + (", ".join(f"({x},{y})" for x,y in onion_dsps) if onion_dsps else "none"))
    lines.append("dish_dispensers:  " + (", ".join(f"({x},{y})" for x,y in dish_dsps) if dish_dsps else "none"))
    lines.append("serving_locations: " + (", ".join(f"({x},{y})" for x,y in serves) if serves else "none"))
    lines.append("")
    lines.append("[OBJECT_STATE]")
    if pots_state:
        for p in pots_state:
            status = "ready" if (p["cook"] == 0 and p["onions"] >= 3) else \
                     ("empty" if p["onions"] == 0 and p["cook"] < 0 else f"{p['onions']}/3, cook={p['cook']}")
            lines.append(f"pot@({p['pos'][0]},{p['pos'][1]}): onions={p['onions']}, cook={p['cook']} ({status})")
    else:
        lines.append("no pots detected")
    lines.append("soups_not_in_pots: " + (", ".join(f"({x},{y})" for x,y in soups_notpot) if soups_notpot else "none"))
    lines.append("loose_dishes:      " + (", ".join(f"({x},{y})" for x,y in dishes_loose) if dishes_loose else "none"))
    lines.append("loose_onions:      " + (", ".join(f"({x},{y})" for x,y in onions_loose) if onions_loose else "none"))
    return "\n".join(lines)

def describe_state(mdp, state, mode: str = "both") -> str:
    """
    mode: 'ctx' | 'full' | 'both'
    """
    if mode == "ctx":
        return build_ctx_text(mdp, state, include_coords=False)
    if mode == "full":
        return build_full_text(mdp, state)
    return build_ctx_text(mdp, state, include_coords=False) + "\n" + build_full_text(mdp, state)
