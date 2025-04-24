from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import *
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.agents.agent import Agent

from enum import Enum



"""
Writter by Johnny Liu
Modified by Zhihan (Minor edits)

"""

# mdp = OvercookedGridworld.from_layout_name(
#         "counter_circuit_o_1order"
# )
mdp = OvercookedGridworld.from_layout_name(
        "random3"
)

class Interests(Enum):
    POT = 1
    PLATES = 2
    SERVE = 3
    ONION = 4

class RotateAgent(Agent):
    def __init__(self, direction=True): # direction: the rotation direction of the agent
        self.direction = direction

    # def __repr__(self):
    #     return f"RotateAgent-{'right' if self.direction else 'left'}"

    COOK_STATE = [0, 0]
    AGENT_TO_POT = [(4, 0), (3, 0)]
    # 0: no onions, need to grab onion
    # 1: 1 onion, need to grab onion
    # 2: 2 onion, need to grab onion and activate pot
    # 3: need to activate pot
    # 4: need plate

    OBJECTS = {
        (6, 1): [],
        (5, 1): [],
        (4, 1): [((4, 0), Interests.POT)],
        (3, 1): [((3, 0), Interests.POT)],
        (2, 1): [],
        (1, 1): [],
        (1, 2): [((0, 2), Interests.PLATES)],
        (1, 3): [],
        (2, 3): [],
        (3, 3): [((3, 4), Interests.ONION)],
        (4, 3): [((4, 4), Interests.ONION)],
        (5, 3): [],
        (6, 3): [],
        (6, 2): [((7, 2), Interests.SERVE)]
    }

    COUNTERCLOCKWISE = {
        (6, 1): Direction.WEST,
        (5, 1): Direction.WEST,
        (4, 1): Direction.WEST,
        (3, 1): Direction.WEST,
        (2, 1): Direction.WEST,
        (1, 1): Direction.SOUTH,
        (1, 2): Direction.SOUTH,
        (1, 3): Direction.EAST,
        (2, 3): Direction.EAST,
        (3, 3): Direction.EAST,
        (4, 3): Direction.EAST,
        (5, 3): Direction.EAST,
        (6, 3): Direction.NORTH,
        (6, 2): Direction.NORTH
    }

    CLOCKWISE = {
        (6, 1): Direction.SOUTH,
        (5, 1): Direction.EAST,
        (4, 1): Direction.EAST,
        (3, 1): Direction.EAST,
        (2, 1): Direction.EAST,
        (1, 1): Direction.EAST,
        (1, 2): Direction.NORTH,
        (1, 3): Direction.NORTH,
        (2, 3): Direction.WEST,
        (3, 3): Direction.WEST,
        (4, 3): Direction.WEST,
        (5, 3): Direction.WEST,
        (6, 3): Direction.WEST,
        (6, 2): Direction.SOUTH
    }

    # def doInteract(self, loc, me, dir):
    #     target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
    #     if target == loc:
    #         return 'interact', {}
    #     else:
    #         return dir, {}

    def action(self, state):
        me = state.players[self.agent_index]

        objectsOfInterest = self.OBJECTS[me.position]
        for loc, object in objectsOfInterest:
            if object == Interests.PLATES:
                if self.COOK_STATE[self.agent_index] == 4 and me.held_object == None:
                    target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                    if target == loc:
                        return 'interact', {}
                    else:
                        return Direction.WEST, {} 
            elif object == Interests.ONION:
                if self.COOK_STATE[self.agent_index] in [0, 1, 2] and me.held_object == None:
                    target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                    if target == loc:
                        return 'interact', {}
                    else:
                        return Direction.SOUTH, {} 
            elif object == Interests.POT:
                if loc == self.AGENT_TO_POT[self.agent_index]: # if it is my pot
                    if me.held_object != None and me.held_object.name == "onion": 
                        target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                        if target == loc:
                            self.COOK_STATE[self.agent_index] += 1    
                            return 'interact', {}
                        else:
                            return Direction.NORTH, {}
                    elif self.COOK_STATE[self.agent_index] == 3:
                        target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                        if target == loc:
                            self.COOK_STATE[self.agent_index] += 1    
                            return 'interact', {}
                        else:
                            return Direction.NORTH, {}
                    elif me.held_object != None and me.held_object.name == "dish":
                        target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                        if target == loc:
                            return 'interact', {}
                        else:
                            return Direction.NORTH, {}
            elif object == Interests.SERVE:
                if me.held_object != None and me.held_object.name == "soup":
                    target = (me.position[0] + me.orientation[0], me.position[1] + me.orientation[1])
                    if target == loc:
                        self.COOK_STATE[self.agent_index] = 0 
                        return 'interact', {}
                    else:
                        return Direction.EAST, {}
                    
        return self.CLOCKWISE[me.position] if self.direction else self.COUNTERCLOCKWISE[me.position], {}

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index 

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        self.agent_index = None
        self.mdp = None
        self.COOK_STATE = [0, 0]

class OnionOnlyAgent(Agent):
    """
    A heuristic agent that only picks up onions and deposits them into pots.
    Ignores plates, dishes, soups, and serving entirely.
    """
    def __init__(self, mlam):
        # mlam: MediumLevelActionManager from your OvercookedEnv
        self.mlam = mlam
        self.mdp = mlam.mdp

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def reset(self):
        # No history needed for this simple heuristic
        pass

    def action(self, state):
        player = state.players[self.agent_index]
        # Get onion counters and pot states
        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state,
            list(self.mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states = self.mlam.mdp.get_pot_states(state)

        # 1) If not holding anything, plan to pick up an onion
        if not player.has_object():
            goals = self.mlam.pickup_onion_actions(counter_objects)

        # 2) If holding an onion, plan to put it in a pot
        elif player.get_object().name == "onion":
            goals = self.mlam.put_onion_in_pot_actions(pot_states)

        # 3) Otherwise, do nothing
        else:
            return Action.STAY, {}

        # Filter for valid goals
        valid_goals = [
            g for g in goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, g
            )
        ]
        if not valid_goals:
            return Action.STAY, {}

        # Choose the lowest‐cost plan among valid goals
        start = player.pos_and_or
        best_action = Action.STAY
        min_cost = float("inf")
        for goal in valid_goals:
            plan, _, cost = self.mlam.motion_planner.get_plan(start, goal)
            if cost < min_cost:
                min_cost = cost
                best_action = plan[0][0]

        return best_action, {}
