#!/usr/bin/env python
"""
Simplified State Converter for Training Data Generation
Converts state dictionaries to text descriptions without MDP dependencies.
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SimpleGameObject:
    """Simplified game object representation."""
    type: str
    pos: Tuple[int, int]
    status: str = "normal"
    holding: Optional[str] = None

@dataclass
class SimpleAgentState:
    """Simplified agent state representation."""
    agent_id: int
    pos: Tuple[int, int]
    facing: str = "UP"
    holding: Optional[str] = None
    status: str = "active"

class SimpleStateConverter:
    """Simple state converter for training data generation."""
    
    def __init__(self):
        # Object type mappings
        self.object_types = {
            'onion': 'ingredient',
            'pot': 'cookware',
            'dish': 'serving',
            'counter': 'surface',
            'stove': 'cooking_station',
            'fire': 'hazard'
        }
        
        # Status mappings
        self.status_mappings = {
            'cooking_time': 'cooking',
            'is_ready': 'ready',
            'is_empty': 'empty',
            'burning': 'burning',
            'active': 'active'
        }
    
    def state_to_text(self, state: Dict) -> str:
        """Convert a state dictionary to natural language text description."""
        if not state:
            return "Empty game state"
        
        text_parts = []
        
        # Describe agents
        if 'agents' in state:
            text_parts.append(self._describe_agents(state['agents']))
        
        # Describe objects
        if 'objects' in state:
            text_parts.append(self._describe_objects(state['objects']))
        
        # Describe layout
        if 'layout' in state:
            text_parts.append(f"The kitchen layout is {state['layout']}")
        
        # Describe orders
        if 'orders' in state and state['orders']:
            text_parts.append(f"There are {len(state['orders'])} active orders")
        
        # Describe time
        if 'time_remaining' in state:
            time_left = state['time_remaining']
            if time_left > 0:
                text_parts.append(f"Time remaining: {time_left} seconds")
            else:
                text_parts.append("Time is up")
        
        # Combine all parts
        if text_parts:
            return ". ".join(text_parts) + "."
        else:
            return "Game state information not available."
    
    def _describe_agents(self, agents: List[Dict]) -> str:
        """Generate text description of agents."""
        if not agents:
            return "No agents present"
        
        descriptions = []
        for agent in agents:
            agent_id = agent.get('id', 'Unknown')
            pos = agent.get('pos', (0, 0))
            facing = agent.get('facing', 'UP')
            holding = agent.get('holding')
            
            # Position description
            pos_desc = self._describe_position(pos)
            
            # Holding description
            if holding:
                holding_desc = f"holding a {holding}"
            else:
                holding_desc = "not holding anything"
            
            # Combine
            agent_desc = f"Agent {agent_id} is at {pos_desc}, facing {facing}, and {holding_desc}"
            descriptions.append(agent_desc)
        
        return " ".join(descriptions)
    
    def _describe_objects(self, objects: List[Dict]) -> str:
        """Generate text description of objects."""
        if not objects:
            return "No objects present"
        
        # Group objects by type
        object_groups = {}
        for obj in objects:
            obj_type = obj.get('type', 'unknown')
            if obj_type not in object_groups:
                object_groups[obj_type] = []
            object_groups[obj_type].append(obj)
        
        descriptions = []
        for obj_type, obj_list in object_groups.items():
            if len(obj_list) == 1:
                obj = obj_list[0]
                pos = obj.get('pos', (0, 0))
                status = obj.get('status', 'normal')
                pos_desc = self._describe_position(pos)
                
                if status == 'normal':
                    descriptions.append(f"There is a {obj_type} at {pos_desc}")
                else:
                    descriptions.append(f"There is a {obj_type} at {pos_desc} with status '{status}'")
            else:
                pos_descriptions = []
                for obj in obj_list:
                    pos = obj.get('pos', (0, 0))
                    status = obj.get('status', 'normal')
                    pos_desc = self._describe_position(pos)
                    
                    if status == 'normal':
                        pos_descriptions.append(pos_desc)
                    else:
                        pos_descriptions.append(f"{pos_desc} (status: {status})")
                
                descriptions.append(f"There are {len(obj_list)} {obj_type}s at {', '.join(pos_descriptions)}")
        
        return ". ".join(descriptions)
    
    def _describe_position(self, pos: Tuple[int, int]) -> str:
        """Convert position coordinates to natural language description."""
        x, y = pos
        
        # Create descriptive position names
        if x == 0 and y == 0:
            return "the top-left corner"
        elif x == 0 and y == 2:
            return "the top-right corner"
        elif x == 2 and y == 0:
            return "the bottom-left corner"
        elif x == 2 and y == 2:
            return "the bottom-right corner"
        elif x == 1 and y == 1:
            return "the center"
        elif x == 0:
            return f"the left side at row {y+1}"
        elif x == 2:
            return f"the right side at row {y+1}"
        elif y == 0:
            return f"the top at column {x+1}"
        elif y == 2:
            return f"the bottom at column {x+1}"
        else:
            return f"position ({x+1}, {y+1})"
    
    def convert_to_simple_format(self, state: Dict) -> Dict:
        """Convert state to a simplified format for training."""
        simple_state = {
            'agents': [],
            'objects': [],
            'layout': state.get('layout', 'unknown'),
            'orders': state.get('orders', []),
            'time_remaining': state.get('time_remaining', 300)
        }
        
        # Convert agents
        if 'agents' in state:
            for agent in state['agents']:
                simple_agent = SimpleAgentState(
                    agent_id=agent.get('id', 0),
                    pos=agent.get('pos', (0, 0)),
                    facing=agent.get('facing', 'UP'),
                    holding=agent.get('holding'),
                    status=agent.get('status', 'active')
                )
                simple_state['agents'].append(simple_agent)
        
        # Convert objects
        if 'objects' in state:
            for obj in state['objects']:
                simple_obj = SimpleGameObject(
                    type=obj.get('type', 'unknown'),
                    pos=obj.get('pos', (0, 0)),
                    status=obj.get('status', 'normal'),
                    holding=obj.get('holding')
                )
                simple_state['objects'].append(simple_obj)
        
        return simple_state

def main():
    """Demo the simple state converter."""
    print("=== Simple State Converter Demo ===\n")
    
    converter = SimpleStateConverter()
    
    # Test state
    test_state = {
        'agents': [
            {'id': 1, 'pos': (1, 1), 'facing': 'UP', 'holding': 'onion'},
            {'id': 2, 'pos': (2, 2), 'facing': 'DOWN', 'holding': None}
        ],
        'objects': [
            {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
            {'type': 'onion', 'pos': (0, 0), 'status': 'available'},
            {'type': 'counter', 'pos': (1, 0), 'status': 'normal'}
        ],
        'layout': 'random3',
        'orders': ['onion soup'],
        'time_remaining': 250
    }
    
    print("Test State:")
    print(json.dumps(test_state, indent=2))
    print("\nText Description:")
    text_desc = converter.state_to_text(test_state)
    print(text_desc)
    
    print("\nSimple Format:")
    simple_state = converter.convert_to_simple_format(test_state)
    print(json.dumps(simple_state, default=lambda x: x.__dict__, indent=2))

if __name__ == "__main__":
    main() 