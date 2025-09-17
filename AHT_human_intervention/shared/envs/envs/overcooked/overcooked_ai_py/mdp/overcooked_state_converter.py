#!/usr/bin/env python
"""
Overcooked State to Text Converter
Converts Overcooked game states directly to natural language descriptions
instead of using lossless encoding.
"""

from typing import List, Dict, Tuple, Any, Optional
from .overcooked_mdp import OvercookedState, PlayerState, ObjectState, OvercookedGridworld


class OvercookedStateConverter:
    """Converts Overcooked game states to natural language text descriptions."""
    
    def __init__(self, mdp: OvercookedGridworld):
        """
        Initialize the converter with an MDP instance.
        
        Args:
            mdp: The OvercookedGridworld MDP instance
        """
        self.mdp = mdp
        self.layout_name = getattr(mdp, 'layout_name', 'unknown')
        
        # Direction to readable text mapping
        self.direction_names = {
            (0, -1): "north",
            (0, 1): "south", 
            (-1, 0): "west",
            (1, 0): "east"
        }
        
        # Object type descriptions
        self.object_descriptions = {
            'onion': 'onion',
            'tomato': 'tomato',
            'dish': 'dish',
            'soup': 'soup'
        }
    
    def state_to_text(self, state: OvercookedState) -> str:
        """
        Convert an Overcooked state to a comprehensive natural language description.
        
        Args:
            state: The OvercookedState to convert
            
        Returns:
            A natural language description of the game state
        """
        if not state:
            return "Empty game state"
        
        text_parts = []
        
        # Layout information
        text_parts.append(f"The kitchen layout is '{self.layout_name}'")
        
        # Player/Agent descriptions
        text_parts.append(self._describe_players(state))
        
        # Map features description
        text_parts.append(self._describe_map_features())
        
        # Object and state descriptions
        text_parts.append(self._describe_objects_and_states(state))
        
        # Order information
        if state.order_list:
            text_parts.append(self._describe_orders(state.order_list))
        
        # Timestep information
        text_parts.append(f"Current timestep: {state.timestep}")
        
        return ". ".join(text_parts) + "."
    
    def _describe_players(self, state: OvercookedState) -> str:
        """Generate comprehensive description of all players."""
        if not state.players:
            return "No players present"
        
        descriptions = []
        for i, player in enumerate(state.players):
            # Position description
            pos_desc = self._describe_position(player.position)
            
            # Orientation description
            orientation_desc = self.direction_names.get(player.orientation, "unknown direction")
            
            # Held object description
            if player.has_object():
                held_obj = player.get_object()
                if held_obj.name == 'soup':
                    soup_type, num_items, cook_time = held_obj.state
                    holding_desc = f"holding {num_items} {soup_type} soup"
                    if cook_time > 0:
                        holding_desc += f" that has been cooking for {cook_time} timesteps"
                else:
                    holding_desc = f"holding a {held_obj.name}"
            else:
                holding_desc = "not holding anything"
            
            # Combine player description
            player_desc = f"Player {i} is at {pos_desc}, facing {orientation_desc}, and {holding_desc}"
            descriptions.append(player_desc)
        
        return " ".join(descriptions)
    
    def _describe_map_features(self) -> str:
        """Describe the static map features."""
        features = []
        
        # Counters
        counter_locs = self.mdp.get_counter_locations()
        if counter_locs:
            features.append(f"{len(counter_locs)} counter{'s' if len(counter_locs) > 1 else ''}")
        
        # Pots
        pot_locs = self.mdp.get_pot_locations()
        if pot_locs:
            features.append(f"{len(pot_locs)} pot{'s' if len(pot_locs) > 1 else ''}")
        
        # Onion dispensers
        onion_disp_locs = self.mdp.get_onion_dispenser_locations()
        if onion_disp_locs:
            features.append(f"{len(onion_disp_locs)} onion dispenser{'s' if len(onion_disp_locs) > 1 else ''}")
        
        # Dish dispensers
        dish_disp_locs = self.mdp.get_dish_dispenser_locations()
        if dish_disp_locs:
            features.append(f"{len(dish_disp_locs)} dish dispenser{'s' if len(dish_disp_locs) > 1 else ''}")
        
        # Serving locations
        serve_locs = self.mdp.get_serving_locations()
        if serve_locs:
            features.append(f"{len(serve_locs)} serving location{'s' if len(serve_locs) > 1 else ''}")
        
        if features:
            return f"The kitchen contains: {', '.join(features)}"
        else:
            return "The kitchen has no special features"
    
    def _describe_objects_and_states(self, state: OvercookedState) -> str:
        """Describe all objects and their current states."""
        if not state.objects:
            return "No objects are present in the kitchen"
        
        # Group objects by type
        objects_by_type = {}
        for pos, obj in state.objects.items():
            obj_type = obj.name
            if obj_type not in objects_by_type:
                objects_by_type[obj_type] = []
            objects_by_type[obj_type].append((pos, obj))
        
        descriptions = []
        
        for obj_type, obj_list in objects_by_type.items():
            if obj_type == 'soup':
                descriptions.append(self._describe_soups(obj_list))
            elif obj_type == 'onion':
                descriptions.append(self._describe_onions(obj_list))
            elif obj_type == 'dish':
                descriptions.append(self._describe_dishes(obj_list))
            else:
                # Generic object description
                pos_descriptions = [self._describe_position(pos) for pos, _ in obj_list]
                if len(pos_descriptions) == 1:
                    descriptions.append(f"There is a {obj_type} at {pos_descriptions[0]}")
                else:
                    descriptions.append(f"There are {len(pos_descriptions)} {obj_type}s at {', '.join(pos_descriptions)}")
        
        return ". ".join(descriptions)
    
    def _describe_soups(self, soup_list: List[Tuple[Tuple[int, int], ObjectState]]) -> str:
        """Describe soup objects with their cooking states."""
        if not soup_list:
            return ""
        
        # Group soups by location type (pots vs counters)
        pot_soups = []
        counter_soups = []
        
        for pos, soup in soup_list:
            if pos in self.mdp.get_pot_locations():
                pot_soups.append((pos, soup))
            else:
                counter_soups.append((pos, soup))
        
        descriptions = []
        
        # Describe soups in pots
        if pot_soups:
            pot_desc = []
            for pos, soup in pot_soups:
                soup_type, num_items, cook_time = soup.state
                pos_desc = self._describe_position(pos)
                pot_desc.append(f"{num_items} {soup_type} soup at {pos_desc} (cooking for {cook_time} timesteps)")
            descriptions.append(f"Soups in pots: {', '.join(pot_desc)}")
        
        # Describe soups on counters
        if counter_soups:
            counter_desc = []
            for pos, soup in counter_soups:
                soup_type, num_items, cook_time = soup.state
                pos_desc = self._describe_position(pos)
                if cook_time >= self.mdp.soup_cooking_time:
                    status = "fully cooked"
                elif cook_time > 0:
                    status = f"partially cooked ({cook_time}/{self.mdp.soup_cooking_time})"
                else:
                    status = "uncooked"
                counter_desc.append(f"{num_items} {soup_type} soup at {pos_desc} ({status})")
            descriptions.append(f"Soups on counters: {', '.join(counter_desc)}")
        
        return ". ".join(descriptions)
    
    def _describe_onions(self, onion_list: List[Tuple[Tuple[int, int], ObjectState]]) -> str:
        """Describe onion objects."""
        if not onion_list:
            return ""
        
        pos_descriptions = [self._describe_position(pos) for pos, _ in onion_list]
        if len(pos_descriptions) == 1:
            return f"There is an onion at {pos_descriptions[0]}"
        else:
            return f"There are {len(pos_descriptions)} onions at {', '.join(pos_descriptions)}"
    
    def _describe_dishes(self, dish_list: List[Tuple[Tuple[int, int], ObjectState]]) -> str:
        """Describe dish objects."""
        if not dish_list:
            return ""
        
        pos_descriptions = [self._describe_position(pos) for pos, _ in dish_list]
        if len(pos_descriptions) == 1:
            return f"There is a dish at {pos_descriptions[0]}"
        else:
            return f"There are {len(pos_descriptions)} dishes at {', '.join(pos_descriptions)}"
    
    def _describe_orders(self, order_list: List[str]) -> str:
        """Describe current orders."""
        if not order_list:
            return "No orders are currently active"
        
        if len(order_list) == 1:
            return f"Current order: {order_list[0]}"
        else:
            order_text = ", ".join(order_list)
            return f"Current orders ({len(order_list)}): {order_text}"
    
    def _describe_position(self, pos: Tuple[int, int]) -> str:
        """Convert position coordinates to natural language description."""
        x, y = pos
        width, height = self.mdp.shape
        
        # Create descriptive position names based on grid dimensions
        if width <= 5 and height <= 8:  # Small grid
            if x == 0 and y == 0:
                return "the top-left corner"
            elif x == width-1 and y == 0:
                return "the top-right corner"
            elif x == 0 and y == height-1:
                return "the bottom-left corner"
            elif x == width-1 and y == height-1:
                return "the bottom-right corner"
            elif x == width//2 and y == height//2:
                return "the center"
            elif x == 0:
                return f"the left edge at row {y+1}"
            elif x == width-1:
                return f"the right edge at row {y+1}"
            elif y == 0:
                return f"the top edge at column {x+1}"
            elif y == height-1:
                return f"the bottom edge at column {x+1}"
            else:
                return f"position ({x+1}, {y+1})"
        else:
            # For larger grids, use coordinate-based description
            return f"position ({x+1}, {y+1})"
    
    def get_summary_stats(self, state: OvercookedState) -> Dict[str, Any]:
        """Get a summary of key game statistics."""
        stats = {
            'layout': self.layout_name,
            'timestep': state.timestep,
            'num_players': len(state.players),
            'num_orders': len(state.order_list) if state.order_list else 0,
            'objects_summary': {}
        }
        
        # Count objects by type
        for pos, obj in state.objects.items():
            obj_type = obj.name
            if obj_type not in stats['objects_summary']:
                stats['objects_summary'][obj_type] = 0
            stats['objects_summary'][obj_type] += 1
        
        # Add player-held objects
        for player in state.players:
            if player.has_object():
                obj_type = player.get_object().name
                if obj_type not in stats['objects_summary']:
                    stats['objects_summary'][obj_type] = 0
                stats['objects_summary'][obj_type] += 1
        
        return stats


def create_state_converter(mdp: OvercookedGridworld) -> OvercookedStateConverter:
    """Factory function to create a state converter for a given MDP."""
    return OvercookedStateConverter(mdp)


def state_to_text(state):
    """Simple function to convert state to text for backward compatibility."""
    # This is a simplified version for the rollout script
    if not state:
        return "Empty game state"
    
    text_parts = []
    
    # Player descriptions
    if state.players:
        for i, player in enumerate(state.players):
            pos = player.position
            orientation = player.orientation
            direction_names = {(0, -1): "north", (0, 1): "south", (-1, 0): "west", (1, 0): "east"}
            direction = direction_names.get(orientation, "unknown direction")
            
            if player.has_object():
                held_obj = player.get_object()
                holding = f"holding a {held_obj.name}"
            else:
                holding = "not holding anything"
            
            text_parts.append(f"Player {i+1} is at position {pos} facing {direction} and {holding}")
    
    # Objects
    if state.objects:
        obj_descriptions = []
        for pos, obj in state.objects.items():
            obj_descriptions.append(f"{obj.name} at {pos}")
        text_parts.append(f"Objects: {', '.join(obj_descriptions)}")
    
    # Orders
    if state.order_list:
        text_parts.append(f"Orders: {', '.join(state.order_list)}")
    
    return ". ".join(text_parts) + "."


# Example usage and testing
if __name__ == "__main__":
    # This would be used in practice like:
    # from overcooked_mdp import OvercookedGridworld
    # mdp = OvercookedGridworld(...)
    # converter = create_state_converter(mdp)
    # text_description = converter.state_to_text(state)
    # print(text_description)
    
    print("Overcooked State Converter initialized.")
    print("Use create_state_converter(mdp) to create an instance.") 