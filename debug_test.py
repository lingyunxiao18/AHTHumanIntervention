#!/usr/bin/env python

import traceback
import os
from AHT_human_intervention.intervention_LLM_module import process_command

print(f"API Key available: {bool(os.getenv('OPENAI_API_KEY'))}")

try:
    print("Testing process_command with empty string...")
    result = process_command('')
    print(f"Empty string result: {result}")
    print(f"Type: {type(result)}")
    
    print("\nTesting process_command with test command...")
    result = process_command('test command')
    print(f"Test command result: {result}")
    print(f"Type: {type(result)}")
    
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc() 