#!/bin/bash
# Quick test script for the 4 recommended layouts

echo "🧪 Testing 4 Recommended Overcooked Layouts"
echo "=========================================="
echo ""
echo "Available test scripts:"
echo "1. test_4_layouts.py - Interactive testing with manual intervention"
echo "2. test_multiple_layouts.py - Automated testing with auto-intervention"
echo ""
echo "Recommended layouts to test:"
echo "• corridor - Bottlenecks & coordination challenges"
echo "• scenario2 - Strategic positioning decisions"
echo "• schelling - Classic coordination game scenario" 
echo "• unident_s - Asymmetric design challenges"
echo ""

# Check if we're in the right directory
if [ ! -f "test_4_layouts.py" ]; then
    echo "❌ Error: Please run this script from the AHT_human_intervention directory"
    exit 1
fi

echo "🚀 Starting interactive layout testing..."
echo "Press 'n' to switch layouts, 'p' for intervention, ESC to quit"
echo ""

# Run the interactive test
python3 test_4_layouts.py
