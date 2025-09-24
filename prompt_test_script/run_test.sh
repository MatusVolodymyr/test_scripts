#!/bin/bash
# Simple test runner - activates venv and runs the test

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AI Detector Test...${NC}"
echo

# Activate virtual environment
source venv/bin/activate

# Run the test
python test_detector_prompts.py --input sample_prompts.csv --generate-and-test

echo
echo -e "${GREEN}✅ Test completed! Check the test_results/ directory for reports.${NC}"
