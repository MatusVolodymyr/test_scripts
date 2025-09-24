#!/bin/bash
# Simple test runner - activates venv and runs the basic API test

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 Starting AI Detector API Test...${NC}"
echo

# Activate virtual environment
source venv/bin/activate

# Load environment variables
source .env 2>/dev/null || true

# Check if sample data exists
if [ ! -f "sample_test_data.csv" ]; then
    echo -e "${YELLOW}⚠️  No sample_test_data.csv found. Please add your test data.${NC}"
    echo "Expected CSV format: text,label,model_name,word_count,sentence_count,domain,source,confidence,language"
    echo
    read -p "Press Enter to continue with a different CSV file, or Ctrl+C to exit..."
    echo
    echo "Available CSV files:"
    ls -1 *.csv 2>/dev/null || echo "No CSV files found in current directory"
    echo
    read -p "Enter CSV filename to test: " CSV_FILE
    if [ ! -f "$CSV_FILE" ]; then
        echo "File not found: $CSV_FILE"
        exit 1
    fi
else
    CSV_FILE="sample_test_data.csv"
fi

# Run the test
echo "Testing with file: $CSV_FILE"
python test_detector_api.py --input "$CSV_FILE" --detailed

echo
echo -e "${GREEN}✅ Test completed! Check the test_results/ directory for detailed reports.${NC}"
