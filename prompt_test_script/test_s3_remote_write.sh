#!/bin/bash
# Test script for S3 and Prometheus Remote Write features

echo "=========================================="
echo "AI Detector Evaluation - S3 & Remote Write Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please copy .env.example to .env and configure your settings"
    exit 1
fi

# Source environment variables
source .env

# Test 1: Basic functionality (no S3, no metrics)
echo -e "${YELLOW}Test 1: Basic Generate-Only (Baseline)${NC}"
python test_detector_prompts.py \
    --input sample_prompts.csv \
    --generate-only \
    --output-dir test_results/test1_baseline

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test 1 passed${NC}"
else
    echo -e "${RED}❌ Test 1 failed${NC}"
fi
echo ""

# Test 2: S3 Upload (if configured)
if [ -n "$S3_BUCKET_NAME" ]; then
    echo -e "${YELLOW}Test 2: Generate with S3 Upload${NC}"
    python test_detector_prompts.py \
        --input sample_prompts.csv \
        --generate-only \
        --output-dir test_results/test2_s3 \
        --s3-bucket "$S3_BUCKET_NAME" \
        --s3-prefix "test-runs/$(date +%Y%m%d)/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Test 2 passed - Check S3 bucket: $S3_BUCKET_NAME${NC}"
    else
        echo -e "${RED}❌ Test 2 failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Test 2 skipped - S3_BUCKET_NAME not configured${NC}"
fi
echo ""

# Test 3: Full evaluation with all features
echo -e "${YELLOW}Test 3: Full Evaluation with All Features${NC}"

# Build command
CMD="python test_detector_prompts.py \
    --input sample_prompts.csv \
    --generate-and-test \
    --output-dir test_results/test3_full \
    --save-jsonl \
    --save-manifest \
    --run-id test-$(date +%s) \
    --dataset-version v1.0-test \
    --commit-sha $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"

# Add S3 if configured
if [ -n "$S3_BUCKET_NAME" ]; then
    CMD="$CMD --s3-bucket $S3_BUCKET_NAME --s3-prefix test-runs/full/"
fi

# Add metrics if configured
if [ -n "$PUSHGATEWAY_URL" ]; then
    CMD="$CMD --pushgateway-url $PUSHGATEWAY_URL"
    echo "Using Pushgateway: $PUSHGATEWAY_URL"
elif [ -n "$PROMETHEUS_REMOTE_WRITE_URL" ]; then
    CMD="$CMD --use-remote-write"
    echo "Using Remote Write: $PROMETHEUS_REMOTE_WRITE_URL"
fi

echo "Running: $CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test 3 passed${NC}"
else
    echo -e "${RED}❌ Test 3 failed${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Test results saved to: test_results/"
echo ""
echo "Output files:"
ls -lh test_results/test*/

if [ -n "$S3_BUCKET_NAME" ]; then
    echo ""
    echo "S3 files (check your bucket):"
    echo "  s3://$S3_BUCKET_NAME/test-runs/"
fi

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
