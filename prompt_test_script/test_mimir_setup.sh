#!/bin/bash
# Quick test script for Mimir setup

set -e

echo "=========================================="
echo "Testing Production-Like Setup with Mimir"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
echo -e "${YELLOW}1. Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"
echo ""

# Check docker-compose
echo -e "${YELLOW}2. Checking Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose found${NC}"
echo ""

# Check .env file
echo -e "${YELLOW}3. Checking .env file...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env with your API keys${NC}"
    exit 1
fi
echo -e "${GREEN}✅ .env file exists${NC}"
echo ""

# Start services
echo -e "${YELLOW}4. Starting Mimir and Grafana...${NC}"
docker-compose up -d mimir grafana

echo "Waiting for services to be ready..."
sleep 10

# Check Mimir health
echo -e "${YELLOW}5. Checking Mimir health...${NC}"
if curl -s http://localhost:9009/ready > /dev/null; then
    echo -e "${GREEN}✅ Mimir is ready${NC}"
else
    echo -e "${RED}❌ Mimir is not ready${NC}"
    echo "Check logs: docker-compose logs mimir"
    exit 1
fi
echo ""

# Check Grafana
echo -e "${YELLOW}6. Checking Grafana...${NC}"
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Grafana is ready${NC}"
else
    echo -e "${RED}❌ Grafana is not ready${NC}"
    echo "Check logs: docker-compose logs grafana"
    exit 1
fi
echo ""

# Run evaluation (generate only to save time)
echo -e "${YELLOW}7. Running evaluation test...${NC}"
echo "This will generate AI text and push metrics to Mimir"
echo ""

docker-compose run --rm ai-detector-eval \
    --input /tmp/input/prompts.csv \
    --generate-only \
    --output-dir /tmp/output \
    --run-id test-$(date +%s) \
    --dataset-version v1.0-test \
    --use-remote-write

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Evaluation completed${NC}"
else
    echo -e "${RED}❌ Evaluation failed${NC}"
    exit 1
fi
echo ""

# Check metrics
echo -e "${YELLOW}8. Checking metrics in Mimir...${NC}"
sleep 5

METRICS=$(curl -s "http://localhost:9009/prometheus/api/v1/query?query=ai_detector_eval_success")

if echo "$METRICS" | grep -q "success"; then
    echo -e "${GREEN}✅ Metrics found in Mimir${NC}"
    echo "Query result:"
    echo "$METRICS" | python3 -m json.tool || echo "$METRICS"
else
    echo -e "${YELLOW}⚠️  Metrics not found yet (may take a moment to propagate)${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Access points:"
echo "  - Grafana:  http://localhost:3000 (admin/admin)"
echo "  - Mimir:    http://localhost:9009"
echo ""
echo "Next steps:"
echo "  1. Open Grafana at http://localhost:3000"
echo "  2. Go to Explore → Select 'Mimir' datasource"
echo "  3. Try query: ai_detector_eval_success"
echo ""
echo "To run full evaluation:"
echo "  docker-compose up ai-detector-eval"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
