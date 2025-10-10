# AI Detector Evaluation Suite

Production-ready evaluation tool for AI content detectors. Generates synthetic test data using multiple LLM providers, tests against detector APIs, calculates performance metrics, and exports results with OpenTelemetry metrics to Prometheus-compatible backends.

## Features

- ✅ **Multi-LLM Support**: OpenAI GPT-4/5, Anthropic Claude with universal system prompt
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- ✅ **OpenTelemetry Export**: Native Prometheus Remote Write (protobuf) with automatic 5-second export
- ✅ **Multiple Output Formats**: JSON, JSONL (streaming), CSV, Manifest
- ✅ **Container-Ready**: Docker image with production best practices
- ✅ **AWS Integration**: Secrets Manager support, CloudWatch compatible logging

---

## Quick Start

### 1. Build Docker Image

```bash
docker build -t ai-detector-eval:latest .
```

### 2. Run Evaluation

```bash
docker run --rm \
  -e OPENAI_API_KEY="sk-proj-..." \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e DETECTOR_API_URL="https://your-detector.example.com" \
  -e DETECTOR_API_TOKEN="your-token" \
  -e PROMETHEUS_REMOTE_WRITE_URL="https://your-mimir.example.com/api/v1/push" \
  -e PROMETHEUS_TENANT_ID="your-tenant-id" \
  -v $(pwd)/prompts.csv:/tmp/input/prompts.csv:ro \
  -v $(pwd)/output:/tmp/output \
  ai-detector-eval:latest \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --save-manifest \
  --output-dir /tmp/output
```

### 3. View Results

```bash
ls -lh output/
# Generated files:
# - generated_test_data_20251010_*.csv  (generated texts backup)
# - prompt_test_report_20251010_*.json  (detailed results)
# - results_20251010_*.jsonl            (streaming format)
# - manifest_20251010_*.json            (run metadata)
```

---

## Environment Variables

### Required for Text Generation
```bash
OPENAI_API_KEY=sk-proj-...          # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...        # Anthropic API key
```

### Required for Testing
```bash
DETECTOR_API_URL=https://...        # Your detector API endpoint
DETECTOR_API_TOKEN=your-token       # Authentication token (if required)
```

### Required for Metrics Export
```bash
PROMETHEUS_REMOTE_WRITE_URL=https://mimir.example.com/api/v1/push
PROMETHEUS_TENANT_ID=your-tenant-id  # Mimir tenant ID (e.g., "team-ml")
```

### Optional Metadata
```bash
RUN_ID=eval-$(date +%s)             # Unique run identifier
DATASET_VERSION=v1.0                # Dataset version label
GIT_COMMIT_SHA=$(git rev-parse HEAD) # Git commit for reproducibility
ENVIRONMENT=production              # Environment name (dev/staging/prod)
CLUSTER=us-east-1                   # Cluster/region identifier
```

---

## Command-Line Usage

### Basic Usage
```bash
python test_detector_prompts.py \
  --input prompts.csv \
  --generate-and-test
```

### Production Run with All Features
```bash
python test_detector_prompts.py \
  --input prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --save-manifest \
  --run-id "eval-$(date +%s)" \
  --dataset-version "v1.0" \
  --commit-sha "$(git rev-parse HEAD)" \
  --exit-on-error
```

### Mode Options
```bash
--generate-only       # Generate AI text without testing detector
--test-only          # Test pre-generated data only
--generate-and-test  # Full workflow (generate + test)
```

### Output Options
```bash
--output-dir DIR     # Output directory (default: test_results)
--save-jsonl         # Save streaming JSONL format (recommended for S3)
--save-manifest      # Generate manifest.json with run metadata
--detailed           # Verbose progress output
--quiet              # Minimal output
--exit-on-error      # Exit with code 1 if tests fail (for CI/CD)
```

---

## Input CSV Format

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `type` | Yes | `prompt` or `human_text` | `prompt` |
| `content` | Yes | Prompt or text content | `Write about climate change` |
| `model` | For prompts | AI model to use | `gpt-4o`, `claude-3-5-haiku` |
| `expected_label` | Yes | Expected detector result | `ai`, `human` |
| `temperature` | No | Generation randomness (0-1) | `0.7` |
| `max_tokens` | No | Maximum tokens | `500` |
| `style` | No | Writing style | `formal`, `casual` |
| `domain` | No | Content domain | `technology`, `science` |

### Supported AI Models

**OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`  
**Anthropic**: `claude-3-5-haiku`, `claude-3-5-sonnet`, `claude-3-opus`

---

## Output Files

### Standard Outputs
- **`generated_test_data_*.csv`**: Generated texts with metadata (backup)
- **`prompt_test_report_*.json`**: Detailed test report with full metrics

### Production Outputs (with flags)
- **`results_*.jsonl`**: Streaming JSONL format (use `--save-jsonl`)
- **`manifest_*.json`**: Run metadata (use `--save-manifest`)

### Manifest Example
```json
{
  "run_id": "eval-1728489600",
  "dataset_version": "v1.0",
  "commit_sha": "abc123def",
  "timestamp": "2025-10-10T12:00:00Z",
  "total_samples": 50,
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1": 0.92
  }
}
```

---

## OpenTelemetry Metrics

The script exports 9 metrics to Prometheus-compatible backends (Mimir, Cortex, etc.) via OpenTelemetry:

| Metric Name | Type | Description |
|------------|------|-------------|
| `ai_detector_eval_success` | Gauge | 1 if successful, 0 if failed |
| `ai_detector_eval_run_timestamp_seconds` | Gauge | Unix timestamp of run start |
| `ai_detector_eval_duration_seconds` | Gauge | Total run duration in seconds |
| `ai_detector_eval_samples_total` | Counter | Sample counts with `label="ai"` or `"human"` |
| `ai_detector_eval_confusion_matrix` | Counter | Confusion matrix with `kind="tp"/"tn"/"fp"/"fn"` |
| `ai_detector_eval_accuracy` | Gauge | Overall accuracy (0-1) |
| `ai_detector_eval_precision` | Gauge | Precision score (0-1) |
| `ai_detector_eval_recall` | Gauge | Recall score (0-1) |
| `ai_detector_eval_f1` | Gauge | F1 score (0-1) |

### Automatic Export
- Metrics are exported every **5 seconds** via OpenTelemetry's `PeriodicExportingMetricReader`
- Uses **Prometheus Remote Write** protocol with native protobuf encoding
- No intermediate services required (Pushgateway, Grafana Agent eliminated)

### Resource Attributes
All metrics include these resource attributes for filtering:
- `service.name`: `ai-detector-eval`
- `environment`: From `ENVIRONMENT` env var (default: `local`)
- `cluster`: From `CLUSTER` env var (default: `local-test`)

### Example Grafana Query
```promql
# Current accuracy
ai_detector_eval_accuracy_1

# Filter by environment
ai_detector_eval_accuracy_1{environment="production"}

# Sample distribution
sum by (label) (ai_detector_eval_samples_total_1)

# Confusion matrix
sum by (kind) (ai_detector_eval_confusion_matrix_1)
```

---

## Docker

### Build
```bash
docker build -t ai-detector-eval:latest .
```

### Run Production Container
```bash
docker run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e DETECTOR_API_URL="${DETECTOR_API_URL}" \
  -e DETECTOR_API_TOKEN="${DETECTOR_API_TOKEN}" \
  -e PROMETHEUS_REMOTE_WRITE_URL="${PROMETHEUS_REMOTE_WRITE_URL}" \
  -e PROMETHEUS_TENANT_ID="${PROMETHEUS_TENANT_ID}" \
  -e RUN_ID="eval-$(date +%s)" \
  -e DATASET_VERSION="v1.0" \
  -e GIT_COMMIT_SHA="$(git rev-parse HEAD)" \
  -e ENVIRONMENT="production" \
  -e CLUSTER="us-east-1" \
  -v /path/to/prompts.csv:/tmp/input/prompts.csv:ro \
  -v $(pwd)/output:/tmp/output \
  ai-detector-eval:latest \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --save-manifest \
  --output-dir /tmp/output \
  --exit-on-error
```

### Local Development Stack
For local testing with Mimir and Grafana:
```bash
docker-compose up
```

This starts:
- **Mimir** on http://localhost:9009 (metrics backend)
- **Grafana** on http://localhost:3000 (visualization) - login: admin/admin
- **Evaluation script** (runs once and exits)

View metrics in Grafana:
1. Open http://localhost:3000
2. Navigate to Explore (compass icon)
3. Select "Mimir" datasource
4. Query: `ai_detector_eval_accuracy_1`

---

## Integration with Customer's Grafana

**Your customer's existing Grafana setup does NOT need any changes from this repository.**

### What Your Customer Needs:
1. **Add Mimir datasource** to their existing Grafana:
   - URL: Their Mimir endpoint (e.g., `https://mimir.example.com/prometheus`)
   - HTTP Header: `X-Scope-OrgID: your-tenant-id`

2. **Run your containerized script** with their environment variables:
   ```bash
   PROMETHEUS_REMOTE_WRITE_URL=https://mimir.example.com/api/v1/push
   PROMETHEUS_TENANT_ID=your-tenant-id
   ```

3. **Query metrics** in their existing Grafana dashboards using `ai_detector_eval_*` metrics

### What They DON'T Need:
- ❌ The `grafana` service from this repo's `docker-compose.yml`
- ❌ The `grafana-datasource.yaml` configuration file
- ❌ Any local Grafana setup

**The `docker-compose.yml` in this repo is only for local development/testing.** Production deployment only needs your containerized script running with the correct environment variables pointing to their existing Mimir/Grafana infrastructure.

---

## Example Results

```
DETECTOR TEST RESULTS
====================================
Total samples: 25
Successful predictions: 25
Failed requests: 0

PERFORMANCE METRICS:
Accuracy:  0.920
Precision: 0.900
Recall:    0.947
F1 Score:  0.923

CONFUSION MATRIX:
True Positives:  18
True Negatives:  5
False Positives: 1
False Negatives: 1

Average response time: 145.2ms

✅ Metrics exported to Mimir via OpenTelemetry
```

---

## Troubleshooting

### Metrics Not Appearing in Grafana

**Issue**: 401 error or "no org id" when querying Mimir  
**Solution**: Ensure your Mimir datasource in Grafana has the `X-Scope-OrgID` header configured:
```yaml
jsonData:
  httpHeaderName1: 'X-Scope-OrgID'
secureJsonData:
  httpHeaderValue1: 'your-tenant-id'
```

**Issue**: Metrics not showing up  
**Solution**: Check that `PROMETHEUS_REMOTE_WRITE_URL` is accessible from your container:
```bash
curl -X POST ${PROMETHEUS_REMOTE_WRITE_URL} -I
# Should return 400 (Bad Request) or 204 (No Content), not connection errors
```

### API Key Issues

**Issue**: "Invalid API key" errors  
**Solution**: 
- Verify API keys are valid and not expired
- Check format: OpenAI starts with `sk-proj-`, Anthropic with `sk-ant-`
- For AWS Secrets Manager, ensure IAM permissions are correct

### Docker Networking

**Issue**: Cannot reach detector API on localhost  
**Solution**: Use `host.docker.internal` for Mac/Windows:
```bash
DETECTOR_API_URL=http://host.docker.internal:8000
```

For Linux, use `--network host` mode:
```bash
docker run --network host ...
```

---

## Development

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run locally
python test_detector_prompts.py \
  --input sample_prompts.csv \
  --generate-and-test
```

### Project Structure
```
prompt_test_script/
├── test_detector_prompts.py    # Main evaluation script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Production container image
├── docker-compose.yml          # Local dev stack (Mimir + Grafana)
├── .env.example                # Environment template
├── .dockerignore              # Build optimization
├── sample_prompts.csv          # Example input data
├── mimir-config.yaml          # Mimir configuration (local dev)
├── grafana-datasource.yaml    # Grafana datasource (local dev)
└── test_results/              # Output directory
```

### Dependencies
```
openai>=1.0.0                                    # OpenAI API client
anthropic>=0.18.0                                # Anthropic API client
requests>=2.28.0                                 # HTTP client
tqdm>=4.64.0                                    # Progress bars
python-dotenv>=1.0.0                            # Environment variables
opentelemetry-api>=1.20.0                       # OpenTelemetry API
opentelemetry-sdk>=1.20.0                       # OpenTelemetry SDK
opentelemetry-exporter-prometheus-remote-write>=0.41b0  # Prometheus exporter
boto3>=1.28.0                                   # AWS SDK (optional)
```

### Exit Codes
- **0**: Successful execution
- **1**: Error (API failure, validation error, test failure with `--exit-on-error`)
- **130**: Interrupted by user (Ctrl+C)

---

## License

[Your License Here]

## Support

For issues or questions, please open an issue in the repository or contact your DevOps team.

---

**Migration Note**: This tool uses OpenTelemetry with native Prometheus Remote Write (protobuf). Previous versions used `prometheus-client` with Pushgateway and Grafana Agent. The new architecture is simpler, more efficient, and production-ready.
