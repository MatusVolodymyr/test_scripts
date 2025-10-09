# AI Detector Evaluation Suite

Evaluation tool for AI content detectors. Generates synthetic test data using multiple LLM providers (OpenAI GPT-4/5, Anthropic Claude), tests against detector APIs, calculates performance metrics, and exports results in multiple formats.

## Features

### Text Generation
- ✅ Multi-provider support (OpenAI GPT-4o/GPT-5/GPT-5-mini/GPT-5-nano, Anthropic Claude)
- ✅ Universal system prompt with dynamic parameter mapping
- ✅ Human-like text generation with style/domain controls
- ✅ Temperature & length mapping to natural language instructions
- ✅ GPT-5 Responses API support

### Testing & Metrics
- ✅ Accuracy, Precision, Recall, F1 score
- ✅ Confusion matrix (TP, TN, FP, FN)
- ✅ Per-model performance breakdown
- ✅ Prometheus metrics export (9 metrics tracked)

### Output Formats
- ✅ **JSON**: Detailed test report with full metrics
- ✅ **JSONL**: Streaming format for S3/large-scale storage
- ✅ **CSV**: Generated test data backup
- ✅ **Manifest**: Run metadata (run_id, dataset version, commit SHA)

### Production (not tested)
- ✅ Docker containerization (Python 3.11-slim)
- ✅ AWS Secrets Manager integration ready
- ✅ Prometheus/Grafana integration
- ✅ Proper exit codes (0=success, 1=error, 130=interrupt)
- ✅ CloudWatch logging compatible

---

## Quick Start

### Option 1: Docker (Recommended)

1. **Set up environment:**
```bash
cd prompt_test_script
cp .env.example .env
# Edit .env with your API keys
```

2. **Run with Docker Compose:**
```bash
docker-compose up
```

This starts:
- Prometheus Pushgateway (http://localhost:9091)
- Evaluation script with full metrics export
- Results saved to `./test_results/`

3. **View results:**
```bash
ls -la test_results/
open http://localhost:9091  # View metrics
```

### Option 2: Local Python

1. **Install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - DETECTOR_API_URL
# - DETECTOR_API_TOKEN
```

3. **Run evaluation:**
```bash
python test_detector_prompts.py \
  --input sample_prompts.csv \
  --generate-and-test
```

---

## Usage Examples

### Basic: Generate and Test
```bash
python test_detector_prompts.py \
  --input sample_prompts.csv \
  --generate-and-test \
  --api-url http://localhost:8000
```

### Generate Only (No Testing)
```bash
python test_detector_prompts.py \
  --input sample_prompts.csv \
  --generate-only \
  --output-dir test_results
```

### Test Pre-Generated Data
```bash
python test_detector_prompts.py \
  --input test_results/generated_test_data_20251009_123456.csv \
  --test-only \
  --api-url http://localhost:8000
```

### Production Run with All Features
```bash
python test_detector_prompts.py \
  --input sample_prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --save-manifest \
  --run-id "eval-$(date +%s)" \
  --dataset-version "v1.0" \
  --commit-sha "$(git rev-parse HEAD)" \
  --pushgateway-url "http://localhost:9091" \
  --exit-on-error
```

### Docker with Custom Prompts
```bash
docker run --rm \
  --env-file .env \
  -v /path/to/prompts.csv:/tmp/input/prompts.csv:ro \
  -v $(pwd)/test_results:/tmp/output \
  ai-detector-eval:latest \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --output-dir /tmp/output
```

---

## CLI Arguments

### Required Arguments
```
--input PATH              Input CSV file with prompts/test data
```

### Mode Selection (choose one)
```
--generate-only           Generate AI text without testing
--test-only              Test pre-generated data only
--generate-and-test      Generate and test (default workflow)
```

### API Configuration
```
--api-url URL            Detector API URL (default: http://localhost:8000)
--token TOKEN            API authentication token
```

### Production Features
```
--run-id ID              Unique run identifier for tracking
--dataset-version VER    Dataset version label
--commit-sha SHA         Git commit SHA for reproducibility
--pushgateway-url URL    Prometheus Pushgateway URL for metrics
--save-jsonl             Save results in JSONL format (S3-friendly)
--save-manifest          Generate manifest.json with run metadata
--exit-on-error          Exit with code 1 if tests fail (for CI/CD)
```

### Output Options
```
--output-dir DIR         Output directory (default: test_results)
--detailed               Show detailed progress (verbose mode)
--quiet                  Minimal output
```

---

## Input CSV Format

The input CSV should have these columns:

| Column | Description | Required | Examples |
|--------|-------------|----------|----------|
| `type` | `prompt` or `human_text` | Yes | `prompt`, `human_text` |
| `content` | Prompt text or human text | Yes | `Write a story about...` |
| `model` | AI model to use | For prompts | `gpt-4o`, `gpt-5`, `claude-3-5-haiku` |
| `temperature` | Generation randomness (0-1) | No | `0.7`, `0.9` |
| `max_tokens` | Maximum tokens to generate | No | `400`, `600` |
| `style` | Writing style | No | `formal`, `casual`, `conversational` |
| `domain` | Content domain | No | `technology`, `literature`, `science` |
| `expected_label` | Expected detector result | Yes | `ai`, `human` |
| `confidence` | Confidence in label | No | `1.0`, `0.8` |
| `language` | Content language | No | `en`, `es`, `fr` |
| `notes` | Additional notes | No | Any text |

**Note**: The `human_style` column has been removed. Text style is now controlled through the universal system prompt using `temperature`, `max_tokens`, `style`, and `domain` parameters.

---

## Universal System Prompt

The script uses an intelligent system prompt that adapts based on parameters:

### Temperature Mapping
- **≤ 0.3**: "Be precise and measured in your writing"
- **0.4-0.7**: "Write naturally and conversationally"  
- **≥ 0.8**: "Be creative and expressive in your writing"

### Length Mapping (max_tokens)
- **≤ 300**: "Keep it concise and to the point"
- **≤ 500**: "Provide a moderate amount of detail"
- **≤ 800**: "Write in depth with thorough explanations"
- **> 800**: "Write comprehensively with rich detail"

### Style & Domain Integration
Dynamically incorporates the `style` (e.g., "formal", "casual") and `domain` (e.g., "technology", "academic") into natural language instructions.

---

## Supported AI Models

### OpenAI Models
- `gpt-4o`, `gpt-4o-mini`
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- **Note**: GPT-5 models use the OpenAI Responses API with fixed decoding parameters

### Anthropic Models
- `claude-3-5-haiku`, `claude-3-5-sonnet`, `claude-3-opus`
- Other Claude variants

---

## Output Files

### Standard Outputs
- **`generated_test_data_YYYYMMDD_HHMMSS.csv`**: Generated texts (backup)
- **`prompt_test_report_YYYYMMDD_HHMMSS.json`**: Detailed test report

### Production Outputs (Optional)
- **`results_YYYYMMDD_HHMMSS.jsonl`**: Streaming JSONL format (enabled with `--save-jsonl`)
- **`manifest_YYYYMMDD_HHMMSS.json`**: Run metadata (enabled with `--save-manifest`)

### Manifest Example
```json
{
  "run_id": "eval-1728489600",
  "dataset_version": "v1.0",
  "commit_sha": "abc123def456",
  "timestamp": "2025-10-09T12:00:00Z",
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

## Prometheus Metrics

When `--pushgateway-url` is specified, the following metrics are exported:

| Metric | Type | Description |
|--------|------|-------------|
| `ai_detector_eval_success` | Gauge | 1 if successful, 0 if failed |
| `ai_detector_eval_run_timestamp_seconds` | Gauge | Unix timestamp of run |
| `ai_detector_eval_duration_seconds` | Gauge | Total run duration |
| `ai_detector_eval_samples_total{label="ai\|human"}` | Gauge | Sample counts by label |
| `ai_detector_eval_confusion_matrix{kind="tp\|tn\|fp\|fn"}` | Gauge | Confusion matrix values |
| `ai_detector_eval_accuracy` | Gauge | Overall accuracy |
| `ai_detector_eval_precision` | Gauge | Precision score |
| `ai_detector_eval_recall` | Gauge | Recall score |
| `ai_detector_eval_f1` | Gauge | F1 score |

---

## Docker

### Build Image
```bash
docker build -t ai-detector-eval:latest .
```

### Run with Docker Compose (Local Testing)
```bash
docker-compose up
```

This starts:
- **Pushgateway** on port 9091
- **Evaluation container** with metrics export

### Run Standalone Container
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/sample_prompts.csv:/tmp/input/prompts.csv:ro \
  -v $(pwd)/test_results:/tmp/output \
  ai-detector-eval:latest \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --output-dir /tmp/output
```

### Environment Variables (Docker)
```bash
# Required
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
DETECTOR_API_URL=http://detector:8000
DETECTOR_API_TOKEN=your-token

# Optional (for production features)
PUSHGATEWAY_URL=http://pushgateway:9091
RUN_ID=eval-123
DATASET_VERSION=v1.0
GIT_COMMIT_SHA=abc123
```

---

## Cost Estimation

The script tracks API usage costs:

```
💰 Total estimated generation cost: $0.0234
💰 OpenAI cost: $0.0120
💰 Anthropic cost: $0.0114
```

**Note**: These are estimates based on public pricing. Actual costs may vary.

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
```

---

## Troubleshooting

### Common Issues

**Empty output from GPT-5 models:**
- GPT-5 models use the Responses API, not Chat API
- Ensure you're using temperature=1.0 with GPT-5 models
- GPT-5 enforces fixed decoding parameters

**API Key Issues:**
- Verify API keys in `.env` file are valid
- Check API key format (OpenAI: `sk-proj-...`, Anthropic: `sk-ant-...`)
- Ensure sufficient credits in accounts

**Docker networking issues:**
- Use `--network host` for accessing localhost services
- For detector API on host: set `DETECTOR_API_URL=http://host.docker.internal:8000` (Mac/Windows)

**Prometheus metrics not appearing:**
- Verify Pushgateway URL is accessible: `curl http://localhost:9091`
- Ensure `prometheus-client` is installed: `pip install prometheus-client`

**Rate Limits:**
- The script includes retry logic for rate limits
- Consider reducing batch size or adding delays

---

## Development

### Project Structure
```
prompt_test_script/
├── test_detector_prompts.py    # Main evaluation script
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── sample_prompts.csv          # Example prompts
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local testing setup
├── .dockerignore              # Build optimization
└── test_results/              # Output directory
```

### Requirements
- Python 3.11+
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)
- Access to AI detector API

### Dependencies
```
openai>=1.0.0           # OpenAI API client (GPT-4o, GPT-5)
anthropic>=0.18.0       # Anthropic API client (Claude)
requests>=2.28.0        # HTTP requests
tqdm>=4.64.0           # Progress bars
python-dotenv>=1.0.0   # Environment variables
prometheus-client>=0.19.0  # Metrics export
```

### Exit Codes
- **0**: Successful execution
- **1**: Error occurred (API failure, validation error, test failure with `--exit-on-error`)
- **130**: Interrupted by user (Ctrl+C)

---

## Tips for Best Results

1. **Start Small**: Begin with a few samples to test your setup
2. **Cost Monitoring**: Monitor API usage, especially with GPT-4/5 models
3. **Diverse Prompts**: Use varied prompts to test different content types
4. **Parameter Tuning**: Experiment with temperature (0.7-0.9) and max_tokens (400-600) for natural text
5. **Style Variation**: Mix formal/casual styles and different domains
6. **Baseline Testing**: Include human reference texts for comparison

---