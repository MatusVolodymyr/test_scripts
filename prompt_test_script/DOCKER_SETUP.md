# Production-Like Docker Setup with Grafana Mimir

This Docker Compose setup emulates a production environment with:
- **Grafana Mimir** - Modern metrics backend (replaces Pushgateway)
- **Grafana** - Visualization dashboard
- **AI Detector Evaluation** - Your test script with remote write support

## Quick Start

### 1. Prerequisites
```bash
# Make sure you have Docker and Docker Compose installed
docker --version
docker-compose --version

# Configure your API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### 2. Start the Stack
```bash
docker-compose up
```

This will start:
- **Mimir** on http://localhost:9009
- **Grafana** on http://localhost:3000 (username: admin, password: admin)
- **AI Detector Eval** - Runs once and exits

### 3. View Results

**Check Grafana Dashboard:**
1. Open http://localhost:3000
2. Login with admin/admin
3. Go to Explore
4. Select "Mimir" datasource
5. Query metrics: `ai_detector_eval_*`

**Check Local Files:**
```bash
ls -lh test_results/
# You'll see:
# - generated_test_data_*.csv
# - prompt_test_report_*.json
# - results_*.jsonl
# - manifest_*.json
```

## Architecture

```
┌─────────────────────┐
│  AI Detector Eval   │
│   (Your Script)     │
└──────────┬──────────┘
           │ Remote Write
           │ (Prometheus Protocol)
           ▼
┌─────────────────────┐
│   Grafana Mimir     │
│  (Metrics Backend)  │
└──────────┬──────────┘
           │ Query API
           ▼
┌─────────────────────┐
│      Grafana        │
│  (Visualization)    │
└─────────────────────┘
```

## Metrics Exported

The script exports 9 metrics to Mimir:

1. `ai_detector_eval_success` - Success status (1/0)
2. `ai_detector_eval_run_timestamp_seconds` - Run timestamp
3. `ai_detector_eval_duration_seconds` - Run duration
4. `ai_detector_eval_samples_total{label="ai|human"}` - Sample counts
5. `ai_detector_eval_confusion_matrix{kind="tp|tn|fp|fn"}` - Confusion matrix
6. `ai_detector_eval_accuracy` - Accuracy score
7. `ai_detector_eval_precision` - Precision score
8. `ai_detector_eval_recall` - Recall score
9. `ai_detector_eval_f1` - F1 score

## Example Grafana Queries

```promql
# Current accuracy
ai_detector_eval_accuracy

# Success rate over time
rate(ai_detector_eval_success[5m])

# Confusion matrix visualization
sum by (kind) (ai_detector_eval_confusion_matrix)

# Sample distribution
sum by (label) (ai_detector_eval_samples_total)

# F1 score trend
ai_detector_eval_f1
```

## Running Multiple Tests

To run the evaluation multiple times:

```bash
# Run once
docker-compose up ai-detector-eval

# Run again with different parameters
RUN_ID=test-$(date +%s) DATASET_VERSION=v2.0 docker-compose up ai-detector-eval
```

## Stopping the Stack

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Troubleshooting

**Metrics not appearing in Grafana:**
1. Check Mimir is healthy: `curl http://localhost:9009/ready`
2. Check eval container logs: `docker-compose logs ai-detector-eval`
3. Verify remote write URL: `echo $PROMETHEUS_REMOTE_WRITE_URL`

**Eval container fails:**
1. Check API keys in .env file
2. Check detector API is accessible
3. View logs: `docker-compose logs -f ai-detector-eval`

**Grafana can't connect to Mimir:**
1. Restart Grafana: `docker-compose restart grafana`
2. Check network: `docker-compose exec grafana ping mimir`

## Production Differences

This local setup vs. production:

| Component | Local (Docker) | Production (AWS) |
|-----------|---------------|------------------|
| Metrics Backend | Mimir (container) | Grafana Cloud / Self-hosted Mimir |
| Evaluation | Docker container | Lambda / ECS Fargate |
| Scheduling | Manual | EventBridge |
| Storage | Local volumes | S3 |
| Secrets | .env file | AWS Secrets Manager |
| Authentication | None | IAM / API keys |

## Next Steps

To prepare for production:
1. Test S3 upload: Add `--s3-bucket` and `--s3-prefix` flags
2. Test with real detector API
3. Create Grafana dashboards for monitoring
4. Set up alerting rules in Mimir
5. Review DEPLOYMENT.md for AWS setup

## Clean Up

```bash
# Remove all containers, networks, and volumes
docker-compose down -v

# Remove test results
rm -rf test_results/*
```
