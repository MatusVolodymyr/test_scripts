# Quick Reference for DevOps

## For Your Customer's Production Deployment

### What They Need

**Just the container** with these environment variables:

```bash
# API Keys (from their secrets manager)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
DETECTOR_API_TOKEN=their-token

# Endpoints
DETECTOR_API_URL=https://their-detector.example.com
PROMETHEUS_REMOTE_WRITE_URL=https://their-mimir.example.com/api/v1/push
PROMETHEUS_TENANT_ID=their-tenant-id

# Optional metadata
RUN_ID=eval-$(date +%s)
DATASET_VERSION=v1.0
ENVIRONMENT=production
CLUSTER=us-east-1
```

### What They DON'T Need

- ❌ The `mimir` service from docker-compose.yml
- ❌ The `grafana` service from docker-compose.yml
- ❌ The `grafana-datasource.yaml` file
- ❌ Any local setup scripts

**Their existing Grafana/Mimir works as-is!** No changes required.

---

## Metrics in Grafana

Metrics will appear automatically when the container runs.

### Query Examples

```promql
# Accuracy
ai_detector_eval_accuracy_1

# Filter by environment
ai_detector_eval_accuracy_1{environment="production"}

# Sample counts
sum by (label) (ai_detector_eval_samples_total_1)

# Confusion matrix
sum by (kind) (ai_detector_eval_confusion_matrix_1)
```

### All Available Metrics

1. `ai_detector_eval_success_1` - Success status (1/0)
2. `ai_detector_eval_run_timestamp_seconds_s` - Run timestamp
3. `ai_detector_eval_duration_seconds_s` - Duration
4. `ai_detector_eval_samples_total_1{label="ai"|"human"}` - Sample counts
5. `ai_detector_eval_confusion_matrix_1{kind="tp"|"tn"|"fp"|"fn"}` - Confusion matrix
6. `ai_detector_eval_accuracy_1` - Accuracy
7. `ai_detector_eval_precision_1` - Precision
8. `ai_detector_eval_recall_1` - Recall
9. `ai_detector_eval_f1_1` - F1 score

---

## Docker Commands

### Build
```bash
docker build -t ai-detector-eval:latest .
```

### Run (Production Example)
```bash
docker run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e DETECTOR_API_URL="https://detector.example.com" \
  -e DETECTOR_API_TOKEN="${DETECTOR_API_TOKEN}" \
  -e PROMETHEUS_REMOTE_WRITE_URL="https://mimir.example.com/api/v1/push" \
  -e PROMETHEUS_TENANT_ID="team-ml" \
  -v /path/to/prompts.csv:/tmp/input/prompts.csv:ro \
  -v $(pwd)/output:/tmp/output \
  ai-detector-eval:latest \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --output-dir /tmp/output
```

### Test Locally (with Mimir + Grafana)
```bash
docker-compose up
# View in Grafana: http://localhost:3000 (admin/admin)
```

---

## Key Files

| File | Purpose | Production? |
|------|---------|-------------|
| `Dockerfile` | Container image | ✅ Yes |
| `test_detector_prompts.py` | Main script | ✅ Yes |
| `requirements.txt` | Dependencies | ✅ Yes |
| `docker-compose.yml` | Local dev stack | ❌ No (dev only) |
| `mimir-config.yaml` | Mimir config | ❌ No (dev only) |
| `grafana-datasource.yaml` | Grafana datasource | ❌ No (dev only) |
| `README.md` | Documentation | 📖 Reference |
| `PRODUCTION_DEPLOYMENT.md` | Deploy guide | 📖 Reference |

---

## Troubleshooting

### Metrics not appearing?

1. **Check logs**: `docker logs <container-id>`
2. **Test Mimir connection**:
   ```bash
   curl -X POST https://mimir.example.com/api/v1/push \
     -H "X-Scope-OrgID: tenant-id" \
     -H "Content-Type: application/x-protobuf" \
     --data-binary @/dev/null -v
   ```
3. **Verify Grafana datasource** has X-Scope-OrgID header

### Container exits immediately?

- Check environment variables are set
- Verify input file path is mounted correctly
- Look at container logs: `docker logs <container-id>`

### API errors?

- Verify API keys are valid
- Check DETECTOR_API_URL is accessible
- Review rate limits

---

## Migration from v1.0

If upgrading from prometheus-client version:

1. **Environment variables changed**:
   - Remove: `PUSHGATEWAY_URL`
   - Add: `PROMETHEUS_REMOTE_WRITE_URL`
   - Add: `PROMETHEUS_TENANT_ID`

2. **Rebuild image**: `docker build -t ai-detector-eval:latest .`

3. **No data format changes**: Input CSV and output files unchanged

4. **Metrics names unchanged**: Same metric names (with `_1` suffix in Grafana)

See `CHANGELOG.md` for full details.

---

## Questions?

- **Architecture questions**: See `README.md`
- **Deployment questions**: See `PRODUCTION_DEPLOYMENT.md`
- **Changes**: See `CHANGELOG.md`
- **Issues**: Check container logs first
