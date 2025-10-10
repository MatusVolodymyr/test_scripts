# Changelog

## [2.0.0] - 2025-10-10

### 🚀 Major Changes - OpenTelemetry Migration

**BREAKING CHANGES**: Complete migration from `prometheus-client` to OpenTelemetry with Prometheus Remote Write.

#### Architecture Simplification
- ❌ **Removed**: Prometheus Pushgateway dependency
- ❌ **Removed**: Grafana Agent for metrics conversion
- ✅ **Added**: Native OpenTelemetry metrics export with protobuf encoding
- ✅ **Added**: Direct Prometheus Remote Write to Mimir/Cortex

**Before** (4 services):
```
Script → prometheus_client (text) → Pushgateway → Grafana Agent → Mimir
```

**After** (2 services for local dev):
```
Script → OpenTelemetry (protobuf) → Mimir
```

#### Metrics Export Changes
- **Removed**: `--pushgateway-url` CLI argument
- **Removed**: `PUSHGATEWAY_URL` environment variable
- **Added**: `PROMETHEUS_REMOTE_WRITE_URL` environment variable (Mimir endpoint)
- **Added**: `PROMETHEUS_TENANT_ID` environment variable (tenant ID)
- **Added**: Automatic export every 5 seconds via `PeriodicExportingMetricReader`
- **Added**: Resource attributes: `service.name`, `environment`, `cluster`

#### Dependencies
- **Removed**: `prometheus-client>=0.19.0`
- **Added**: `opentelemetry-api>=1.20.0`
- **Added**: `opentelemetry-sdk>=1.20.0`
- **Added**: `opentelemetry-exporter-prometheus-remote-write>=0.41b0`

#### Code Changes
- Complete rewrite of metrics export logic
- New `_init_otel_metrics()` method for OpenTelemetry setup
- New `export_metrics_to_mimir()` using observable gauges and counters
- Metrics now use callbacks for dynamic values
- Automatic export without manual flush

### 📦 Containerization Improvements

#### Production-Ready Configuration
- Updated Docker Compose with clear separation of local dev vs production
- Added comprehensive production deployment guide (`PRODUCTION_DEPLOYMENT.md`)
- Included ECS, Kubernetes, and CI/CD examples
- Added customer integration documentation

#### Documentation Overhaul
- ✅ **New**: `README.md` - Production-focused documentation
- ✅ **New**: `PRODUCTION_DEPLOYMENT.md` - Detailed deployment guide
- ❌ **Removed**: `QUICK_START.md` - Local setup guide (obsolete)
- ❌ **Removed**: `DOCKER_SETUP.md` - Docker setup guide (consolidated)
- ❌ **Removed**: Local helper scripts (`setup_and_run.sh`, `run_test.sh`, etc.)

#### Environment Variables
- Updated `.env.example` with new OpenTelemetry variables
- Removed old Pushgateway configuration
- Added optional metadata variables (`ENVIRONMENT`, `CLUSTER`)

### 🎯 Customer Integration

#### Simplified Deployment
- Customers **do NOT need** to deploy Grafana or Mimir from this repo
- Only the `ai-detector-eval` container is needed in production
- Works with customer's existing Grafana + Mimir infrastructure
- Just needs `PROMETHEUS_REMOTE_WRITE_URL` and `PROMETHEUS_TENANT_ID`

#### Local Development
- `docker-compose.yml` now clearly marked as LOCAL DEVELOPMENT ONLY
- Includes Mimir + Grafana for testing full metrics pipeline locally
- No production changes required to customer's infrastructure

### 📊 Metrics (No Changes)

All 9 metrics remain the same:
- `ai_detector_eval_success`
- `ai_detector_eval_run_timestamp_seconds`
- `ai_detector_eval_duration_seconds`
- `ai_detector_eval_samples_total{label="ai|human"}`
- `ai_detector_eval_confusion_matrix{kind="tp|tn|fp|fn"}`
- `ai_detector_eval_accuracy`
- `ai_detector_eval_precision`
- `ai_detector_eval_recall`
- `ai_detector_eval_f1`

**Note**: Metric names in Grafana will have `_1` suffix (e.g., `ai_detector_eval_accuracy_1`)

### 🔧 Technical Improvements

- **Performance**: Direct protobuf encoding (faster than text format)
- **Reliability**: Automatic retry with exponential backoff
- **Observability**: Resource attributes for better filtering
- **Simplicity**: 50% fewer services to maintain
- **Standards**: Modern OpenTelemetry API (industry standard)

### ⚠️ Migration Notes

**For existing users:**
1. Update environment variables:
   - Remove `PUSHGATEWAY_URL`
   - Add `PROMETHEUS_REMOTE_WRITE_URL`
   - Add `PROMETHEUS_TENANT_ID`
2. Rebuild Docker image: `docker build -t ai-detector-eval:latest .`
3. No changes needed to input data or output formats
4. Metrics will appear in Grafana with same names (plus `_1` suffix)

**For new users:**
- Follow the updated `README.md` for quick start
- Use `PRODUCTION_DEPLOYMENT.md` for production deployment
- Local testing: `docker-compose up` (includes Mimir + Grafana)

### 📝 Files Changed

**Modified:**
- `test_detector_prompts.py` - Complete metrics rewrite
- `requirements.txt` - OpenTelemetry dependencies
- `docker-compose.yml` - Simplified stack
- `.env.example` - Updated environment variables
- `README.md` - Production-focused documentation

**Added:**
- `PRODUCTION_DEPLOYMENT.md` - Deployment guide
- `CHANGELOG.md` - This file
- `grafana-datasource.yaml` - Grafana datasource config (local dev)

**Removed:**
- `QUICK_START.md` - Local setup guide
- `DOCKER_SETUP.md` - Docker guide
- `setup_and_run.sh` - Setup script
- `setup_and_run.bat` - Windows setup script
- `run_test.sh` - Test runner script
- `test_mimir_setup.sh` - Mimir test script
- `test_s3_remote_write.sh` - S3 test script

---

## [1.0.0] - 2025-09-24

Initial release with prometheus-client and Pushgateway support.
