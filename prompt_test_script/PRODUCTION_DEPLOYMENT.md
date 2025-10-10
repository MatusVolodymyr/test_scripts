# Production Deployment Guide

This guide shows how to deploy the AI Detector Evaluation tool in production environments.

## Prerequisites

- Docker installed and configured
- Access to your customer's Mimir endpoint
- API keys stored securely (AWS Secrets Manager, Kubernetes Secrets, etc.)
- Container registry access (ECR, Docker Hub, etc.)

---

## Quick Production Run

### 1. Build and Push Image

```bash
# Build the image
docker build -t ai-detector-eval:latest .

# Tag for your registry
docker tag ai-detector-eval:latest your-registry.example.com/ai-detector-eval:v1.0.0

# Push to registry
docker push your-registry.example.com/ai-detector-eval:v1.0.0
```

### 2. Run in Production

```bash
docker run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e DETECTOR_API_URL="https://detector.prod.example.com" \
  -e DETECTOR_API_TOKEN="${DETECTOR_API_TOKEN}" \
  -e PROMETHEUS_REMOTE_WRITE_URL="https://mimir.prod.example.com/api/v1/push" \
  -e PROMETHEUS_TENANT_ID="team-ml" \
  -e RUN_ID="eval-$(date +%s)" \
  -e DATASET_VERSION="v1.0" \
  -e GIT_COMMIT_SHA="$(git rev-parse HEAD)" \
  -e ENVIRONMENT="production" \
  -e CLUSTER="us-east-1" \
  -v /path/to/production-prompts.csv:/tmp/input/prompts.csv:ro \
  -v /path/to/output:/tmp/output \
  your-registry.example.com/ai-detector-eval:v1.0.0 \
  --input /tmp/input/prompts.csv \
  --generate-and-test \
  --save-jsonl \
  --save-manifest \
  --output-dir /tmp/output \
  --exit-on-error
```

---

## AWS ECS Deployment

### Task Definition Example

```json
{
  "family": "ai-detector-eval",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/aiDetectorEvalTaskRole",
  "containerDefinitions": [
    {
      "name": "ai-detector-eval",
      "image": "ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ai-detector-eval:latest",
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-detector-eval",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:openai-api-key"
        },
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:anthropic-api-key"
        },
        {
          "name": "DETECTOR_API_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:detector-api-token"
        }
      ],
      "environment": [
        {
          "name": "DETECTOR_API_URL",
          "value": "https://detector.prod.example.com"
        },
        {
          "name": "PROMETHEUS_REMOTE_WRITE_URL",
          "value": "https://mimir.prod.example.com/api/v1/push"
        },
        {
          "name": "PROMETHEUS_TENANT_ID",
          "value": "team-ml"
        },
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "CLUSTER",
          "value": "us-east-1"
        }
      ],
      "command": [
        "--input",
        "/tmp/input/prompts.csv",
        "--generate-and-test",
        "--save-jsonl",
        "--save-manifest",
        "--output-dir",
        "/tmp/output",
        "--exit-on-error"
      ]
    }
  ]
}
```

### Running the ECS Task

```bash
aws ecs run-task \
  --cluster ai-detector-eval-cluster \
  --task-definition ai-detector-eval:1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --overrides '{
    "containerOverrides": [{
      "name": "ai-detector-eval",
      "environment": [
        {"name": "RUN_ID", "value": "eval-'$(date +%s)'"},
        {"name": "DATASET_VERSION", "value": "v1.0"},
        {"name": "GIT_COMMIT_SHA", "value": "'$(git rev-parse HEAD)'"}
      ]
    }]
  }'
```

---

## Kubernetes Deployment

### ConfigMap for Environment Variables

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-detector-eval-config
  namespace: ml-evaluation
data:
  DETECTOR_API_URL: "https://detector.prod.example.com"
  PROMETHEUS_REMOTE_WRITE_URL: "https://mimir.prod.example.com/api/v1/push"
  PROMETHEUS_TENANT_ID: "team-ml"
  ENVIRONMENT: "production"
  CLUSTER: "k8s-prod-us-east-1"
```

### Secret for API Keys

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-detector-eval-secrets
  namespace: ml-evaluation
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  DETECTOR_API_TOKEN: <base64-encoded-token>
```

### Job Definition

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ai-detector-eval-job
  namespace: ml-evaluation
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: ai-detector-eval
        image: your-registry.example.com/ai-detector-eval:v1.0.0
        imagePullPolicy: Always
        
        envFrom:
        - configMapRef:
            name: ai-detector-eval-config
        - secretRef:
            name: ai-detector-eval-secrets
        
        env:
        - name: RUN_ID
          value: "eval-k8s-$(date +%s)"
        - name: DATASET_VERSION
          value: "v1.0"
        - name: GIT_COMMIT_SHA
          value: "abc123def"
        
        args:
        - "--input"
        - "/tmp/input/prompts.csv"
        - "--generate-and-test"
        - "--save-jsonl"
        - "--save-manifest"
        - "--output-dir"
        - "/tmp/output"
        - "--exit-on-error"
        
        volumeMounts:
        - name: input-data
          mountPath: /tmp/input
          readOnly: true
        - name: output-data
          mountPath: /tmp/output
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      
      volumes:
      - name: input-data
        configMap:
          name: evaluation-prompts
      - name: output-data
        emptyDir: {}
```

### Run the Job

```bash
kubectl apply -f ai-detector-eval-job.yaml
kubectl logs -f job/ai-detector-eval-job -n ml-evaluation
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Run AI Detector Evaluation

on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight
  workflow_dispatch:  # Allow manual trigger

jobs:
  evaluate:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/ai-detector-eval:$IMAGE_TAG .
        docker push $ECR_REGISTRY/ai-detector-eval:$IMAGE_TAG
    
    - name: Run evaluation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        DETECTOR_API_TOKEN: ${{ secrets.DETECTOR_API_TOKEN }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker run --rm \
          -e OPENAI_API_KEY \
          -e ANTHROPIC_API_KEY \
          -e DETECTOR_API_URL="${{ secrets.DETECTOR_API_URL }}" \
          -e DETECTOR_API_TOKEN \
          -e PROMETHEUS_REMOTE_WRITE_URL="${{ secrets.PROMETHEUS_REMOTE_WRITE_URL }}" \
          -e PROMETHEUS_TENANT_ID="${{ secrets.PROMETHEUS_TENANT_ID }}" \
          -e RUN_ID="eval-github-$(date +%s)" \
          -e DATASET_VERSION="v1.0" \
          -e GIT_COMMIT_SHA="${{ github.sha }}" \
          -e ENVIRONMENT="production" \
          -e CLUSTER="github-actions" \
          -v $(pwd)/prompts.csv:/tmp/input/prompts.csv:ro \
          -v $(pwd)/output:/tmp/output \
          ${{ steps.login-ecr.outputs.registry }}/ai-detector-eval:$IMAGE_TAG \
          --input /tmp/input/prompts.csv \
          --generate-and-test \
          --save-jsonl \
          --save-manifest \
          --output-dir /tmp/output \
          --exit-on-error
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: output/
```

---

## Grafana Configuration for Customers

**Your customer does NOT need to modify their existing Grafana setup.** They just need to:

### 1. Add Mimir Datasource (if not already configured)

Go to **Configuration → Data Sources → Add data source → Prometheus**

- **Name**: Mimir (or any name)
- **URL**: `https://mimir.prod.example.com/prometheus`
- **Custom HTTP Headers**:
  - Header: `X-Scope-OrgID`
  - Value: `team-ml` (their tenant ID)

### 2. Query Metrics in Their Existing Dashboards

Use these metric names:
```promql
ai_detector_eval_accuracy_1
ai_detector_eval_precision_1
ai_detector_eval_recall_1
ai_detector_eval_f1_1
ai_detector_eval_samples_total_1{label="ai"}
ai_detector_eval_samples_total_1{label="human"}
ai_detector_eval_confusion_matrix_1{kind="tp"}
ai_detector_eval_confusion_matrix_1{kind="tn"}
ai_detector_eval_confusion_matrix_1{kind="fp"}
ai_detector_eval_confusion_matrix_1{kind="fn"}
```

### 3. Filter by Environment/Cluster

```promql
ai_detector_eval_accuracy_1{environment="production"}
ai_detector_eval_accuracy_1{cluster="us-east-1"}
```

---

## Monitoring and Alerting

### Example Prometheus Alert Rules

```yaml
groups:
- name: ai_detector_eval
  interval: 5m
  rules:
  
  # Alert if evaluation fails
  - alert: EvaluationFailed
    expr: ai_detector_eval_success_1 == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "AI detector evaluation failed"
      description: "Evaluation run {{ $labels.run_id }} failed"
  
  # Alert if accuracy drops below threshold
  - alert: LowAccuracy
    expr: ai_detector_eval_accuracy_1 < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "AI detector accuracy is low"
      description: "Accuracy is {{ $value }}, below threshold of 0.8"
  
  # Alert if no metrics received
  - alert: NoMetrics
    expr: absent(ai_detector_eval_success_1)
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "No evaluation metrics received"
      description: "No evaluation has run in the last hour"
```

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "AI Detector Evaluation",
    "panels": [
      {
        "title": "Accuracy",
        "targets": [{
          "expr": "ai_detector_eval_accuracy_1"
        }],
        "type": "graph"
      },
      {
        "title": "Confusion Matrix",
        "targets": [{
          "expr": "sum by (kind) (ai_detector_eval_confusion_matrix_1)"
        }],
        "type": "bargauge"
      }
    ]
  }
}
```

---

## Troubleshooting Production Issues

### Metrics Not Appearing

1. **Check container logs**:
   ```bash
   # Docker
   docker logs <container-id>
   
   # Kubernetes
   kubectl logs -f pod/<pod-name>
   
   # ECS
   aws logs tail /ecs/ai-detector-eval --follow
   ```

2. **Verify Mimir connectivity**:
   ```bash
   curl -X POST https://mimir.prod.example.com/api/v1/push \
     -H "X-Scope-OrgID: team-ml" \
     -H "Content-Type: application/x-protobuf" \
     --data-binary @/dev/null -v
   ```

3. **Check Grafana datasource**:
   - Ensure `X-Scope-OrgID` header is configured
   - Test the connection in Grafana UI

### API Rate Limits

If you hit rate limits:
- Add delays between API calls (modify script)
- Use batch processing
- Implement exponential backoff (already included)

### High Memory Usage

If container runs out of memory:
- Increase memory limits in deployment config
- Process smaller batches
- Stream results to S3 instead of keeping in memory

---

## Cost Optimization

### Estimated Costs per Run

- **OpenAI GPT-4o**: ~$0.01 per sample
- **Anthropic Claude**: ~$0.008 per sample
- **Container compute**: $0.04-0.08 per run (1-2 vCPU for 5-10 minutes)

### Tips to Reduce Costs

1. Use cheaper models for testing (`gpt-4o-mini`, `claude-3-5-haiku`)
2. Cache generated texts and reuse for multiple detector tests
3. Use spot instances or preemptible VMs
4. Schedule runs during off-peak hours
5. Monitor API usage in real-time

---

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use secrets management** (AWS Secrets Manager, Kubernetes Secrets)
3. **Rotate API keys regularly**
4. **Use IAM roles** instead of static credentials when possible
5. **Enable CloudTrail/audit logging** for compliance
6. **Restrict network access** to only required endpoints
7. **Scan container images** for vulnerabilities before deployment

---

## Support

For production issues:
1. Check container logs first
2. Verify environment variables are set correctly
3. Test Mimir connectivity manually
4. Review Grafana datasource configuration
5. Contact your DevOps team for infrastructure issues

---

**Note**: This tool is production-ready. The local `docker-compose.yml` stack (Mimir + Grafana) is only for development. Your customer should use their existing infrastructure.
