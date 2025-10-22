# MAIE Production Runbook

## 🚀 **Quick Start**

### Deploy with Helm (One Command)
```bash
# 1. Update image repository in values.yaml
# 2. Deploy everything
helm install maie ./helm/maie

# 3. Check status
kubectl get pods -l app.kubernetes.io/name=maie
kubectl get svc maie
```

### Deploy with Docker Compose (Local)
```bash
# Build and run locally
docker-compose up --build

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## 📊 **Daily Operations**

### **22:15 UTC: Automated Pipeline**
- ✅ **CronJob** runs `make bt-constrained && make report-html`
- ✅ Updates `expected/` with latest expected returns
- ✅ Generates HTML report in `outputs_from_expected/`

### **Health Checks**
```bash
# API Health
curl http://your-api/health
# Expected: {"status": "ok"}

# Readiness
curl http://your-api/ready
# Expected: {"status": "ready", "model_loaded": "true/false", "expected_available": "true/false"}

# Metrics
curl http://your-api/metrics
# Expected: Prometheus metrics format
```

### **Key Metrics to Monitor**
- **Request latency p95**: Should be < 200ms for `/score_expected`
- **Error rate**: Should be < 1%
- **Expected panel freshness**: `maie_expected_latest_timestamp` should be < 36h old

## 🚨 **Troubleshooting**

### **If `/ready` returns "not_ready"**
```bash
# 1. Check PVC mounts
kubectl describe pod -l app.kubernetes.io/name=maie

# 2. Verify expected panel exists
kubectl exec -it <pod-name> -- ls -la /data/expected/

# 3. Rebuild expected panel
kubectl exec -it <pod-name> -- make build-expected

# 4. Restart deployment
kubectl rollout restart deployment/maie
```

### **If `/explain_local` has high latency**
```bash
# 1. Warm the explainer cache
curl -X POST http://your-api/explain_local \
  -H "Content-Type: application/json" \
  -d '{"prices": {"SIM0001": [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159]}, "ticker": "SIM0001", "top_k": 5}'

# 2. If still slow, toggle to expected-only mode
kubectl set env deployment/maie READINESS_REQUIRE_MODEL=false
```

### **If CronJob fails**
```bash
# 1. Check CronJob status
kubectl get cronjob maie-daily

# 2. Check recent jobs
kubectl get jobs -l app.kubernetes.io/name=maie

# 3. Check job logs
kubectl logs -l job-name=maie-daily-<timestamp>

# 4. Manual run
kubectl create job --from=cronjob/maie-daily maie-daily-manual-$(date +%s)
```

## 🔧 **Maintenance**

### **Backups**
```bash
# Daily backups (automated)
# - mlruns/ (model artifacts)
# - expected/ (expected returns)
# - outputs_from_expected/ (reports, last 90 days)
```

### **Updates**
```bash
# 1. Update image tag in values.yaml
# 2. Upgrade deployment
helm upgrade maie ./helm/maie

# 3. Verify rollout
kubectl rollout status deployment/maie
```

### **Scaling**
```bash
# Horizontal scaling
kubectl scale deployment maie --replicas=5

# Or update values.yaml and upgrade
helm upgrade maie ./helm/maie --set replicaCount=5
```

## 🔒 **Security**

### **Container Security**
- ✅ Non-root user (`app`)
- ✅ Read-only root filesystem (where possible)
- ✅ Minimal base image (`python:3.13-slim`)
- ✅ No secrets in images

### **Network Security**
- ✅ Internal service (ClusterIP)
- ✅ Ingress with TLS (if exposed)
- ✅ Network policies (if needed)

### **Secrets Management**
```bash
# Store API keys in Kubernetes secrets
kubectl create secret generic maie-secrets \
  --from-literal=polygon-api-key=your-key-here

# Reference in deployment
env:
  - name: POLYGON_API_KEY
    valueFrom:
      secretKeyRef:
        name: maie-secrets
        key: polygon-api-key
```

## 📈 **Monitoring & Alerting**

### **Prometheus Alerts**
- **MaieHighLatencyScoreExpected**: p95 > 200ms for 10m
- **MaieReadyFailing**: Readiness failing for 5m
- **MaieNoNewExpected**: Expected panel stale > 36h

### **Custom Metrics**
- `maie_expected_latest_timestamp`: Timestamp of latest expected panel
- `maie_model_loaded`: Boolean indicating model availability
- `maie_api_requests_total`: Request counter by endpoint
- `maie_api_request_duration_seconds`: Request duration histogram

## 🎯 **Performance Tuning**

### **API Performance**
- **Warm explainer cache**: Hit `/explain_local` once on startup
- **Enable model caching**: Set `READINESS_REQUIRE_MODEL=true` for faster responses
- **Resource limits**: CPU 1 core, Memory 2GB per pod

### **Pipeline Performance**
- **Parallel processing**: Multiple workers for feature building
- **Caching**: DuckDB for real data, Parquet for expected returns
- **Incremental updates**: Only rebuild changed components

## 🚀 **Production Checklist**

- [ ] GHCR publish workflow green
- [ ] Deployed with health/readiness probes
- [ ] Nightly CronJob running successfully
- [ ] Prometheus alerts configured
- [ ] Backups automated
- [ ] Security hardening applied
- [ ] Performance monitoring active
- [ ] Documentation updated
- [ ] Team trained on runbook

## 📞 **Emergency Contacts**

- **On-call rotation**: [Your team contacts]
- **Escalation**: [Manager contacts]
- **External dependencies**: [Vendor contacts]

---

**Last Updated**: $(date)
**Version**: 1.0.0
**Maintainer**: [Your team]
