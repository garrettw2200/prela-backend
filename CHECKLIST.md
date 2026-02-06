# Backend Deployment Checklist

Use this checklist to deploy Prela backend services to production.

## Pre-Deployment Setup

### ☐ 1. ClickHouse Cloud Setup
- [ ] Create ClickHouse Cloud account
- [ ] Provision new service (Development or Production tier)
- [ ] Note connection details:
  - [ ] Host: `__________.clickhouse.cloud`
  - [ ] Port: `8443`
  - [ ] User: `default`
  - [ ] Password: (saved securely)
  - [ ] Database: `prela`
- [ ] Test connection with curl or clickhouse-client
- [ ] Verify schema will be auto-created (handled by Trace Service)

### ☐ 2. Upstash Kafka Setup
- [ ] Create Upstash account
- [ ] Create new Kafka cluster
- [ ] Note connection details:
  - [ ] Bootstrap servers: `__________.upstash.io:9092`
  - [ ] Username: (saved)
  - [ ] Password: (saved)
  - [ ] SASL mechanism: `SCRAM-SHA-256`
  - [ ] Security protocol: `SASL_SSL`
- [ ] Create `traces` topic (3 partitions, 7 days retention)
- [ ] Create `spans` topic (6 partitions, 7 days retention)
- [ ] Test connection with kcat/kafkacat

### ☐ 3. Railway Project Setup
- [ ] Create Railway account
- [ ] Create new project: `prela-backend`
- [ ] Add Railway Postgres:
  - [ ] Run: `railway add postgresql`
  - [ ] Note `DATABASE_URL` variable
- [ ] Add Railway Redis:
  - [ ] Run: `railway add redis`
  - [ ] Note `REDIS_URL` variable

---

## Local Development Testing

### ☐ 4. Test Local Environment
- [ ] Copy `.env.example` to `.env`
- [ ] Fill in local values in `.env`:
  ```bash
  CLICKHOUSE_HOST=localhost
  CLICKHOUSE_PORT=8123
  KAFKA_BOOTSTRAP_SERVERS=localhost:19092
  # ... etc
  ```
- [ ] Start infrastructure:
  ```bash
  docker-compose -f docker-compose.local.yml up -d
  ```
- [ ] Verify all containers running:
  - [ ] `docker ps` shows 5 containers (postgres, clickhouse, redis, redpanda, console)
  - [ ] PostgreSQL on `localhost:5432`
  - [ ] ClickHouse on `localhost:8123`
  - [ ] Redis on `localhost:6379`
  - [ ] Redpanda on `localhost:19092`
  - [ ] Redpanda Console on `http://localhost:8080`

### ☐ 5. Test Services Locally
- [ ] Start Ingest Gateway:
  ```bash
  cd services/ingest-gateway
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8001
  ```
- [ ] Verify health: `curl http://localhost:8001/health`
- [ ] Start API Gateway:
  ```bash
  cd services/api-gateway
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8002
  ```
- [ ] Verify health: `curl http://localhost:8002/api/v1/health`
- [ ] Start Trace Service:
  ```bash
  cd services/trace-service
  pip install -r requirements.txt
  python -m app.main
  ```
- [ ] Verify logs show Kafka consumers started

### ☐ 6. Test End-to-End Flow
- [ ] Send test trace:
  ```bash
  curl -X POST http://localhost:8001/v1/traces \
    -H "Content-Type: application/json" \
    -d '{"trace_id": "test-123", "service_name": "test", ...}'
  ```
- [ ] Check Kafka topic in Redpanda Console (`http://localhost:8080`)
- [ ] Verify trace appears in ClickHouse:
  ```sql
  SELECT * FROM prela.traces ORDER BY started_at DESC LIMIT 10;
  ```
- [ ] Query via API Gateway:
  ```bash
  curl http://localhost:8002/api/v1/traces
  curl http://localhost:8002/api/v1/traces/test-123
  ```

---

## Railway Deployment

### ☐ 7. Deploy Ingest Gateway
- [ ] In Railway dashboard, click "+ New" → "GitHub Repo"
- [ ] Select Prela repository
- [ ] Configure service:
  - [ ] Name: `prela-ingest-gateway`
  - [ ] Root Directory: `backend/services/ingest-gateway`
  - [ ] Build: Auto-detected (Dockerfile)
- [ ] Set environment variables:
  ```
  SERVICE_NAME=prela-ingest-gateway
  ENVIRONMENT=production
  LOG_LEVEL=INFO
  KAFKA_BOOTSTRAP_SERVERS=__________.upstash.io:9092
  KAFKA_USERNAME=__________
  KAFKA_PASSWORD=__________
  KAFKA_TOPIC_TRACES=traces
  KAFKA_TOPIC_SPANS=spans
  CORS_ORIGINS=["https://your-frontend-domain.com"]
  ```
- [ ] Click "Deploy"
- [ ] Wait for deployment to complete
- [ ] Note public URL: `https://prela-ingest-gateway-xxx.railway.app`
- [ ] Test health: `curl https://your-url.railway.app/health`

### ☐ 8. Deploy Trace Service
- [ ] In Railway dashboard, click "+ New" → "GitHub Repo"
- [ ] Select Prela repository
- [ ] Configure service:
  - [ ] Name: `prela-trace-service`
  - [ ] Root Directory: `backend/services/trace-service`
  - [ ] Build: Auto-detected (Dockerfile)
- [ ] Set environment variables:
  ```
  SERVICE_NAME=prela-trace-service
  ENVIRONMENT=production
  LOG_LEVEL=INFO
  CLICKHOUSE_HOST=__________.clickhouse.cloud
  CLICKHOUSE_PORT=8443
  CLICKHOUSE_USER=default
  CLICKHOUSE_PASSWORD=__________
  CLICKHOUSE_DATABASE=prela
  KAFKA_BOOTSTRAP_SERVERS=__________.upstash.io:9092
  KAFKA_USERNAME=__________
  KAFKA_PASSWORD=__________
  KAFKA_TOPIC_TRACES=traces
  KAFKA_TOPIC_SPANS=spans
  ```
- [ ] Click "Deploy"
- [ ] Wait for deployment to complete
- [ ] Check logs: Look for "Kafka consumer started" messages
- [ ] Verify ClickHouse schema created:
  ```sql
  SHOW TABLES FROM prela;  -- Should show 'traces' and 'spans'
  ```

### ☐ 9. Deploy API Gateway
- [ ] In Railway dashboard, click "+ New" → "GitHub Repo"
- [ ] Select Prela repository
- [ ] Configure service:
  - [ ] Name: `prela-api-gateway`
  - [ ] Root Directory: `backend/services/api-gateway`
  - [ ] Build: Auto-detected (Dockerfile)
- [ ] Set environment variables:
  ```
  SERVICE_NAME=prela-api-gateway
  ENVIRONMENT=production
  LOG_LEVEL=INFO
  DATABASE_URL=${{Postgres.DATABASE_URL}}
  CLICKHOUSE_HOST=__________.clickhouse.cloud
  CLICKHOUSE_PORT=8443
  CLICKHOUSE_USER=default
  CLICKHOUSE_PASSWORD=__________
  CLICKHOUSE_DATABASE=prela
  KAFKA_BOOTSTRAP_SERVERS=__________.upstash.io:9092
  KAFKA_USERNAME=__________
  KAFKA_PASSWORD=__________
  REDIS_URL=${{Redis.REDIS_URL}}
  JWT_SECRET=__________ (generate with: openssl rand -hex 32)
  CORS_ORIGINS=["https://your-frontend-domain.com"]
  ```
- [ ] Click "Deploy"
- [ ] Wait for deployment to complete
- [ ] Note public URL: `https://prela-api-gateway-xxx.railway.app`
- [ ] Test health: `curl https://your-url.railway.app/api/v1/health`

---

## Production Verification

### ☐ 10. End-to-End Test in Production
- [ ] Send test trace to production Ingest Gateway:
  ```bash
  curl -X POST https://prela-ingest-gateway-xxx.railway.app/v1/traces \
    -H "Content-Type: application/json" \
    -d '{
      "trace_id": "prod-test-123",
      "service_name": "test-service",
      "started_at": "2025-01-27T10:00:00Z",
      "completed_at": "2025-01-27T10:00:01Z",
      "duration_ms": 1000,
      "status": "SUCCESS",
      "spans": [
        {
          "span_id": "span-1",
          "trace_id": "prod-test-123",
          "name": "test-operation",
          "span_type": "LLM",
          "started_at": "2025-01-27T10:00:00Z",
          "ended_at": "2025-01-27T10:00:01Z",
          "duration_ms": 1000,
          "status": "SUCCESS",
          "attributes": {"model": "gpt-4"},
          "events": []
        }
      ]
    }'
  ```
- [ ] Verify response: `{"status": "accepted", "trace_id": "prod-test-123"}`
- [ ] Check Upstash Kafka Console:
  - [ ] Go to Upstash dashboard → Your cluster → Messages
  - [ ] Select `traces` topic
  - [ ] Verify message appears
- [ ] Check ClickHouse Cloud:
  - [ ] Open ClickHouse Cloud console
  - [ ] Run: `SELECT * FROM prela.traces ORDER BY started_at DESC LIMIT 10;`
  - [ ] Verify trace appears
- [ ] Query via API Gateway:
  ```bash
  curl https://prela-api-gateway-xxx.railway.app/api/v1/traces
  curl https://prela-api-gateway-xxx.railway.app/api/v1/traces/prod-test-123
  ```
- [ ] Verify trace and spans returned correctly

### ☐ 11. Load Test (Optional)
- [ ] Use Apache Bench or wrk to send 1000 traces
- [ ] Monitor Railway metrics (CPU, memory, network)
- [ ] Check Kafka consumer lag in Upstash
- [ ] Verify all traces appear in ClickHouse
- [ ] Measure query performance from API Gateway

---

## Post-Deployment

### ☐ 12. Configure Monitoring
- [ ] Railway dashboard → Each service → Settings → Alerts
- [ ] Configure alerts:
  - [ ] CPU usage > 80%
  - [ ] Memory usage > 80%
  - [ ] Deployment failures
  - [ ] Service restarts
- [ ] Set up error tracking (optional):
  - [ ] Add Sentry to each service
  - [ ] Configure error notifications

### ☐ 13. Update SDK Configuration
- [ ] Update Prela SDK to use production endpoints:
  ```python
  # In your application code or environment variables
  PRELA_INGEST_URL=https://prela-ingest-gateway-xxx.railway.app
  PRELA_API_URL=https://prela-api-gateway-xxx.railway.app
  ```
- [ ] Test SDK integration:
  ```python
  import prela

  prela.init(
      service_name="my-app",
      exporter="http",
      http_endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces"
  )

  # Your AI agent code...
  ```

### ☐ 14. Documentation
- [ ] Update main README.md with production URLs
- [ ] Document environment variables in team wiki
- [ ] Share Railway project access with team
- [ ] Document on-call procedures

---

## Maintenance Tasks

### Weekly
- [ ] Review Railway usage and costs
- [ ] Check Kafka consumer lag
- [ ] Review error logs in Railway
- [ ] Verify ClickHouse TTL is working (data < 90 days)

### Monthly
- [ ] Review ClickHouse storage usage
- [ ] Optimize slow queries
- [ ] Update dependencies
- [ ] Review and adjust rate limits

---

## Rollback Plan

If deployment fails:

### Ingest Gateway Rollback
1. Railway dashboard → prela-ingest-gateway → Deployments
2. Click on previous successful deployment
3. Click "Redeploy"

### Trace Service Rollback
1. Railway dashboard → prela-trace-service → Deployments
2. Click on previous successful deployment
3. Click "Redeploy"

### API Gateway Rollback
1. Railway dashboard → prela-api-gateway → Deployments
2. Click on previous successful deployment
3. Click "Redeploy"

### Emergency Shutdown
If critical issue, pause services:
1. Railway dashboard → Service → Settings
2. Toggle "Service" switch to pause
3. Investigate issue in logs
4. Fix and redeploy

---

## Support Contacts

- **Railway Support**: https://railway.app/help
- **ClickHouse Support**: https://clickhouse.com/support
- **Upstash Support**: https://upstash.com/docs

---

## Success Criteria

Deployment is successful when:

- ✅ All three services are running (green status in Railway)
- ✅ Health checks passing for Ingest Gateway and API Gateway
- ✅ Test trace flows through: Ingest → Kafka → ClickHouse → API
- ✅ No errors in Railway logs
- ✅ Kafka consumer lag is zero or minimal
- ✅ Query response time < 500ms for traces endpoint
- ✅ Monitoring and alerts configured

---

**Estimated Deployment Time**: 2-3 hours (first time)

**Cost After Deployment**: ~$25-115/month

**Status**: ☐ Not Started | ☐ In Progress | ☐ Completed | ☐ Verified
