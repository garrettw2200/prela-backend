# Railway Deployment Guide

Complete guide for deploying Prela backend services to Railway with ClickHouse Cloud and Upstash Kafka.

## Prerequisites

1. **Railway Account** - Sign up at https://railway.app
2. **ClickHouse Cloud Account** - Sign up at https://clickhouse.com/cloud
3. **Upstash Account** - Sign up at https://upstash.com
4. **Railway CLI** (optional) - `npm install -g @railway/cli`

## Architecture Overview

```
Railway Services:
├── prela-ingest-gateway  (public URL)
├── prela-api-gateway     (public URL)
├── prela-trace-service   (private worker)
├── Railway Postgres      (managed)
└── Railway Redis         (managed)

External Services:
├── ClickHouse Cloud      (trace/span storage)
└── Upstash Kafka         (message queue)
```

## Step 1: Setup ClickHouse Cloud

### Create Service

1. Go to https://clickhouse.com/cloud
2. Click "Create Service"
3. Choose:
   - **Region**: Choose closest to your Railway region
   - **Tier**: Development (free) or Production
   - **Name**: prela-clickhouse

4. Wait for service to provision (~2 minutes)

### Get Connection Details

After service is ready:

1. Click "Connect"
2. Note these values:
   - **Host**: `xxxxxxxx.clickhouse.cloud`
   - **Port**: `8443` (HTTPS)
   - **User**: `default`
   - **Password**: (shown once, save it!)
   - **Database**: `default` (we'll use `prela`)

### Test Connection

```bash
# Using clickhouse-client (if installed)
clickhouse-client --host xxxxxxxx.clickhouse.cloud \
  --port 9440 \
  --secure \
  --user default \
  --password YOUR_PASSWORD \
  --query "SELECT 1"

# Or using curl
curl 'https://xxxxxxxx.clickhouse.cloud:8443/?query=SELECT%201' \
  --user 'default:YOUR_PASSWORD'
```

## Step 2: Setup Upstash Kafka

### Create Cluster

1. Go to https://console.upstash.com/kafka
2. Click "Create Cluster"
3. Choose:
   - **Name**: prela-kafka
   - **Region**: Choose closest to your Railway region
   - **Type**: Pay as you go (free tier available)

4. Wait for cluster to provision (~1 minute)

### Create Topics

1. In cluster dashboard, go to "Topics"
2. Create two topics:
   - **Name**: `traces`
     - **Partitions**: 3
     - **Retention**: 7 days
   - **Name**: `spans`
     - **Partitions**: 6
     - **Retention**: 7 days

### Get Connection Details

1. In cluster dashboard, go to "Details"
2. Note these values:
   - **Bootstrap Servers**: `xxxxxxxx.upstash.io:9092`
   - **Username**: (copy from UI)
   - **Password**: (copy from UI)
   - **SASL Mechanism**: SCRAM-SHA-256
   - **Security Protocol**: SASL_SSL

### Test Connection

```bash
# Using kafkacat/kcat (if installed)
echo "test" | kcat -P \
  -b xxxxxxxx.upstash.io:9092 \
  -t traces \
  -X security.protocol=SASL_SSL \
  -X sasl.mechanisms=SCRAM-SHA-256 \
  -X sasl.username=YOUR_USERNAME \
  -X sasl.password=YOUR_PASSWORD
```

## Step 3: Setup Railway Project

### Option A: Railway Dashboard (Recommended)

1. Go to https://railway.app/new
2. Select "Empty Project"
3. Name it: `prela-backend`

### Option B: Railway CLI

```bash
railway login
railway init
railway project create prela-backend
```

## Step 4: Add Railway Services

### Add Postgres

1. In Railway dashboard, click "+ New"
2. Select "Database" → "PostgreSQL"
3. Name: `prela-postgres`
4. Wait for provisioning

Railway provides these variables automatically:
- `DATABASE_URL`
- `POSTGRES_HOST`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`

### Add Redis

1. Click "+ New"
2. Select "Database" → "Redis"
3. Name: `prela-redis`
4. Wait for provisioning

Railway provides these variables automatically:
- `REDIS_URL`
- `REDIS_HOST`
- `REDIS_PORT`

## Step 5: Deploy Ingest Gateway

### Create Service

1. Click "+ New" → "GitHub Repo"
2. Connect your Prela repository
3. Configure service:
   - **Name**: `prela-ingest-gateway`
   - **Root Directory**: `backend/services/ingest-gateway`
   - **Build Command**: (auto-detected from Dockerfile)

### Set Environment Variables

Go to service settings → "Variables" and add:

```bash
# Service Config
SERVICE_NAME=prela-ingest-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Upstash Kafka (from Step 2)
KAFKA_BOOTSTRAP_SERVERS=xxxxxxxx.upstash.io:9092
KAFKA_USERNAME=your-upstash-username
KAFKA_PASSWORD=your-upstash-password
KAFKA_TOPIC_TRACES=traces
KAFKA_TOPIC_SPANS=spans

# CORS (add your frontend domain later)
CORS_ORIGINS=["https://your-frontend-domain.com"]
```

### Deploy

1. Click "Deploy"
2. Wait for build and deployment (~3-5 minutes)
3. Railway will assign a public URL: `https://prela-ingest-gateway-xxx.railway.app`
4. Test: `curl https://your-url.railway.app/health`

## Step 6: Deploy Trace Service

### Create Service

1. Click "+ New" → "GitHub Repo"
2. Select Prela repository
3. Configure service:
   - **Name**: `prela-trace-service`
   - **Root Directory**: `backend/services/trace-service`
   - **Build Command**: (auto-detected from Dockerfile)

### Set Environment Variables

```bash
# Service Config
SERVICE_NAME=prela-trace-service
ENVIRONMENT=production
LOG_LEVEL=INFO

# ClickHouse Cloud (from Step 1)
CLICKHOUSE_HOST=xxxxxxxx.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_DATABASE=prela

# Upstash Kafka (from Step 2)
KAFKA_BOOTSTRAP_SERVERS=xxxxxxxx.upstash.io:9092
KAFKA_USERNAME=your-upstash-username
KAFKA_PASSWORD=your-upstash-password
KAFKA_TOPIC_TRACES=traces
KAFKA_TOPIC_SPANS=spans
```

### Deploy

1. Click "Deploy"
2. Wait for deployment
3. Check logs to verify Kafka consumer started

## Step 7: Deploy API Gateway

### Create Service

1. Click "+ New" → "GitHub Repo"
2. Select Prela repository
3. Configure service:
   - **Name**: `prela-api-gateway`
   - **Root Directory**: `backend/services/api-gateway`
   - **Build Command**: (auto-detected from Dockerfile)

### Set Environment Variables

```bash
# Service Config
SERVICE_NAME=prela-api-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (from Step 4)
DATABASE_URL=${{Postgres.DATABASE_URL}}

# ClickHouse Cloud (from Step 1)
CLICKHOUSE_HOST=xxxxxxxx.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_DATABASE=prela

# Upstash Kafka (for future features)
KAFKA_BOOTSTRAP_SERVERS=xxxxxxxx.upstash.io:9092
KAFKA_USERNAME=your-upstash-username
KAFKA_PASSWORD=your-upstash-password

# Redis (from Step 4)
REDIS_URL=${{Redis.REDIS_URL}}

# Auth
JWT_SECRET=<generate-with-openssl-rand-hex-32>

# CORS (add your frontend domain)
CORS_ORIGINS=["https://your-frontend-domain.com"]
```

### Deploy

1. Click "Deploy"
2. Wait for deployment
3. Railway will assign a public URL: `https://prela-api-gateway-xxx.railway.app`
4. Test: `curl https://your-url.railway.app/api/v1/health`

## Step 8: Configure SDK

Update your Prela SDK configuration to use the Railway endpoints:

### Environment Variables (for your application)

```bash
# In your application that uses Prela SDK
PRELA_INGEST_URL=https://prela-ingest-gateway-xxx.railway.app
PRELA_API_URL=https://prela-api-gateway-xxx.railway.app
```

### SDK Configuration (future - not yet implemented)

```python
import prela

prela.init(
    service_name="my-app",
    exporter="http",
    http_endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces"
)
```

## Step 9: Verify End-to-End

### 1. Send Test Trace

```bash
curl -X POST https://prela-ingest-gateway-xxx.railway.app/v1/traces \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "test-production-123",
    "service_name": "test-service",
    "started_at": "2025-01-27T10:00:00Z",
    "completed_at": "2025-01-27T10:00:01Z",
    "duration_ms": 1000,
    "status": "SUCCESS",
    "spans": [
      {
        "span_id": "span-1",
        "trace_id": "test-production-123",
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

Expected: `{"status": "accepted", "trace_id": "test-production-123"}`

### 2. Check Kafka (Upstash Console)

1. Go to Upstash console
2. Select your cluster
3. Go to "Messages"
4. Select `traces` topic
5. Verify message appears

### 3. Check ClickHouse

```bash
# Query via ClickHouse Cloud console
SELECT * FROM prela.traces ORDER BY started_at DESC LIMIT 10;
SELECT * FROM prela.spans ORDER BY started_at DESC LIMIT 10;
```

### 4. Query via API

```bash
# List traces
curl https://prela-api-gateway-xxx.railway.app/api/v1/traces

# Get specific trace
curl https://prela-api-gateway-xxx.railway.app/api/v1/traces/test-production-123
```

## Step 10: Monitoring

### Railway Metrics

Each service in Railway dashboard shows:
- CPU usage
- Memory usage
- Network traffic
- Logs (real-time and historical)

### Check Logs

```bash
# Using Railway CLI
railway logs --service prela-ingest-gateway
railway logs --service prela-api-gateway
railway logs --service prela-trace-service
```

### Set Up Alerts (Optional)

1. Railway dashboard → Service → Settings → Alerts
2. Configure alerts for:
   - High CPU usage (>80%)
   - High memory usage (>80%)
   - Deployment failures

## Troubleshooting

### Ingest Gateway Returns 500

**Check logs:**
```bash
railway logs --service prela-ingest-gateway
```

**Common issues:**
- Kafka credentials incorrect → verify KAFKA_USERNAME and KAFKA_PASSWORD
- Kafka broker unreachable → check KAFKA_BOOTSTRAP_SERVERS format
- SSL/TLS issues → ensure security_protocol=SASL_SSL

### Trace Service Not Consuming

**Check logs:**
```bash
railway logs --service prela-trace-service
```

**Common issues:**
- ClickHouse connection failed → verify CLICKHOUSE_HOST and CLICKHOUSE_PASSWORD
- Kafka consumer not starting → check Kafka credentials
- Schema initialization failed → verify ClickHouse database exists

### API Gateway Can't Query Traces

**Check logs:**
```bash
railway logs --service prela-api-gateway
```

**Common issues:**
- ClickHouse query timeout → increase query timeout
- Table not found → verify Trace Service initialized schema
- Connection pool exhausted → increase pool size

### High Kafka Consumer Lag

**Solutions:**
1. Scale Trace Service horizontally:
   - Railway dashboard → Trace Service → Settings → Scale
   - Increase replicas to 2-3
2. Increase batch size in consumer code
3. Optimize ClickHouse insert performance

## Cost Optimization

### Railway (~$25-45/month)
- **Dev**: Use Hobby plan ($5/service)
- **Production**: Monitor usage, scale as needed
- **Optimization**: Use cron jobs instead of always-on for non-critical services

### ClickHouse Cloud (~$0-50/month)
- **Storage**: ~$0.11/GB/month
- **Compute**: Pay per query
- **Optimization**:
  - Set TTL (90 days by default)
  - Use compression
  - Partition by month

### Upstash Kafka (~$0-20/month)
- **Free tier**: 10K messages/day
- **Paid**: ~$0.40/million messages
- **Optimization**:
  - Batch spans instead of individual messages
  - Reduce retention period if possible
  - Use compression

## Authentication Strategy

Prela uses **Clerk** for user authentication and authorization across the platform.

### Why Clerk?

- **Frontend + Backend Auth**: Seamless integration with React dashboard and FastAPI
- **API Key Management**: Clerk metadata stores project-scoped API keys
- **JWT Validation**: Clerk provides JWT tokens that backends verify
- **Multi-tenant Ready**: Clerk organizations map to Prela projects
- **Minimal Code**: Drop-in components for login, signup, user management

### Clerk Setup (Required Before Launch)

1. **Create Clerk Account**: https://clerk.com
2. **Create Application**: Name it "Prela" and choose "React" as framework
3. **Get API Keys**:
   - `CLERK_PUBLISHABLE_KEY` (for frontend)
   - `CLERK_SECRET_KEY` (for backend verification)

4. **Configure Clerk Application**:
   - Enable email/password authentication
   - Enable organizations (for multi-project support)
   - Configure JWT template with custom claims

### Integration Points

**Frontend (React Dashboard):**
```tsx
import { ClerkProvider, SignIn, useAuth } from '@clerk/clerk-react';

// Wrap app with Clerk provider
<ClerkProvider publishableKey={process.env.CLERK_PUBLISHABLE_KEY}>
  <App />
</ClerkProvider>
```

**API Gateway (FastAPI):**
```python
from clerk_backend_api import Clerk

clerk = Clerk(bearer_auth=os.environ["CLERK_SECRET_KEY"])

async def verify_token(request: Request):
    """Verify Clerk JWT token and extract project_id."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    session = clerk.verify_token(token)
    project_id = session.public_user_data.metadata.get("project_id")
    return project_id
```

**Ingest Gateway (API Key):**
```python
# SDK uses API keys (stored in Clerk user metadata)
# Format: pk_<project_id>_<random>
# Backend validates against Clerk metadata
```

### Clerk Configuration for Railway

Add to API Gateway environment variables:
```bash
CLERK_SECRET_KEY=sk_live_xxxxx
CLERK_PUBLISHABLE_KEY=pk_live_xxxxx  # For CORS/validation
```

Add to Frontend environment variables:
```bash
VITE_CLERK_PUBLISHABLE_KEY=pk_live_xxxxx
```

### API Key Generation Flow

1. User signs up via Clerk in React dashboard
2. User creates project → Clerk organization created
3. User generates API key → Stored in Clerk user metadata:
   ```json
   {
     "project_id": "proj_abc123",
     "api_keys": {
       "pk_proj_abc123_xyzabc": {
         "name": "Production Key",
         "created_at": "2026-01-27T10:00:00Z",
         "last_used": "2026-01-27T12:30:00Z"
       }
     }
   }
   ```
4. SDK sends API key in `X-API-Key` header
5. Ingest Gateway validates against Clerk metadata

### Benefits Over Custom Auth

- ✅ **No Auth Code**: Clerk handles signup, login, password reset, MFA
- ✅ **Security**: Industry-standard JWT validation
- ✅ **Compliance**: Clerk handles GDPR, session management
- ✅ **User Management**: Admin dashboard for managing users
- ✅ **Fast Integration**: ~1 hour to integrate vs. weeks for custom

### Cost

- **Free Tier**: 10,000 MAU (monthly active users)
- **Pro Tier**: $25/month for unlimited MAU
- **Perfect for MVP**: Start free, upgrade when scaling

## Next Steps

1. **Setup Clerk**: Create account and configure application (1 hour)
2. **Integrate Frontend**: Add ClerkProvider to React dashboard (2 hours)
3. **Integrate Backend**: Add JWT verification to API Gateway (2 hours)
4. **API Key System**: Build API key generation and validation (3 hours)
5. **Setup Monitoring**: Integrate Sentry or DataDog
6. **Add Alerting**: Configure alerts for errors and performance
7. **Scale**: Add load balancers and horizontal scaling as needed

## Support

- **Railway Docs**: https://docs.railway.app
- **ClickHouse Docs**: https://clickhouse.com/docs
- **Upstash Docs**: https://upstash.com/docs/kafka

---

**Deployment Checklist:**

- [ ] ClickHouse Cloud service created and tested
- [ ] Upstash Kafka cluster created with topics
- [ ] Railway project created
- [ ] Railway Postgres added
- [ ] Railway Redis added
- [ ] **Clerk account created and configured**
- [ ] **Clerk API keys added to Railway environment**
- [ ] Ingest Gateway deployed and healthy
- [ ] Trace Service deployed and consuming
- [ ] API Gateway deployed and querying
- [ ] **API Gateway JWT verification implemented**
- [ ] **Frontend Clerk integration implemented**
- [ ] End-to-end test successful (with authentication)
- [ ] Monitoring and alerts configured
- [ ] SDK configured to use production endpoints
