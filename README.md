# Prela Backend

Backend infrastructure for Prela, organized into two categories:
- **`/services/`** - Core product services (api-gateway, ingest-gateway, trace-service)
- **`/agents/`** - Growth/DevRel AI agents (Scout, Sentinel, Amplifier, Radar)

> **📁 For agent documentation**, see [/internal/planning/COMMUNICATION_AGENTS_IMPLEMENTATION.md](../internal/planning/COMMUNICATION_AGENTS_IMPLEMENTATION.md) and [/internal/planning/REPOSITORY_STRUCTURE.md](../internal/planning/REPOSITORY_STRUCTURE.md)

---

## Product Services

Backend infrastructure for Prela observability platform, designed for deployment on Railway with ClickHouse Cloud and Upstash Kafka.

### Architecture

```
┌─────────────┐
│   Prela SDK │
└──────┬──────┘
       │ HTTP POST
       ▼
┌─────────────────┐      ┌──────────────┐
│ Ingest Gateway  │─────▶│ Upstash      │
│  (FastAPI)      │      │ Kafka        │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐      ┌──────────────┐
                         │ Trace Service│─────▶│ ClickHouse   │
                         │  (Worker)    │      │ Cloud        │
                         └──────────────┘      └──────────────┘
                                                       ▲
                                                       │
┌─────────────┐      ┌─────────────────┐            │
│  Frontend   │◀─────│  API Gateway    │────────────┘
│   (React)   │      │   (FastAPI)     │
└─────────────┘      └─────────────────┘
```

## Services

### 1. **Ingest Gateway** (Port 8000)
- **Purpose**: High-throughput ingestion endpoint for traces/spans
- **Technology**: FastAPI + Uvicorn
- **Responsibilities**:
  - Accept trace/span data from Prela SDK
  - Validate and forward to Kafka
  - Minimize latency and maximize throughput

**Endpoints:**
- `POST /v1/traces` - Ingest complete trace
- `POST /v1/spans` - Ingest single span
- `POST /v1/batch` - Ingest span batch (high efficiency)
- `GET /health` - Health check

### 2. **Trace Service** (Background Worker)
- **Purpose**: Process traces/spans from Kafka and write to ClickHouse
- **Technology**: Python asyncio + aiokafka
- **Responsibilities**:
  - Consume from Kafka topics (traces, spans)
  - Transform and insert into ClickHouse
  - Handle schema initialization

### 3. **API Gateway** (Port 8000)
- **Purpose**: Query API for frontend and external consumers
- **Technology**: FastAPI + Uvicorn
- **Responsibilities**:
  - Query traces and spans from ClickHouse
  - Serve frontend requests
  - Authentication and authorization (future)

**Endpoints:**
- `GET /api/v1/traces` - List traces with filters
- `GET /api/v1/traces/{trace_id}` - Get single trace with spans
- `GET /api/v1/traces/{trace_id}/spans` - List spans for trace
- `GET /api/v1/search` - Search traces by content
- `GET /api/v1/health` - Health check

### 4. **Shared Module**
- **Purpose**: Common utilities across services
- **Contents**:
  - `config.py` - Environment variable configuration (Pydantic)
  - `clickhouse.py` - ClickHouse Cloud client utilities
  - `kafka.py` - Upstash Kafka producer/consumer

## Tech Stack

### Cloud Services (Production)
- **Railway** - Application hosting (3 services)
- **ClickHouse Cloud** - Trace/span storage (columnar database)
- **Upstash Kafka** - Message queue (serverless Kafka)
- **Railway Postgres** - User data, auth (relational database)
- **Railway Redis** or **Upstash Redis** - Caching, sessions

### Local Development
- **PostgreSQL 16** - User data
- **ClickHouse 24.1** - Trace storage
- **Redis 7** - Cache
- **Redpanda** - Lightweight Kafka replacement
- **Redpanda Console** - Kafka UI (http://localhost:8080)

### Languages & Frameworks
- **Python 3.11+** - All services
- **FastAPI 0.109** - API framework
- **Uvicorn** - ASGI server
- **aiokafka** - Async Kafka client
- **clickhouse-connect** - ClickHouse client
- **Pydantic** - Configuration and validation

## Local Development

### Prerequisites
- Docker Desktop or Docker + Docker Compose
- Python 3.11+
- Make (optional, for convenience commands)

### Environment Setup

All backend services share a single `.env` configuration file located at `backend/.env`. This approach works because all services import from the shared `services/shared/config.py` module.

**Setup Steps:**

1. **Copy the example file:**
   ```bash
   cd backend
   cp .env.example .env
   ```

2. **Update required variables:**
   - For local development, the defaults in `.env.example` work with docker-compose
   - For Clerk authentication (API Gateway only): Set `CLERK_SECRET_KEY` and `CLERK_PUBLISHABLE_KEY`
   - For Stripe billing (API Gateway only): Set Stripe keys and price IDs from your Stripe dashboard

3. **Required vs Optional Variables:**

   **Required for all services:**
   - `DATABASE_URL` - PostgreSQL connection
   - `CLICKHOUSE_*` - ClickHouse connection details
   - `KAFKA_BOOTSTRAP_SERVERS` - Kafka/Redpanda connection
   - `REDIS_URL` - Redis connection

   **Required for API Gateway only:**
   - `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY` - Authentication
   - `STRIPE_SECRET_KEY` - Billing (if using billing features)

   **Optional:**
   - `SMTP_*` - Email notifications (can be left empty)
   - `STRIPE_WEBHOOK_SECRET` - Only needed after setting up webhooks

**Important:** The `.env` file is gitignored. Never commit secrets to version control.

### 1. Start Infrastructure

```bash
cd backend
docker-compose -f docker-compose.local.yml up -d
```

This starts:
- PostgreSQL on `localhost:5432`
- ClickHouse on `localhost:8123` (HTTP), `localhost:9000` (native)
- Redis on `localhost:6379`
- Redpanda (Kafka) on `localhost:19092`
- Redpanda Console on `http://localhost:8080`

### 2. Configure Environment Variables

If you haven't already, create your `.env` file as described in the [Environment Setup](#environment-setup) section above. The `.env` file at `backend/.env` is shared by all services.

### 3. Run Services Locally

**Option A: Direct Python (development)**

```bash
# Terminal 1: Ingest Gateway
cd services/ingest-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001

# Terminal 2: API Gateway
cd services/api-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002

# Terminal 3: Trace Service
cd services/trace-service
pip install -r requirements.txt
python -m app.main
```

**Option B: Docker Compose (closer to production)**

```bash
# TODO: Add docker-compose.yml for services
```

### 4. Verify Services

```bash
# Ingest Gateway
curl http://localhost:8001/health

# API Gateway
curl http://localhost:8002/api/v1/health

# Check Kafka topics in Redpanda Console
open http://localhost:8080
```

### 5. Send Test Data

```bash
# Send a test trace
curl -X POST http://localhost:8001/v1/traces \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "test-trace-123",
    "service_name": "test-service",
    "started_at": "2025-01-27T10:00:00Z",
    "completed_at": "2025-01-27T10:00:01Z",
    "duration_ms": 1000,
    "status": "SUCCESS",
    "spans": [
      {
        "span_id": "span-1",
        "trace_id": "test-trace-123",
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

# Query traces
curl http://localhost:8002/api/v1/traces

# Get specific trace
curl http://localhost:8002/api/v1/traces/test-trace-123
```

## Railway Deployment

### 1. Setup Railway Project

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project
railway init
```

### 2. Add Services

Create three Railway services:
1. **prela-ingest-gateway** (from `services/ingest-gateway`)
2. **prela-api-gateway** (from `services/api-gateway`)
3. **prela-trace-service** (from `services/trace-service`)

### 3. Add Cloud Services

**ClickHouse Cloud:**
1. Sign up at https://clickhouse.com/cloud
2. Create a service
3. Get connection details (host, port, user, password)
4. Add to Railway environment variables

**Upstash Kafka:**
1. Sign up at https://upstash.com
2. Create a Kafka cluster
3. Get bootstrap servers and SASL credentials
4. Add to Railway environment variables

**Railway Postgres:**
```bash
railway add postgresql
```

**Railway Redis:**
```bash
railway add redis
```

### 4. Configure Environment Variables

For each service, add in Railway dashboard:

**Common Variables:**
```
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
JWT_SECRET=<generate-secure-secret>
```

**ClickHouse Cloud:**
```
CLICKHOUSE_HOST=<your-instance>.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=<your-password>
CLICKHOUSE_DATABASE=prela
```

**Upstash Kafka:**
```
KAFKA_BOOTSTRAP_SERVERS=<your-cluster>.upstash.io:9092
KAFKA_USERNAME=<your-username>
KAFKA_PASSWORD=<your-password>
KAFKA_TOPIC_TRACES=traces
KAFKA_TOPIC_SPANS=spans
```

**API Gateway only:**
```
CORS_ORIGINS=["https://your-frontend-domain.com"]
```

### 5. Deploy

```bash
# Deploy all services
railway up
```

Railway will:
1. Build Docker images from Dockerfiles
2. Deploy services
3. Expose public URLs for API/Ingest gateways
4. Set up health checks

### 6. Verify Deployment

```bash
# Check service health
curl https://your-ingest-gateway.railway.app/health
curl https://your-api-gateway.railway.app/api/v1/health

# Test ingestion
curl -X POST https://your-ingest-gateway.railway.app/v1/traces \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "test", ...}'
```

## ClickHouse Schema

The schema is automatically initialized by the Trace Service on startup:

**Traces Table:**
```sql
CREATE TABLE traces (
    trace_id String,
    service_name String,
    started_at DateTime64(6),
    completed_at DateTime64(6),
    duration_ms Float64,
    status String,
    root_span_id String,
    span_count UInt32,
    metadata String,
    created_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
ORDER BY (service_name, started_at, trace_id)
PARTITION BY toYYYYMM(started_at)
TTL started_at + INTERVAL 90 DAY
```

**Spans Table:**
```sql
CREATE TABLE spans (
    span_id String,
    trace_id String,
    parent_span_id String,
    name String,
    span_type String,
    service_name String,
    started_at DateTime64(6),
    ended_at DateTime64(6),
    duration_ms Float64,
    status String,
    attributes String,
    events String,
    created_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
ORDER BY (trace_id, started_at, span_id)
PARTITION BY toYYYYMM(started_at)
TTL started_at + INTERVAL 90 DAY
```

**Features:**
- Monthly partitioning for efficient queries
- 90-day TTL for automatic data retention
- Optimized for time-range queries
- JSON storage for flexible attributes/events

## Kafka Topics

**traces:**
- Partition key: `trace_id`
- Consumer group: `trace-service-traces`
- Retention: 7 days

**spans:**
- Partition key: `trace_id` (ensures span ordering)
- Consumer group: `trace-service-spans`
- Retention: 7 days

## Monitoring & Observability

### Health Checks
- Ingest Gateway: `GET /health`
- API Gateway: `GET /api/v1/health`
- Trace Service: No HTTP endpoint (monitor logs)

### Logs
All services use structured logging (JSON format in production):

```python
logger.info("Message", extra={
    "trace_id": "...",
    "span_count": 10,
    "duration_ms": 123.45
})
```

Railway automatically captures and displays logs.

### Metrics (Future)
- Request rate, latency (FastAPI metrics)
- Kafka consumer lag
- ClickHouse query performance
- Error rates

## Cost Estimation

**Railway:**
- Ingest Gateway: ~$5-10/month (usage-based)
- API Gateway: ~$5-10/month
- Trace Service: ~$5-10/month
- Postgres: ~$5/month
- Redis: ~$5/month
- **Total: ~$25-45/month**

**ClickHouse Cloud:**
- Free tier: 30GB storage, sufficient for starting
- Paid: ~$0.11/GB/month storage + compute
- **Estimate: $0-50/month** (depends on volume)

**Upstash Kafka:**
- Free tier: 10K messages/day
- Paid: ~$0.40/million messages
- **Estimate: $0-20/month** (depends on volume)

**Grand Total: $25-115/month** for production-ready platform

## Development Roadmap

### Phase 1: MVP (Current)
- ✅ Service structure
- ✅ Local development environment
- ✅ Basic ingestion pipeline
- ✅ Query API
- ✅ Railway deployment configs

### Phase 2: Production Hardening
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] API key management
- [ ] User management (Postgres)
- [ ] Error tracking (Sentry)
- [ ] Metrics (Prometheus)

### Phase 3: Advanced Features
- [ ] Real-time trace streaming (WebSockets)
- [ ] Aggregation views (hourly/daily stats)
- [ ] Alerts and notifications
- [ ] Team collaboration features
- [ ] Cost controls and quotas

## Troubleshooting

### Local Development

**ClickHouse won't start:**
```bash
# Check logs
docker logs prela-clickhouse

# Increase ulimits in docker-compose.local.yml
```

**Kafka connection errors:**
```bash
# Verify Redpanda is running
docker ps | grep redpanda

# Check topics in console
open http://localhost:8080
```

**Import errors (shared module):**
```bash
# Ensure PYTHONPATH includes backend directory
export PYTHONPATH=/Users/gw/prela/backend:$PYTHONPATH
```

### Railway Deployment

**Service won't start:**
- Check environment variables are set correctly
- Verify ClickHouse/Kafka credentials
- Check Railway logs for errors

**Health check failing:**
- Increase `healthcheckTimeout` in railway.toml
- Verify service is listening on correct port (`$PORT`)

**Kafka consumer lag:**
- Scale up Trace Service replicas in Railway
- Increase batch sizes in consumer

---

## Communication Agents

The `/agents/` directory contains AI-powered agents for developer relations and growth:

### Agent Overview

**Scout** - Community engagement + issue tracking
- Monitors Reddit, HN, Twitter, Discord for Prela mentions
- Generates RAG-powered responses using docs + messaging
- Extracts feature requests/bugs and creates Linear issues

**Sentinel** - Competitive intelligence
- Tracks competitor releases (Langfuse, LangSmith, Arize)
- Analyzes user sentiment about competitors
- Generates weekly competitive intelligence reports

**Amplifier** - Content repurposing + battle cards
- Converts docs into Twitter threads, Reddit posts, HN comments
- Auto-generates battle cards (Prela vs competitors)
- Creates positioning docs and use case stories

**Radar** - Analytics & monitoring
- Tracks Prela metrics (GitHub stars, PyPI downloads, Discord members)
- Monitors competitor activity and market trends
- Generates weekly analytics reports

### Directory Structure

```
/agents/
├── shared/              # Shared utilities (LLM client, vector DB, platforms)
├── scout/               # Community engagement + issue tracking
├── sentinel/            # Competitive intelligence
├── amplifier/           # Content repurposing + battle cards
├── radar/               # Analytics & monitoring
└── docker-compose.yml   # Run all agents locally
```

### Documentation

For complete implementation details, see:
- [COMMUNICATION_AGENTS_IMPLEMENTATION.md](../internal/planning/COMMUNICATION_AGENTS_IMPLEMENTATION.md) - Full implementation guide
- [REPOSITORY_STRUCTURE.md](../internal/planning/REPOSITORY_STRUCTURE.md) - Detailed folder organization and design decisions

### Why Separate from Services?

**Product Services** (`/backend/services/`):
- Core product functionality (trace ingestion, storage, querying)
- Customer-facing features
- Required for product to function

**Communication Agents** (`/agents/`):
- Growth and developer relations
- Community engagement and competitive intelligence
- Enhances go-to-market but not product functionality

This separation allows agents to:
- Evolve independently from core product
- Have different dependencies (Claude API, Linear, platform APIs)
- Be deployed separately (different Railway services)
- Be open-sourced or extracted to separate repo if needed

---

## Contributing

See main project [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines.

## License

See main project [LICENSE](/LICENSE).
