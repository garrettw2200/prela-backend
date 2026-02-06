# Manual Deployment Steps - Railway Web UI

Since the Railway CLI requires interactive input, please follow these steps using the Railway web dashboard.

**Project URL:** https://railway.com/project/a1cfc0a4-cfcc-469a-9333-d856e0bc8812

---

## Step 1: Add PostgreSQL Database (5 minutes)

1. Open your project: https://railway.com/project/a1cfc0a4-cfcc-469a-9333-d856e0bc8812
2. Click **"+ New"** button (top right)
3. Select **"Database"**
4. Choose **"PostgreSQL"**
5. Wait for provisioning (~30 seconds)
6. Click on the PostgreSQL service
7. Go to **"Variables"** tab
8. Copy the `DATABASE_URL` value - you'll need this for migrations

**Save your DATABASE_URL here:**

DATABASE_PUBLIC_URL="postgresql://${{PGUSER}}:${{POSTGRES_PASSWORD}}@${{RAILWAY_TCP_PROXY_DOMAIN}}:${{RAILWAY_TCP_PROXY_PORT}}/${{PGDATABASE}}"
DATABASE_URL="postgresql://${{PGUSER}}:${{POSTGRES_PASSWORD}}@${{RAILWAY_PRIVATE_DOMAIN}}:5432/${{PGDATABASE}}"
PGDATA="/var/lib/postgresql/data/pgdata"
PGDATABASE="${{POSTGRES_DB}}"
PGHOST="${{RAILWAY_PRIVATE_DOMAIN}}"
PGPASSWORD="${{POSTGRES_PASSWORD}}"
PGPORT="5432"
PGUSER="${{POSTGRES_USER}}"
POSTGRES_DB="railway"
POSTGRES_PASSWORD="ECODOtCTMmNZEKnPHucmFYnDCljXoIQf"
POSTGRES_USER="postgres"
RAILWAY_DEPLOYMENT_DRAINING_SECONDS="60"
SSL_CERT_DAYS="820"

```
postgresql://postgres:[password]@[host]:[port]/railway
```

---

## Step 2: Add Redis Database (5 minutes)

1. Click **"+ New"** button again
2. Select **"Database"**
3. Choose **"Redis"**
4. Wait for provisioning (~30 seconds)
5. Click on the Redis service
6. Go to **"Variables"** tab
7. Copy the `REDIS_URL` value

**Save your REDIS_URL here:**

REDIS_PASSWORD="YHuDYpDyHGHQqGjvjCMlMzCszwTmHJNm"
REDIS_PUBLIC_URL="redis://default:${{REDIS_PASSWORD}}@${{RAILWAY_TCP_PROXY_DOMAIN}}:${{RAILWAY_TCP_PROXY_PORT}}"
REDIS_URL="redis://${{REDISUSER}}:${{REDIS_PASSWORD}}@${{REDISHOST}}:${{REDISPORT}}"
REDISHOST="${{RAILWAY_PRIVATE_DOMAIN}}"
REDISPASSWORD="${{REDIS_PASSWORD}}"
REDISPORT="6379"
REDISUSER="default"

```
redis://default:[password]@[host]:[port]
```

---

## Step 3: Run PostgreSQL Migrations (5 minutes)

Once you have the `DATABASE_URL`, run this command in your terminal:

```bash
cd /Users/gw/prela/backend

# Run the migration
psql "$DATABASE_URL" < migrations/001_create_users_and_subscriptions.sql

# Verify tables were created
psql "$DATABASE_URL" -c "\dt"
```

**Expected output:**
```
                List of relations
 Schema |      Name       | Type  |  Owner
--------+-----------------+-------+---------
 public | api_keys        | table | postgres
 public | subscriptions   | table | postgres
 public | usage_records   | table | postgres
 public | users           | table | postgres
```

---

## Step 4: Generate JWT Secret (1 minute)

Run this command to generate a secure random secret:

```bash
openssl rand -base64 32
```

**Save the output - you'll need it for environment variables:**
```
_________________________________
```

---

## Step 5: Prepare Environment Variables

Before deploying services, prepare these values:

### From ClickHouse Cloud:
- `CLICKHOUSE_HOST`: ____________.clickhouse.cloud
- `CLICKHOUSE_PASSWORD`: _______________________

### From Clerk Dashboard:
- `CLERK_PUBLISHABLE_KEY`: pk_test_____________ or pk_live_____________
- `CLERK_SECRET_KEY`: sk_test_____________ or sk_live_____________
- `CLERK_JWKS_URL`: https://_______.clerk.accounts.dev/.well-known/jwks.json

### From Stripe (already in STRIPE_CONFIGURATION.md):
- `STRIPE_SECRET_KEY`: sk_test_your_stripe_secret_key_here
- `STRIPE_LUNCH_MONEY_PRICE_ID`: price_1SwT6ZGw4cezTOlQ8WiBaznB
- `STRIPE_PRO_PRICE_ID`: price_1SwT7OGw4cezTOlQZfg2iMsm

### Generated:
- `JWT_SECRET`: (from openssl command above)

---

## Step 6: Deploy API Gateway (20 minutes)

### 6.1 Create GitHub Repository (if not done)

First, ensure your backend code is in a GitHub repository:

```bash
cd /Users/gw/prela

# Check git status
git status

# If not already committed:
git add backend/
git commit -m "Prepare backend for Railway deployment

- Updated ingest-gateway requirements.txt
- Added Railway deployment documentation
- Ready for production deployment

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git push
```

### 6.2 Connect GitHub to Railway

1. In Railway project, click **"+ New"**
2. Select **"GitHub Repo"**
3. Authorize Railway to access your GitHub (if first time)
4. Select your `prela` repository
5. Railway will ask for the service to deploy

### 6.3 Configure API Gateway Service

**Service Settings:**
- **Name:** `prela-api-gateway`
- **Root Directory:** `backend/services/api-gateway`
- **Build Method:** Dockerfile (auto-detected from railway.toml)

**Environment Variables:**
Click on the service → **Variables** tab → Add these:

```bash
# Service Configuration
SERVICE_NAME=prela-api-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (use Railway references)
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# ClickHouse Cloud (replace with your values)
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_PORT=8443
CLICKHOUSE_DATABASE=prela

# Clerk Authentication (replace with your values)
CLERK_PUBLISHABLE_KEY=pk_test_or_pk_live_your_key
CLERK_SECRET_KEY=sk_test_or_sk_live_your_key
CLERK_JWKS_URL=https://your-app.clerk.accounts.dev/.well-known/jwks.json

# Stripe
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_will_set_after_creating_webhook
STRIPE_LUNCH_MONEY_PRICE_ID=price_1SwT6ZGw4cezTOlQ8WiBaznB
STRIPE_PRO_PRICE_ID=price_1SwT7OGw4cezTOlQZfg2iMsm

# Pro Tier Overage Price IDs
STRIPE_PRO_TRACES_PRICE_ID=price_1SwTBcGw4cezTOlQxIugsnHg
STRIPE_PRO_USERS_PRICE_ID=price_1SwTBdGw4cezTOlQHqP0Znfn
STRIPE_PRO_AI_HALLUCINATION_PRICE_ID=price_1SwTBdGw4cezTOlQIgycGjVV
STRIPE_PRO_AI_DRIFT_PRICE_ID=price_1SwTBeGw4cezTOlQLCHm8Kh8
STRIPE_PRO_AI_NLP_PRICE_ID=price_1SwTBfGw4cezTOlQUJU1iPnt
STRIPE_PRO_RETENTION_PRICE_ID=price_1SwTBfGw4cezTOlQUT0YFX75

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
CORS_ORIGINS=["https://dashboard.prela.app","http://localhost:5173"]

# JWT Security (use your generated secret)
JWT_SECRET=your_generated_jwt_secret_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
```

**Deploy:**
1. Railway will automatically start building
2. Watch the **Logs** tab for build progress
3. Build takes ~5-10 minutes
4. Once deployed, you'll see "Deployed successfully"

**Get the URL:**
1. Click **Settings** → **Networking**
2. Railway will show a public URL like: `prela-api-gateway-production.up.railway.app`
3. Test it: `curl https://[your-url]/api/v1/health`

---

## Step 7: Deploy Ingest Gateway (20 minutes)

### 7.1 Create Second Service

1. In Railway project, click **"+ New"**
2. Select **"GitHub Repo"**
3. Select your `prela` repository again
4. This will create a second service from the same repo

### 7.2 Configure Ingest Gateway Service

**Service Settings:**
- **Name:** `prela-ingest-gateway`
- **Root Directory:** `backend/services/ingest-gateway`
- **Build Method:** Dockerfile (auto-detected)

**Environment Variables:**
```bash
# Service Configuration
SERVICE_NAME=prela-ingest-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (use Railway references)
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# ClickHouse Cloud (same as API Gateway)
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_PORT=8443
CLICKHOUSE_DATABASE=prela

# API Configuration
API_PORT=8001
API_HOST=0.0.0.0
CORS_ORIGINS=["https://dashboard.prela.app","http://localhost:5173"]
```

**Deploy:**
- Railway will auto-build and deploy
- Watch logs for any errors
- Test: `curl https://[ingest-url]/health`

---

## Step 8: Configure Custom Domain (30 minutes)

### 8.1 Add Domain in Railway

1. Click on **API Gateway** service
2. Go to **Settings** → **Networking**
3. Click **"+ Custom Domain"**
4. Enter: `api.prela.app`
5. Railway will provide DNS instructions

### 8.2 Update DNS Records

Railway will tell you to create a CNAME record. Go to your DNS provider and add:

**Type:** CNAME
**Name:** api
**Value:** prela-api-gateway-production.up.railway.app (or whatever Railway provides)
**TTL:** 3600 (or Auto)

### 8.3 Wait for DNS Propagation

```bash
# Check DNS (may take 5-60 minutes)
dig api.prela.app

# Once propagated, test:
curl https://api.prela.app/api/v1/health

# Expected: {"status":"healthy"}
```

---

## Step 9: Create Stripe Webhook (15 minutes)

### 9.1 Create Webhook Endpoint

**Option A: Stripe Dashboard**
1. Go to https://dashboard.stripe.com/webhooks
2. Click **"+ Add endpoint"**
3. **Endpoint URL:** `https://api.prela.app/api/v1/billing/webhooks/stripe`
4. **Events to send:**
   - customer.subscription.created
   - customer.subscription.updated
   - customer.subscription.deleted
   - invoice.created
   - invoice.paid
   - invoice.payment_failed
   - checkout.session.completed
5. Click **"Add endpoint"**
6. Copy the **Signing secret** (starts with `whsec_`)

**Option B: Stripe CLI**
```bash
stripe webhook_endpoints create \
  --url="https://api.prela.app/api/v1/billing/webhooks/stripe" \
  --enabled-events="customer.subscription.created" \
  --enabled-events="customer.subscription.updated" \
  --enabled-events="customer.subscription.deleted" \
  --enabled-events="invoice.created" \
  --enabled-events="invoice.paid" \
  --enabled-events="invoice.payment_failed" \
  --enabled-events="checkout.session.completed"
```

### 9.2 Update Railway Environment Variable

1. Go to API Gateway service in Railway
2. Go to **Variables** tab
3. Find `STRIPE_WEBHOOK_SECRET`
4. Update value to: `whsec_your_signing_secret`
5. Railway will auto-redeploy

### 9.3 Test Webhook

```bash
# Send test event
stripe trigger checkout.session.completed \
  --webhook-endpoint https://api.prela.app/api/v1/billing/webhooks/stripe

# Check Railway logs for:
# "Received Stripe webhook: checkout.session.completed"
```

---

## Step 10: Final Testing (15 minutes)

### 10.1 Health Checks

```bash
# API Gateway
curl https://api.prela.app/api/v1/health
# Expected: {"status":"healthy"}

# Ingest Gateway
curl https://api.prela.app/health
# Expected: {"status":"healthy"}

# Root endpoint
curl https://api.prela.app/
# Expected: {"service":"prela-api-gateway","version":"0.1.0","status":"running"}
```

### 10.2 Authentication Tests

```bash
# Should return 401 (expected - no auth)
curl -X GET https://api.prela.app/api/v1/api-keys
# Expected: {"detail":"Not authenticated"}

# Should return 401 (expected - no API key)
curl -X POST https://api.prela.app/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"traces":[]}'
# Expected: {"detail":"Missing API key..."}
```

**✅ If you get 401 errors, authentication is working correctly!**

### 10.3 Check Railway Logs

1. Go to each service in Railway
2. Click **"Logs"** tab
3. Look for:
   - ✅ "Starting API Gateway (production)"
   - ✅ "ClickHouse client initialized"
   - ✅ "Rate limiter initialized"
   - ❌ No errors or exceptions

### 10.4 Database Verification

```bash
# Check PostgreSQL tables
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM users;"
# Expected: 0 (no users yet - will be created in Phase 2)

# Check Redis connection
redis-cli -u "$REDIS_URL" ping
# Expected: PONG
```

---

## ✅ Deployment Complete!

**Phase 1 is complete when:**
- [  ] PostgreSQL and Redis provisioned
- [  ] Database migrations run successfully
- [  ] API Gateway deployed and healthy
- [  ] Ingest Gateway deployed and healthy
- [  ] Custom domain `api.prela.app` working
- [  ] Stripe webhook created and receiving events
- [  ] All health checks passing
- [  ] Authentication returning 401 (as expected)
- [  ] Zero errors in production logs

---

## Next Steps

1. **Phase 2:** Deploy frontend with Clerk integration
2. **Phase 3:** Update documentation
3. **Phase 4:** Launch announcement

---

## Troubleshooting

If you encounter issues, see:
- [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md) - Full troubleshooting section
- Railway logs in dashboard
- Check environment variables are set correctly

---

**When you're done with these steps, let me know and I'll help with Phase 2!**
