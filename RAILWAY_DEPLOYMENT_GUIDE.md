# Railway Deployment Guide - Prela Backend

**Date:** February 5, 2026
**Status:** Ready for Production Deployment
**Architecture:** Simplified (No Kafka - Direct ClickHouse writes)

---

## Overview

This guide walks through deploying the Prela backend to Railway with the simplified architecture:
- Direct HTTP → ClickHouse writes (no Kafka)
- PostgreSQL for users/subscriptions
- Redis for rate limiting
- Clerk for authentication
- Stripe for billing

**Services to Deploy:**
1. **API Gateway** - Main REST API (public URL: api.prela.app)
2. **Ingest Gateway** - Trace ingestion (public URL: ingest.prela.app or same as API Gateway)
3. **PostgreSQL** - Railway managed database
4. **Redis** - Railway managed Redis (for rate limiting)

**External Services:**
- **ClickHouse Cloud** - Trace/span storage
- **Clerk** - Authentication
- **Stripe** - Billing

---

## Prerequisites

### 1. Railway Account
- Sign up at https://railway.app
- Install Railway CLI (optional): `npm install -g @railway/cli`

### 2. ClickHouse Cloud Account
- Sign up at https://clickhouse.com/cloud
- Already set up if you followed previous docs

### 3. Clerk Account
- Sign up at https://clerk.com
- Already configured (see CLERK_AUTH_IMPLEMENTATION.md)

### 4. Stripe Account
- Already configured (see STRIPE_CONFIGURATION.md)

### 5. Domain Access
- Access to configure DNS for `prela.app` domain
- Need to add A/CNAME records for `api.prela.app`

---

## Step 1: Create Railway Project

### 1.1 Create New Project

```bash
# Option A: Using Railway CLI
railway login
railway init

# Option B: Using Web UI
# 1. Go to https://railway.app/new
# 2. Click "New Project"
# 3. Name it: "prela-backend"
```

### 1.2 Add PostgreSQL Database

```bash
# CLI method
railway add --database postgres

# Web UI method:
# 1. In your project, click "New"
# 2. Select "Database" → "PostgreSQL"
# 3. Railway will provision and provide DATABASE_URL automatically
```

**Important:** Save the `DATABASE_URL` - you'll need it for migrations.

### 1.3 Add Redis Database

```bash
# CLI method
railway add --database redis

# Web UI method:
# 1. In your project, click "New"
# 2. Select "Database" → "Redis"
# 3. Railway will provision and provide REDIS_URL automatically
```

---

## Step 2: Run Database Migrations

Before deploying services, run the PostgreSQL migrations:

### 2.1 Get PostgreSQL Connection String

From Railway dashboard:
1. Click on your PostgreSQL database
2. Go to "Variables" tab
3. Copy the `DATABASE_URL` (format: `postgresql://user:pass@host:port/db`)

### 2.2 Run Migration

```bash
# Navigate to backend directory
cd /Users/gw/prela/backend

# Install psql if not already installed (macOS)
brew install postgresql

# Run migration
psql "$DATABASE_URL" < migrations/001_create_users_and_subscriptions.sql

# Verify tables created
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

## Step 3: Configure Environment Variables

### 3.1 Required Variables for API Gateway

In Railway dashboard, go to API Gateway service → Variables tab:

```bash
# Service Configuration
SERVICE_NAME=prela-api-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (auto-provided by Railway - reference other service)
DATABASE_URL=${{Postgres.DATABASE_URL}}

# ClickHouse Cloud (from your ClickHouse setup)
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_PORT=8443
CLICKHOUSE_DATABASE=prela

# Redis (auto-provided by Railway - reference other service)
REDIS_URL=${{Redis.REDIS_URL}}

# Clerk Authentication (from Clerk dashboard)
CLERK_PUBLISHABLE_KEY=pk_live_xxx  # or pk_test_xxx for testing
CLERK_SECRET_KEY=sk_live_xxx       # or sk_test_xxx for testing
CLERK_JWKS_URL=https://your-app.clerk.accounts.dev/.well-known/jwks.json

# Stripe (from STRIPE_CONFIGURATION.md)
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_xxx  # Create webhook endpoint first (see Step 5)
STRIPE_LUNCH_MONEY_PRICE_ID=price_1SwT6ZGw4cezTOlQ8WiBaznB
STRIPE_PRO_PRICE_ID=price_1SwT7OGw4cezTOlQZfg2iMsm

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
CORS_ORIGINS=["https://dashboard.prela.app","http://localhost:5173"]

# JWT (generate a random secret)
JWT_SECRET=your-random-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60
```

### 3.2 Required Variables for Ingest Gateway

In Railway dashboard, go to Ingest Gateway service → Variables tab:

```bash
# Service Configuration
SERVICE_NAME=prela-ingest-gateway
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (auto-provided by Railway)
DATABASE_URL=${{Postgres.DATABASE_URL}}

# ClickHouse Cloud (same as API Gateway)
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
CLICKHOUSE_PORT=8443
CLICKHOUSE_DATABASE=prela

# Redis (auto-provided by Railway)
REDIS_URL=${{Redis.REDIS_URL}}

# API Configuration
API_PORT=8001
API_HOST=0.0.0.0
CORS_ORIGINS=["https://dashboard.prela.app","http://localhost:5173"]
```

**Pro Tip:** Use Railway's variable referencing syntax `${{ServiceName.VARIABLE_NAME}}` to avoid duplicating values.

---

## Step 4: Deploy Services

### 4.1 Deploy API Gateway

#### Option A: Using GitHub (Recommended)

1. **Push code to GitHub** (if not already done):
   ```bash
   cd /Users/gw/prela
   git add backend/services/api-gateway
   git commit -m "Prepare API Gateway for Railway deployment"
   git push
   ```

2. **Connect to Railway**:
   - In Railway project, click "New" → "GitHub Repo"
   - Select your repository
   - Set root directory: `backend/services/api-gateway`
   - Railway will detect `railway.toml` and `Dockerfile`

3. **Configure service**:
   - Name: `prela-api-gateway`
   - Railway will automatically build and deploy

#### Option B: Using Railway CLI

```bash
cd /Users/gw/prela/backend/services/api-gateway

# Login to Railway
railway login

# Link to your project
railway link

# Deploy
railway up
```

### 4.2 Deploy Ingest Gateway

Follow same steps as API Gateway but with:
- Root directory: `backend/services/ingest-gateway`
- Service name: `prela-ingest-gateway`

### 4.3 Verify Deployments

```bash
# Check API Gateway health
curl https://prela-api-gateway.up.railway.app/api/v1/health

# Check Ingest Gateway health
curl https://prela-ingest-gateway.up.railway.app/health

# Expected response for both:
{"status": "healthy"}
```

---

## Step 5: Configure Custom Domain

### 5.1 Add Domain to Railway

1. Go to API Gateway service in Railway
2. Click "Settings" → "Domains"
3. Click "Custom Domain"
4. Enter: `api.prela.app`
5. Railway will provide DNS records

### 5.2 Update DNS Records

In your domain registrar (e.g., Cloudflare, Namecheap):

**For api.prela.app:**
```
Type: CNAME
Name: api
Value: prela-api-gateway.up.railway.app
TTL: Auto or 3600
```

**Alternative (if CNAME not supported):**
```
Type: A
Name: api
Value: [IP from Railway]
TTL: Auto or 3600
```

### 5.3 Wait for DNS Propagation

```bash
# Check DNS propagation (may take 5-60 minutes)
dig api.prela.app

# Test endpoint
curl https://api.prela.app/api/v1/health
```

---

## Step 6: Create Stripe Webhook Endpoint

Now that the backend is deployed, create the Stripe webhook:

### 6.1 Using Stripe CLI

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

### 6.2 Using Stripe Dashboard

1. Go to https://dashboard.stripe.com/webhooks
2. Click "+ Add endpoint"
3. Endpoint URL: `https://api.prela.app/api/v1/billing/webhooks/stripe`
4. Select events:
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.created`
   - `invoice.paid`
   - `invoice.payment_failed`
   - `checkout.session.completed`
5. Click "Add endpoint"
6. Copy the signing secret (starts with `whsec_`)

### 6.3 Update Railway Environment Variable

Go to API Gateway service → Variables:
```bash
STRIPE_WEBHOOK_SECRET=whsec_your_signing_secret_here
```

Redeploy the service (Railway will auto-redeploy on variable change).

---

## Step 7: End-to-End Testing

### 7.1 Test API Gateway Endpoints

```bash
# 1. Health check
curl https://api.prela.app/api/v1/health

# Expected: {"status": "healthy"}

# 2. Test root endpoint
curl https://api.prela.app/

# Expected: {"service":"prela-api-gateway","version":"0.1.0","status":"running"}
```

### 7.2 Test Authentication (Manual)

Since we don't have a user yet, we can't fully test auth. This will be tested in Phase 2 after frontend deployment.

**For now, verify the auth endpoints exist:**

```bash
# This should return 401 (expected - no auth)
curl -X GET https://api.prela.app/api/v1/api-keys

# Expected: {"detail": "Not authenticated"}
```

### 7.3 Test Ingest Gateway

```bash
# This should return 401 (expected - no API key)
curl -X POST https://api.prela.app/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"traces":[]}'

# Expected: {"detail": "Missing API key..."}
```

**✅ If you get 401 errors, authentication is working correctly!**

### 7.4 Test Stripe Webhook

```bash
# Use Stripe CLI to send test event
stripe trigger checkout.session.completed \
  --webhook-endpoint https://api.prela.app/api/v1/billing/webhooks/stripe
```

Check Railway logs:
1. Go to API Gateway service
2. Click "Logs" tab
3. Look for: `"Received Stripe webhook: checkout.session.completed"`

---

## Step 8: Monitor Logs

### 8.1 View Real-Time Logs

**Railway Dashboard:**
- Click on service → "Logs" tab
- Real-time streaming logs

**Railway CLI:**
```bash
# API Gateway logs
railway logs --service prela-api-gateway

# Ingest Gateway logs
railway logs --service prela-ingest-gateway
```

### 8.2 Key Things to Monitor

✅ **Good signs:**
- `"Starting API Gateway (production)"`
- `"ClickHouse client initialized"`
- `"Rate limiter initialized"`
- No errors or exceptions

❌ **Red flags:**
- `Connection refused` - Check DATABASE_URL or CLICKHOUSE_HOST
- `Authentication failed` - Check CLERK_SECRET_KEY
- `KeyError` - Missing environment variable

---

## Step 9: Set Up Health Check Monitoring

### 9.1 Railway Built-in Health Checks

Railway automatically monitors the health check paths defined in `railway.toml`:
- API Gateway: `/api/v1/health`
- Ingest Gateway: `/health`

If health checks fail, Railway will:
1. Restart the service (up to 3 times)
2. Show status as "Unhealthy" in dashboard

### 9.2 External Monitoring (Optional)

Set up external monitoring with services like:
- **UptimeRobot** - Free tier, 5-minute checks
- **Pingdom** - More features, paid
- **StatusCake** - Free tier available

Monitor:
- `https://api.prela.app/api/v1/health`
- Response: `{"status":"healthy"}`
- Alert if down for >2 minutes

---

## Troubleshooting

### Issue: Service Won't Start

**Symptoms:** Logs show import errors or crashes immediately

**Solutions:**
1. Check `requirements.txt` - all dependencies listed?
2. Check Dockerfile - copying shared modules correctly?
3. Verify environment variables - no typos?

```bash
# Check Railway build logs
railway logs --service prela-api-gateway | grep ERROR
```

### Issue: Database Connection Fails

**Symptoms:** `asyncpg.exceptions.InvalidPasswordError`

**Solutions:**
1. Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/db`
2. Check PostgreSQL service is running in Railway
3. Use Railway's variable reference: `${{Postgres.DATABASE_URL}}`

### Issue: ClickHouse Connection Fails

**Symptoms:** `Connection refused to clickhouse.cloud:8443`

**Solutions:**
1. Verify ClickHouse Cloud service is running
2. Check firewall - allow Railway IPs (or use public access)
3. Test connection manually:
   ```bash
   curl 'https://your-host.clickhouse.cloud:8443/?query=SELECT%201' \
     --user 'default:your-password'
   ```

### Issue: Clerk Authentication Not Working

**Symptoms:** `401 Unauthorized` for valid JWT tokens

**Solutions:**
1. Verify `CLERK_JWKS_URL` is correct
2. Check JWT token is not expired
3. Ensure `CLERK_SECRET_KEY` matches your Clerk dashboard
4. Check Clerk app is in production mode (if using live keys)

### Issue: Stripe Webhooks Not Received

**Symptoms:** No logs showing webhook events

**Solutions:**
1. Verify webhook URL in Stripe dashboard: `https://api.prela.app/api/v1/billing/webhooks/stripe`
2. Check webhook signing secret is correct
3. Test with Stripe CLI:
   ```bash
   stripe listen --forward-to https://api.prela.app/api/v1/billing/webhooks/stripe
   ```
4. Check Railway logs for webhook errors

### Issue: Rate Limiting Not Working

**Symptoms:** Can exceed rate limits

**Solutions:**
1. Verify Redis is running: `railway logs --service redis`
2. Check `REDIS_URL` environment variable
3. Test Redis connection:
   ```bash
   redis-cli -u "$REDIS_URL" ping
   # Expected: PONG
   ```

---

## Security Checklist

Before going live, verify:

- [ ] All environment variables use Railway secrets (not committed to git)
- [ ] `ENVIRONMENT=production` is set
- [ ] CORS origins only include your domains
- [ ] Clerk is in production mode (if using live keys)
- [ ] Stripe is in live mode (when ready to accept real payments)
- [ ] Railway services are in private network (only API Gateway exposed)
- [ ] Database backups are enabled in Railway
- [ ] Monitoring is set up for all services
- [ ] Error tracking configured (e.g., Sentry)

---

## Cost Estimate

**Railway Monthly Costs:**
- PostgreSQL: $10/month
- Redis: $0 (free tier) or $5/month if exceeds free limits
- API Gateway compute: ~$10-15/month
- Ingest Gateway compute: ~$5-10/month
- **Total Railway: ~$25-40/month**

**External Services:**
- ClickHouse Cloud (Development tier): $29/month
- Upstash Redis: $0 (free tier)
- Clerk: $0 (free up to 5,000 MAU)
- Stripe: 2.9% + $0.30 per transaction
- **Total External: ~$29/month + transaction fees**

**Grand Total: ~$55-70/month** (before revenue)

**Break-even:** 4-5 Lunch Money subscribers ($14/month)

---

## Next Steps

✅ **Phase 1 Complete:** Backend deployed to production!

**Next (Phase 2):** Deploy frontend with Clerk integration
- Integrate @clerk/clerk-react in frontend
- Deploy frontend to Railway or Vercel
- Test end-to-end: sign up → create API key → send traces → view dashboard

---

## Useful Commands

```bash
# Deploy updates to API Gateway
cd /Users/gw/prela/backend/services/api-gateway
railway up

# View logs
railway logs --service prela-api-gateway --tail

# Open Railway dashboard
railway open

# Check service status
railway status

# Run PostgreSQL query
psql "$DATABASE_URL" -c "SELECT * FROM users;"

# Test API locally before deploying
cd /Users/gw/prela/backend/services/api-gateway
uvicorn app.main:app --reload
```

---

## Support

If you encounter issues:
1. Check Railway logs first
2. Review this troubleshooting section
3. Search Railway Discord community
4. Open issue in GitHub repo

---

**Document Version:** 1.0
**Last Updated:** February 5, 2026
**Status:** Ready for production deployment
