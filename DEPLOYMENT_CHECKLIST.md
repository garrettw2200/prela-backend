# Railway Deployment Checklist

**Date:** February 5, 2026
**Estimated Time:** 4-6 hours
**Guide:** See RAILWAY_DEPLOYMENT_GUIDE.md for detailed instructions

---

## Pre-Deployment Checklist

### Prerequisites
- [ ] Railway account created and verified
- [ ] ClickHouse Cloud instance running (save host, password)
- [ ] Clerk application configured (save publishable key, secret key, JWKS URL)
- [ ] Stripe products and prices configured (save price IDs)
- [ ] Domain access to configure DNS for `api.prela.app`

### Information to Gather

**ClickHouse:**
- Host: `___________________.clickhouse.cloud`
- Password: `___________________________________`
- Database: `prela`

**Clerk:**
- Publishable Key: `pk_test_________________________`
- Secret Key: `sk_test_________________________`
- JWKS URL: `https://_________.clerk.accounts.dev/.well-known/jwks.json`

**Stripe:**
- Secret Key: `sk_test_________________________`
- Lunch Money Price ID: `price_1SwT6ZGw4cezTOlQ8WiBaznB`
- Pro Price ID: `price_1SwT7OGw4cezTOlQZfg2iMsm`

---

## Step-by-Step Deployment

### Step 1: Create Railway Project (15 min)
- [ ] Create new Railway project named "prela-backend"
- [ ] Add PostgreSQL database
- [ ] Add Redis database
- [ ] Save `DATABASE_URL` from PostgreSQL service
- [ ] Save `REDIS_URL` from Redis service

### Step 2: Run Database Migrations (10 min)
- [ ] Copy `DATABASE_URL` from Railway
- [ ] Run: `psql "$DATABASE_URL" < backend/migrations/001_create_users_and_subscriptions.sql`
- [ ] Verify tables created: `psql "$DATABASE_URL" -c "\dt"`
- [ ] Should see: `users`, `subscriptions`, `api_keys`, `usage_records`

### Step 3: Deploy API Gateway (30 min)
- [ ] Create new service in Railway
- [ ] Connect to GitHub repo
- [ ] Set root directory: `backend/services/api-gateway`
- [ ] Name service: `prela-api-gateway`
- [ ] Add environment variables (see Step 3.1 in guide)
  - [ ] SERVICE_NAME, ENVIRONMENT, LOG_LEVEL
  - [ ] DATABASE_URL, REDIS_URL (reference other services)
  - [ ] CLICKHOUSE_HOST, CLICKHOUSE_PASSWORD, etc.
  - [ ] CLERK_PUBLISHABLE_KEY, CLERK_SECRET_KEY, CLERK_JWKS_URL
  - [ ] STRIPE_SECRET_KEY, STRIPE_LUNCH_MONEY_PRICE_ID, etc.
  - [ ] CORS_ORIGINS, API_PORT, API_HOST
  - [ ] JWT_SECRET (generate random string)
- [ ] Deploy service
- [ ] Wait for build to complete (5-10 min)
- [ ] Check logs for errors
- [ ] Test health endpoint: `curl https://[railway-url]/api/v1/health`

### Step 4: Deploy Ingest Gateway (30 min)
- [ ] Create new service in Railway
- [ ] Connect to GitHub repo
- [ ] Set root directory: `backend/services/ingest-gateway`
- [ ] Name service: `prela-ingest-gateway`
- [ ] Add environment variables (see Step 3.2 in guide)
  - [ ] SERVICE_NAME, ENVIRONMENT, LOG_LEVEL
  - [ ] DATABASE_URL, REDIS_URL (reference other services)
  - [ ] CLICKHOUSE_HOST, CLICKHOUSE_PASSWORD, etc.
  - [ ] CORS_ORIGINS, API_PORT, API_HOST
- [ ] Deploy service
- [ ] Wait for build to complete (5-10 min)
- [ ] Check logs for errors
- [ ] Test health endpoint: `curl https://[railway-url]/health`

### Step 5: Configure Custom Domain (30 min)
- [ ] Go to API Gateway service → Settings → Domains
- [ ] Add custom domain: `api.prela.app`
- [ ] Copy DNS records provided by Railway
- [ ] Update DNS in domain registrar (Cloudflare/Namecheap/etc.)
  - Type: CNAME
  - Name: api
  - Value: [Railway URL]
- [ ] Wait for DNS propagation (5-60 min)
- [ ] Test: `dig api.prela.app`
- [ ] Test: `curl https://api.prela.app/api/v1/health`

### Step 6: Create Stripe Webhook (15 min)
- [ ] Go to Stripe dashboard → Webhooks
- [ ] Click "+ Add endpoint"
- [ ] Endpoint URL: `https://api.prela.app/api/v1/billing/webhooks/stripe`
- [ ] Select events:
  - [ ] customer.subscription.created
  - [ ] customer.subscription.updated
  - [ ] customer.subscription.deleted
  - [ ] invoice.created
  - [ ] invoice.paid
  - [ ] invoice.payment_failed
  - [ ] checkout.session.completed
- [ ] Save endpoint
- [ ] Copy signing secret (starts with `whsec_`)
- [ ] Update Railway environment variable: `STRIPE_WEBHOOK_SECRET=whsec_...`
- [ ] Redeploy API Gateway service
- [ ] Test webhook: `stripe trigger checkout.session.completed`

### Step 7: End-to-End Testing (30 min)
- [ ] Test API Gateway health: `curl https://api.prela.app/api/v1/health`
- [ ] Test Ingest Gateway health: `curl https://api.prela.app/health`
- [ ] Test authentication (should return 401): `curl https://api.prela.app/api/v1/api-keys`
- [ ] Test trace ingestion (should return 401): `curl -X POST https://api.prela.app/api/v1/batch -H "Content-Type: application/json" -d '{"traces":[]}'`
- [ ] Check Railway logs - no errors
- [ ] Check ClickHouse - can connect
- [ ] Check PostgreSQL - tables exist
- [ ] Check Redis - can ping
- [ ] Test Stripe webhook - receive events in logs

### Step 8: Monitoring Setup (20 min)
- [ ] Verify Railway health checks are working
- [ ] Set up external monitoring (UptimeRobot/Pingdom)
  - URL: `https://api.prela.app/api/v1/health`
  - Expected: `{"status":"healthy"}`
  - Alert if down >2 minutes
- [ ] Configure error tracking (Sentry/Rollbar - optional)
- [ ] Set up log aggregation (optional)

---

## Verification Checklist

### Services Running
- [ ] API Gateway: Status = "Healthy" in Railway
- [ ] Ingest Gateway: Status = "Healthy" in Railway
- [ ] PostgreSQL: Status = "Healthy" in Railway
- [ ] Redis: Status = "Healthy" in Railway

### Endpoints Working
- [ ] `https://api.prela.app/` returns service info
- [ ] `https://api.prela.app/api/v1/health` returns `{"status":"healthy"}`
- [ ] `https://api.prela.app/health` returns `{"status":"healthy"}`

### Authentication Working
- [ ] Endpoints return 401 without auth (expected)
- [ ] Clerk JWKS URL reachable
- [ ] API key authentication logic in place

### Database Connectivity
- [ ] PostgreSQL tables exist (users, subscriptions, api_keys, usage_records)
- [ ] ClickHouse connection successful
- [ ] Redis connection successful

### Stripe Integration
- [ ] Webhook endpoint created
- [ ] Webhook secret configured
- [ ] Test webhook received in logs

### DNS & Domains
- [ ] `api.prela.app` resolves correctly
- [ ] HTTPS certificate issued (Railway auto-generates)
- [ ] No certificate errors

---

## Post-Deployment

### Security Review
- [ ] All secrets in Railway environment variables (not in code)
- [ ] CORS origins restricted to actual domains
- [ ] ENVIRONMENT=production set
- [ ] JWT_SECRET is random and secure
- [ ] Rate limiting enabled

### Cost Monitoring
- [ ] Railway usage dashboard reviewed
- [ ] ClickHouse usage reviewed
- [ ] Cost alerts configured

### Documentation
- [ ] Update internal docs with Railway URLs
- [ ] Update team with new endpoints
- [ ] Save all credentials in secure location (1Password/etc.)

---

## Troubleshooting

If deployment fails, check:
1. Railway build logs for errors
2. Railway runtime logs for crashes
3. Environment variables - all required ones set?
4. Database migrations - ran successfully?
5. ClickHouse/Redis - can connect?
6. Dockerfile - building correctly?
7. requirements.txt - all dependencies listed?

See RAILWAY_DEPLOYMENT_GUIDE.md "Troubleshooting" section for detailed solutions.

---

## Success Criteria

✅ **Phase 1 Complete** when:
- [ ] Both services deployed and running
- [ ] Health checks passing
- [ ] Domain configured and working
- [ ] Stripe webhooks receiving events
- [ ] Zero errors in production logs
- [ ] All databases connected and working

**Time to complete:** 4-6 hours

**Next:** Phase 2 - Frontend deployment with Clerk integration

---

## Quick Reference

**Railway Dashboard:**
https://railway.app/project/[your-project-id]

**API Gateway Logs:**
```bash
railway logs --service prela-api-gateway
```

**Test Health:**
```bash
curl https://api.prela.app/api/v1/health
```

**Run Migration:**
```bash
psql "$DATABASE_URL" < backend/migrations/001_create_users_and_subscriptions.sql
```

**Update Environment Variable:**
```bash
railway variables set STRIPE_WEBHOOK_SECRET=whsec_xxx
```

---

**Created:** February 5, 2026
**Status:** Ready to execute
