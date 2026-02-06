# Railway Deployment - Quick Start

**⏱️ Estimated Time:** 4-6 hours
**📋 Full Guide:** [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md)
**✅ Checklist:** [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

---

## TL;DR - Deployment Steps

### 1. Create Railway Project (15 min)
```bash
# Create project, add PostgreSQL + Redis
railway init
railway add --database postgres
railway add --database redis
```

### 2. Run Migrations (10 min)
```bash
# Get DATABASE_URL from Railway, then:
psql "$DATABASE_URL" < backend/migrations/001_create_users_and_subscriptions.sql
```

### 3. Deploy API Gateway (30 min)
```bash
cd backend/services/api-gateway
# Connect to GitHub in Railway UI
# Set root directory: backend/services/api-gateway
# Add environment variables from .env.railway.template
# Railway will auto-build and deploy
```

### 4. Deploy Ingest Gateway (30 min)
```bash
cd backend/services/ingest-gateway
# Repeat same process as API Gateway
# Set root directory: backend/services/ingest-gateway
```

### 5. Configure Domain (30 min)
```bash
# In Railway: Settings → Domains → Add "api.prela.app"
# Update DNS: CNAME api → [railway-url]
# Test: curl https://api.prela.app/api/v1/health
```

### 6. Create Stripe Webhook (15 min)
```bash
stripe webhook_endpoints create \
  --url="https://api.prela.app/api/v1/billing/webhooks/stripe" \
  --enabled-events="checkout.session.completed" \
  # ... (see full list in guide)

# Copy signing secret to Railway: STRIPE_WEBHOOK_SECRET
```

### 7. Test Everything (30 min)
```bash
# Health checks
curl https://api.prela.app/api/v1/health
curl https://api.prela.app/health

# Auth (should return 401 - expected)
curl https://api.prela.app/api/v1/api-keys

# Check logs in Railway dashboard
railway logs --service prela-api-gateway
```

---

## Required Information

Before starting, gather these credentials:

**ClickHouse:**
- Host: `____________.clickhouse.cloud`
- Password: `_______________________`

**Clerk:**
- Publishable Key: `pk_test_____________`
- Secret Key: `sk_test_____________`
- JWKS URL: `https://______.clerk.accounts.dev/.well-known/jwks.json`

**Stripe:**
- Secret Key: `sk_test_____________` (from STRIPE_CONFIGURATION.md)
- Price IDs: Already documented in STRIPE_CONFIGURATION.md

---

## Environment Variables - Copy/Paste Ready

See [.env.railway.template](.env.railway.template) for complete list.

**Critical variables:**
```bash
# Replace these placeholders:
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_PASSWORD=your-password
CLERK_PUBLISHABLE_KEY=pk_test_xxx
CLERK_SECRET_KEY=sk_test_xxx
CLERK_JWKS_URL=https://your-app.clerk.accounts.dev/.well-known/jwks.json
JWT_SECRET=generate-random-32-char-string
STRIPE_WEBHOOK_SECRET=whsec_xxx_after_webhook_created

# Use Railway references:
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
```

---

## Success Checklist

✅ **Deployment successful when:**
- [ ] API Gateway health check passes: `https://api.prela.app/api/v1/health`
- [ ] Ingest Gateway health check passes: `https://api.prela.app/health`
- [ ] Auth returns 401 (expected): `curl https://api.prela.app/api/v1/api-keys`
- [ ] Stripe webhook receives events (check Railway logs)
- [ ] Railway services show "Healthy" status
- [ ] Zero errors in production logs

---

## Common Issues & Quick Fixes

**Service won't start:**
```bash
# Check build logs
railway logs --service prela-api-gateway | grep ERROR

# Verify environment variables are set
railway variables
```

**Database connection fails:**
```bash
# Use Railway's variable reference (not hardcoded connection string)
DATABASE_URL=${{Postgres.DATABASE_URL}}
```

**ClickHouse connection fails:**
```bash
# Test connection manually
curl 'https://YOUR_HOST.clickhouse.cloud:8443/?query=SELECT%201' \
  --user 'default:YOUR_PASSWORD'
```

**Clerk auth not working:**
- Verify CLERK_JWKS_URL is correct and accessible
- Check CLERK_SECRET_KEY matches your dashboard
- Ensure JWT token is not expired

---

## Cost Summary

**Monthly costs:**
- Railway (PostgreSQL + Redis + Compute): ~$25-40
- ClickHouse Cloud (Development): $29
- **Total: ~$55-70/month**

**Break-even:** 4-5 Lunch Money subscribers at $14/month

---

## Next Steps After Deployment

1. ✅ **Phase 1 Complete:** Backend is live!
2. 🚀 **Phase 2:** Deploy frontend with Clerk integration
3. 📝 **Phase 3:** Update documentation and launch publicly
4. 📊 **Phase 4:** Monitor metrics and gather feedback

---

## Useful Commands

```bash
# Deploy updates
cd backend/services/api-gateway && railway up

# View logs
railway logs --service prela-api-gateway --tail

# Run database query
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM users;"

# Test Stripe webhook
stripe trigger checkout.session.completed

# Generate secure JWT secret
openssl rand -base64 32
```

---

## Support Resources

- **Full Guide:** [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md) - 100+ page detailed guide
- **Checklist:** [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Step-by-step checklist
- **Env Template:** [.env.railway.template](.env.railway.template) - All environment variables
- **Railway Docs:** https://docs.railway.app
- **Railway Discord:** https://discord.gg/railway

---

**Ready to deploy?** Start with the full guide: [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md)

**Created:** February 5, 2026
