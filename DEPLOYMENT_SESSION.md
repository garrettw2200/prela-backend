# Railway Deployment Session - February 5, 2026

**Started:** February 5, 2026
**Account:** garrettwhite_@outlook.com
**Status:** In Progress

---

## Session Log

### Step 1: Create Railway Project ✅ IN PROGRESS

**Account verified:** garrettwhite_@outlook.com

**Next actions:**
1. Create new Railway project
2. Add PostgreSQL database
3. Add Redis database
4. Save connection strings

---

## Required Information Checklist

Before proceeding, please have these ready:

### ClickHouse Cloud
- [ ] Host: `____________.clickhouse.cloud`
- [ ] Password: `_______________________`
- [ ] Database: `prela` (default)

### Clerk Authentication
- [ ] Publishable Key: `pk_test_____________` or `pk_live_____________`
- [ ] Secret Key: `sk_test_____________` or `sk_live_____________`
- [ ] JWKS URL: `https://______.clerk.accounts.dev/.well-known/jwks.json`

### Stripe Configuration
- [ ] Secret Key: `sk_test_51SwRmIGw4cezTOlQ...` (from STRIPE_CONFIGURATION.md)
- [ ] Lunch Money Price ID: `price_1SwT6ZGw4cezTOlQ8WiBaznB`
- [ ] Pro Price ID: `price_1SwT7OGw4cezTOlQZfg2iMsm`
- [ ] Webhook Secret: Will be created later (leave blank for now)

### Other
- [ ] JWT Secret: Will generate with `openssl rand -base64 32`
- [ ] Domain access: Can configure DNS for `api.prela.app`

---

## Commands to Run

### Generate JWT Secret
```bash
openssl rand -base64 32
```

Save output: `_________________________________`

---

## Notes

- Using Railway CLI for faster deployment
- All sensitive data will be stored in Railway environment variables
- No credentials will be committed to git
