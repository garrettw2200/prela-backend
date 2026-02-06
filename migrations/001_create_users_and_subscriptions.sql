-- Migration: Create users and subscriptions tables
-- Date: 2026-02-01
-- Description: User authentication and subscription management for Lunch Money tier

-- Users table (synced from Clerk)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_id TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL,
    full_name TEXT,
    profile_image_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_clerk_id ON users(clerk_id);
CREATE INDEX idx_users_email ON users(email);

-- Subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    tier TEXT NOT NULL DEFAULT 'free',  -- 'free', 'lunch-money', 'pro', 'enterprise'
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'canceled', 'past_due', 'trialing'
    trace_limit INT NOT NULL DEFAULT 100000,  -- 100k for free, 1M for lunch-money
    monthly_usage INT DEFAULT 0,
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_stripe_customer ON subscriptions(stripe_customer_id);
CREATE INDEX idx_subscriptions_stripe_subscription ON subscriptions(stripe_subscription_id);
CREATE INDEX idx_subscriptions_tier ON subscriptions(tier);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);

-- API keys table (for SDK authentication)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT UNIQUE NOT NULL,  -- bcrypt hash of the key
    key_prefix TEXT NOT NULL,  -- First 8 chars for display (e.g., "prela_sk_abc123...")
    name TEXT NOT NULL,  -- User-defined name for the key
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);

-- Usage tracking table (for monthly limits)
CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    traces_count INT DEFAULT 0,
    spans_count INT DEFAULT 0,
    storage_bytes BIGINT DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_records_user ON usage_records(user_id);
CREATE INDEX idx_usage_records_subscription ON usage_records(subscription_id);
CREATE INDEX idx_usage_records_period ON usage_records(period_start, period_end);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE users IS 'User accounts synced from Clerk authentication';
COMMENT ON TABLE subscriptions IS 'Subscription tiers and limits (free, lunch-money, pro, enterprise)';
COMMENT ON TABLE api_keys IS 'API keys for SDK authentication (generated in dashboard)';
COMMENT ON TABLE usage_records IS 'Monthly usage tracking for enforcing tier limits';

COMMENT ON COLUMN subscriptions.tier IS 'Subscription tier: free (100k traces), lunch-money (1M traces), pro (unlimited), enterprise (custom)';
COMMENT ON COLUMN subscriptions.status IS 'Subscription status: active, canceled, past_due, trialing';
COMMENT ON COLUMN subscriptions.trace_limit IS 'Monthly trace limit for this tier';
COMMENT ON COLUMN api_keys.key_hash IS 'bcrypt hash of the API key for secure storage';
COMMENT ON COLUMN api_keys.key_prefix IS 'First 8 characters for display in dashboard (e.g., prela_sk_abc123...)';
