-- Combined schema for E2E tests
-- Sources: migrations/001, 002, 003

-- ============================================
-- 001: Users, subscriptions, API keys
-- ============================================

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

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    tier TEXT NOT NULL DEFAULT 'free',
    status TEXT NOT NULL DEFAULT 'active',
    trace_limit INT NOT NULL DEFAULT 100000,
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

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT UNIQUE NOT NULL,
    key_prefix TEXT NOT NULL,
    name TEXT NOT NULL,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);

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

-- ============================================
-- 002: Overage tracking
-- ============================================

ALTER TABLE subscriptions
ADD COLUMN IF NOT EXISTS overage_traces INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_users INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_hallucination_checks INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_drift_baselines INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_nlp_searches INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_retention_days INT DEFAULT 0;

CREATE TABLE IF NOT EXISTS usage_overages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    traces_included INT NOT NULL,
    traces_used INT NOT NULL,
    traces_overage INT DEFAULT 0,
    traces_overage_cost DECIMAL(10,2) DEFAULT 0.00,
    users_included INT NOT NULL,
    users_active INT NOT NULL,
    users_overage INT DEFAULT 0,
    users_overage_cost DECIMAL(10,2) DEFAULT 0.00,
    ai_hallucination_checks_included INT DEFAULT 0,
    ai_hallucination_checks_used INT DEFAULT 0,
    ai_hallucination_checks_overage INT DEFAULT 0,
    ai_hallucination_checks_cost DECIMAL(10,2) DEFAULT 0.00,
    ai_drift_baselines_included INT DEFAULT 0,
    ai_drift_baselines_used INT DEFAULT 0,
    ai_drift_baselines_overage INT DEFAULT 0,
    ai_drift_baselines_cost DECIMAL(10,2) DEFAULT 0.00,
    ai_nlp_searches_included INT DEFAULT 0,
    ai_nlp_searches_used INT DEFAULT 0,
    ai_nlp_searches_overage INT DEFAULT 0,
    ai_nlp_searches_cost DECIMAL(10,2) DEFAULT 0.00,
    retention_days_included INT DEFAULT 30,
    retention_days_used INT DEFAULT 30,
    retention_overage_cost DECIMAL(10,2) DEFAULT 0.00,
    total_overage_cost DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_overages_subscription ON usage_overages(subscription_id);
CREATE INDEX idx_usage_overages_period ON usage_overages(period_start, period_end);

-- ============================================
-- 003: Data sources
-- ============================================

CREATE TABLE IF NOT EXISTS data_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    config JSONB NOT NULL,
    status TEXT DEFAULT 'active',
    error_message TEXT,
    last_sync_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_data_sources_user_id ON data_sources(user_id);
CREATE INDEX IF NOT EXISTS idx_data_sources_project_id ON data_sources(project_id);
CREATE INDEX IF NOT EXISTS idx_data_sources_status ON data_sources(status);

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 004: Debug session overage tracking
-- ============================================

ALTER TABLE usage_overages
ADD COLUMN IF NOT EXISTS ai_debug_sessions_included INT DEFAULT 50,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_used INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_overage INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_cost DECIMAL(10,2) DEFAULT 0.00;

-- ============================================
-- 007: Add team_id to subscriptions (no FK — team tables not needed for E2E tests)
-- ============================================

ALTER TABLE subscriptions
ADD COLUMN IF NOT EXISTS team_id UUID;
