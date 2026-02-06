-- Migration: Update subscription tiers for new pricing model
-- Date: 2026-02-02
-- Description: Add overage tracking for Pro tier usage-based billing

-- Add overage tracking columns to subscriptions table
ALTER TABLE subscriptions
ADD COLUMN IF NOT EXISTS overage_traces INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_users INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_hallucination_checks INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_drift_baselines INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_ai_nlp_searches INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS overage_retention_days INT DEFAULT 0;

-- Create usage_overages table for detailed Pro tier tracking
CREATE TABLE IF NOT EXISTS usage_overages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,

    -- Trace overages
    traces_included INT NOT NULL,
    traces_used INT NOT NULL,
    traces_overage INT DEFAULT 0,
    traces_overage_cost DECIMAL(10,2) DEFAULT 0.00,

    -- User overages
    users_included INT NOT NULL,
    users_active INT NOT NULL,
    users_overage INT DEFAULT 0,
    users_overage_cost DECIMAL(10,2) DEFAULT 0.00,

    -- AI feature overages
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

    -- Retention overages
    retention_days_included INT DEFAULT 30,
    retention_days_used INT DEFAULT 30,
    retention_overage_cost DECIMAL(10,2) DEFAULT 0.00,

    -- Totals
    total_overage_cost DECIMAL(10,2) DEFAULT 0.00,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_overages_subscription ON usage_overages(subscription_id);
CREATE INDEX idx_usage_overages_period ON usage_overages(period_start, period_end);

-- Update comments for documentation
COMMENT ON COLUMN subscriptions.tier IS 'Subscription tier: free (50k traces), lunch-money (100k traces, $14/mo), pro (1M traces, $79/mo + overages), enterprise (unlimited, custom pricing)';
COMMENT ON COLUMN subscriptions.trace_limit IS 'Monthly trace limit for this tier: free=50k, lunch-money=100k, pro=1M, enterprise=unlimited';

COMMENT ON TABLE usage_overages IS 'Pro tier overage tracking for billing (traces, users, AI features). Used to calculate usage-based charges at end of billing period.';

COMMENT ON COLUMN usage_overages.traces_overage_cost IS 'Cost for traces over 1M limit: $8 per 100k traces';
COMMENT ON COLUMN usage_overages.users_overage_cost IS 'Cost for users over 5: $12 per additional user';
COMMENT ON COLUMN usage_overages.ai_hallucination_checks_cost IS 'Cost for hallucination checks over 10k: $5 per 10k checks';
COMMENT ON COLUMN usage_overages.ai_drift_baselines_cost IS 'Cost for drift baselines over 50: $2 per 10 baselines';
COMMENT ON COLUMN usage_overages.ai_nlp_searches_cost IS 'Cost for NLP searches over 1k: $3 per 1k searches';
COMMENT ON COLUMN usage_overages.retention_overage_cost IS 'Cost for retention beyond 90 days: $10 per 30 days';
