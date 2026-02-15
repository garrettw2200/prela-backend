-- Migration: Add debug session overage tracking
-- Date: 2026-02-14

ALTER TABLE usage_overages
ADD COLUMN IF NOT EXISTS ai_debug_sessions_included INT DEFAULT 50,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_used INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_overage INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS ai_debug_sessions_cost DECIMAL(10,2) DEFAULT 0.00;
