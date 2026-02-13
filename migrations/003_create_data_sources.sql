-- Migration: Create data_sources table for external integrations
-- Date: 2026-02-13
-- Description: Stores connections to external observability platforms (Langfuse, etc.)

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

COMMENT ON TABLE data_sources IS 'External data source connections (Langfuse, etc.) for importing traces';
COMMENT ON COLUMN data_sources.project_id IS 'Prela project ID that imported traces are assigned to';
COMMENT ON COLUMN data_sources.type IS 'Source type: langfuse, otlp, file_upload';
COMMENT ON COLUMN data_sources.config IS 'Connection config: { host, public_key, encrypted_secret_key, last_synced_timestamp }';
COMMENT ON COLUMN data_sources.status IS 'Connection status: active, error, paused';
