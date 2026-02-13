"""Shared configuration for Prela backend services."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables.

    Environment variables can be provided via:
    - Railway environment variables (production)
    - .env file (local development)
    """

    # Service Configuration
    service_name: str = "prela-service"
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"

    # Database (Railway Postgres)
    database_url: str = ""

    # ClickHouse Cloud
    clickhouse_host: str = ""
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_port: int = 8443
    clickhouse_database: str = "prela"

    # Upstash Kafka
    kafka_bootstrap_servers: str = ""
    kafka_username: str = ""
    kafka_password: str = ""
    kafka_topic_traces: str = "traces"
    kafka_topic_spans: str = "spans"

    # Redis (Railway or Upstash)
    redis_url: str = ""

    # Auth & Security
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # Clerk Authentication
    clerk_publishable_key: str = ""
    clerk_secret_key: str = ""
    clerk_jwks_url: str = ""  # e.g., https://your-app.clerk.accounts.dev/.well-known/jwks.json

    # API Configuration
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Rate Limiting
    rate_limit_per_minute: int = 100

    # Security: Payload Size Limits (in bytes)
    max_compressed_payload_size: int = 10 * 1024 * 1024  # 10MB compressed
    max_decompressed_payload_size: int = 50 * 1024 * 1024  # 50MB decompressed
    max_decompression_ratio: int = 10  # Max 10x expansion (zip bomb detection)
    max_webhook_payload_size: int = 5 * 1024 * 1024  # 5MB for webhooks
    max_webhook_data_items: int = 1000  # Max items in webhook data array

    # SMTP Configuration (for email notifications)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_ADDRESS: str = ""

    # Stripe Configuration
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_lunch_money_price_id: str = ""  # Price ID for $14/month
    stripe_pro_price_id: str = ""  # Price ID for $79/month base

    # Pro Tier Overage Price IDs (metered billing)
    stripe_pro_traces_price_id: str = ""  # $8 per 100k traces
    stripe_pro_users_price_id: str = ""  # $12 per user
    stripe_pro_ai_hallucination_price_id: str = ""  # $5 per 10k checks
    stripe_pro_ai_drift_price_id: str = ""  # $2 per 10 baselines
    stripe_pro_ai_nlp_price_id: str = ""  # $3 per 1k searches
    stripe_pro_retention_price_id: str = ""  # $10 per 30 days

    # Data Source Integration
    data_source_encryption_key: str = ""  # Fernet key for encrypting API secrets
    data_source_sync_interval_minutes: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
