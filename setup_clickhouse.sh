#!/bin/bash
# One-time ClickHouse schema setup script
# Run this after creating your ClickHouse Cloud instance

# Check if credentials are provided
if [ -z "$CLICKHOUSE_HOST" ] || [ -z "$CLICKHOUSE_PASSWORD" ]; then
    echo "❌ Error: Please set CLICKHOUSE_HOST and CLICKHOUSE_PASSWORD"
    echo ""
    echo "Usage:"
    echo "  export CLICKHOUSE_HOST='your-host.clickhouse.cloud'"
    echo "  export CLICKHOUSE_PASSWORD='your-password'"
    echo "  ./backend/setup_clickhouse.sh"
    exit 1
fi

echo "🔧 Setting up ClickHouse schema..."
echo "   Host: $CLICKHOUSE_HOST"
echo ""

# Create the prela database first
echo "📦 Creating 'prela' database..."
curl -s "https://${CLICKHOUSE_HOST}:8443/" \
  --user "default:${CLICKHOUSE_PASSWORD}" \
  --data-binary "CREATE DATABASE IF NOT EXISTS prela"

if [ $? -eq 0 ]; then
    echo "✅ Database 'prela' created successfully"
else
    echo "❌ Failed to create database"
    exit 1
fi

echo ""
echo "📊 Creating tables and views..."

# Execute the schema (need to prefix table names with prela.)
curl -s "https://${CLICKHOUSE_HOST}:8443/" \
  --user "default:${CLICKHOUSE_PASSWORD}" \
  --data-binary @backend/services/trace-service/app/db/clickhouse_schemas.sql

if [ $? -eq 0 ]; then
    echo "✅ Schema created successfully"
else
    echo "❌ Failed to create schema"
    exit 1
fi

echo ""
echo "🔍 Verifying setup..."

# Verify tables were created
TABLES=$(curl -s "https://${CLICKHOUSE_HOST}:8443/?query=SHOW+TABLES+FROM+prela" \
  --user "default:${CLICKHOUSE_PASSWORD}")

echo "Tables created:"
echo "$TABLES"

echo ""
echo "✅ ClickHouse setup complete!"
echo ""
echo "📝 Save these credentials for Railway deployment:"
echo "   CLICKHOUSE_HOST=$CLICKHOUSE_HOST"
echo "   CLICKHOUSE_PORT=8443"
echo "   CLICKHOUSE_USER=default"
echo "   CLICKHOUSE_PASSWORD=<your-password>"
echo "   CLICKHOUSE_DATABASE=prela"
