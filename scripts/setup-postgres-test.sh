#!/bin/bash
# Setup PostgreSQL for local testing
# Based on the CI configuration in .github/workflows/rust.yml

set -e

echo "🐘 Setting up PostgreSQL for agentic-api tests..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Stop and remove existing test container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^agentic-postgres-test$'; then
    echo "📦 Stopping existing test container..."
    docker stop agentic-postgres-test >/dev/null 2>&1 || true
    docker rm agentic-postgres-test >/dev/null 2>&1 || true
fi

# Start PostgreSQL container
echo "🚀 Starting PostgreSQL container..."
docker run -d \
    --name agentic-postgres-test \
    -e POSTGRES_DB=agentic_api_test \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    --health-cmd "pg_isready -U postgres -d agentic_api_test" \
    --health-interval 5s \
    --health-timeout 5s \
    --health-retries 10 \
    postgres:17-alpine

# Wait for PostgreSQL to be healthy
echo "⏳ Waiting for PostgreSQL to be ready..."
timeout=30
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker inspect --format='{{.State.Health.Status}}' agentic-postgres-test 2>/dev/null | grep -q "healthy"; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
done

if [ $elapsed -ge $timeout ]; then
    echo ""
    echo "❌ PostgreSQL failed to start within ${timeout} seconds"
    docker logs agentic-postgres-test
    exit 1
fi

echo ""
echo "✅ PostgreSQL test database is running!"
echo ""
echo "🔧 Connection details:"
echo "   URL: postgresql://postgres:postgres@localhost:5432/agentic_api_test?sslmode=disable"
echo ""
echo "📝 To run PostgreSQL tests, set the environment variable:"
echo "   export TEST_POSTGRES_URL='postgresql://postgres:postgres@localhost:5432/agentic_api_test?sslmode=disable'"
echo ""
echo "🧪 Then run tests with:"
echo "   cargo test --package agentic-server-core --test postgres_tenant_isolation_test -- --ignored"
echo ""
echo "🛑 To stop the test database:"
echo "   docker stop agentic-postgres-test"
echo "   docker rm agentic-postgres-test"
