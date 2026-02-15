"""Suite 10: API Key Lifecycle — E2E Tests

Tests API key creation, listing, revocation, and usage for trace ingestion.
Exercises the full CRUD lifecycle against real PostgreSQL.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from .conftest import auth_headers, make_trace


class TestCreateApiKey:
    """POST /api/v1/api-keys — create a new API key."""

    @pytest.mark.asyncio
    async def test_create_api_key_returns_full_key(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Creating an API key returns the full key starting with prela_sk_."""
        resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={"name": "Test Key"},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("prela_sk_")
        assert data["key_prefix"] == data["key"][:16]
        assert data["name"] == "Test Key"
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_create_api_key_without_name_fails(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Creating an API key without a name returns 422."""
        resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 422


class TestListApiKeys:
    """GET /api/v1/api-keys — list user's API keys."""

    @pytest.mark.asyncio
    async def test_list_includes_created_key(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """A newly created key appears in the list."""
        # Create a new key
        create_resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={"name": "Listed Key"},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert create_resp.status_code == 201

        # List keys (includes the fixture key + the new one)
        list_resp = await api_gateway_client.get(
            "/api/v1/api-keys",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert list_resp.status_code == 200
        keys = list_resp.json()
        names = [k["name"] for k in keys]
        assert "Listed Key" in names


class TestRevokeApiKey:
    """DELETE /api/v1/api-keys/{id} — revoke an API key."""

    @pytest.mark.asyncio
    async def test_revoke_removes_from_list(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Revoking a key removes it from the list."""
        # Create a key to revoke
        create_resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={"name": "To Revoke"},
            headers=auth_headers(pro_user["api_key"]),
        )
        key_id = create_resp.json()["id"]

        # Revoke it
        revoke_resp = await api_gateway_client.delete(
            f"/api/v1/api-keys/{key_id}",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert revoke_resp.status_code == 204

        # Verify it's gone
        list_resp = await api_gateway_client.get(
            "/api/v1/api-keys",
            headers=auth_headers(pro_user["api_key"]),
        )
        key_ids = [k["id"] for k in list_resp.json()]
        assert str(key_id) not in [str(kid) for kid in key_ids]


class TestKeyWorksForIngestion:
    """A created API key can be used to ingest traces."""

    @pytest.mark.asyncio
    async def test_new_key_authenticates_ingest(
        self, api_gateway_client: AsyncClient, ingest_client: AsyncClient,
        pro_user: dict,
    ):
        """A newly created key can authenticate trace ingestion."""
        # Create a new key
        create_resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={"name": "Ingest Key"},
            headers=auth_headers(pro_user["api_key"]),
        )
        new_key = create_resp.json()["key"]

        # Use it to ingest a trace
        trace = make_trace(trace_id="ingest_test_001")
        ingest_resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers(new_key),
        )
        assert ingest_resp.status_code == 200
        assert ingest_resp.json()["status"] == "accepted"


class TestRevokedKeyRejected:
    """A revoked API key should be rejected."""

    @pytest.mark.asyncio
    async def test_revoked_key_returns_401(
        self, api_gateway_client: AsyncClient, ingest_client: AsyncClient,
        pro_user: dict,
    ):
        """After revocation, the key returns 401 on ingest."""
        # Create and revoke
        create_resp = await api_gateway_client.post(
            "/api/v1/api-keys",
            json={"name": "Revoke Test"},
            headers=auth_headers(pro_user["api_key"]),
        )
        new_key = create_resp.json()["key"]
        key_id = create_resp.json()["id"]

        await api_gateway_client.delete(
            f"/api/v1/api-keys/{key_id}",
            headers=auth_headers(pro_user["api_key"]),
        )

        # Try to use it
        trace = make_trace(trace_id="revoked_test_001")
        ingest_resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers(new_key),
        )
        assert ingest_resp.status_code == 401


class TestMultipleKeys:
    """Users can create and manage multiple API keys."""

    @pytest.mark.asyncio
    async def test_create_three_keys_all_listed(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Creating 3 keys results in all 3 (+ fixture key) appearing in list."""
        names = ["Key Alpha", "Key Beta", "Key Gamma"]
        for name in names:
            resp = await api_gateway_client.post(
                "/api/v1/api-keys",
                json={"name": name},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp.status_code == 201

        list_resp = await api_gateway_client.get(
            "/api/v1/api-keys",
            headers=auth_headers(pro_user["api_key"]),
        )
        listed_names = [k["name"] for k in list_resp.json()]
        for name in names:
            assert name in listed_names
