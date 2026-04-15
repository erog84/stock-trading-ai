"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestPortfolioEndpoints:
    def test_get_summary(self):
        response = client.get("/api/portfolio/summary")
        assert response.status_code == 200

    def test_get_positions(self):
        response = client.get("/api/portfolio/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_trades(self):
        response = client.get("/api/portfolio/trades")
        assert response.status_code == 200

    def test_get_account(self):
        response = client.get("/api/portfolio/account")
        assert response.status_code == 200
        data = response.json()
        assert "cash" in data
        assert "total_value" in data


class TestModelEndpoints:
    def test_get_training_status(self):
        response = client.get("/api/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_get_metrics_no_model(self):
        response = client.get("/api/models/metrics")
        assert response.status_code == 404


class TestSignalEndpoints:
    def test_signals_endpoint(self):
        response = client.get("/api/signals/latest?tickers=AAPL")
        # 404 if no model trained, 200 if model exists from prior run
        assert response.status_code in (200, 404)
