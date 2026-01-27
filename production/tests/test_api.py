"""Tests for API endpoints."""
import pytest
import sys
sys.path.append('..')


# Note: These tests require trained models to be available
# They are integration tests rather than unit tests

def test_health_endpoint():
    """Test health endpoint."""
    # Import here to avoid issues if Flask is not available
    try:
        from src.api.app import app

        with app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200
            data = response.get_json()
            assert 'status' in data
            assert data['status'] == 'healthy'
    except ImportError:
        pytest.skip("Flask not available")


def test_model_info_endpoint():
    """Test model info endpoint."""
    try:
        from src.api.app import app

        with app.test_client() as client:
            response = client.get('/model/info')
            # May return 503 if no model loaded
            assert response.status_code in [200, 503]
    except ImportError:
        pytest.skip("Flask not available")
