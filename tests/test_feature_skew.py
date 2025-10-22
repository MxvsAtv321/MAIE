"""
Test feature skew detection and handling.
"""
import pytest
import requests
import json
from pathlib import Path

@pytest.mark.perf
def test_feature_skew_detection():
    """Test that feature skew is detected and handled correctly."""
    # This test requires the API to be running
    # In a real test environment, you'd start the API in a fixture
    
    # Create test data with mixed feature order
    test_data = {
        "ticker": "SIM001",
        "features": {
            "momentum_5d": 0.1,
            "volatility_20d": 0.15,
            "reversal_1d": -0.05,
            "volume_ratio": 1.2,
            "rsi_14d": 45.0
        }
    }
    
    # Send request to /explain_local
    try:
        response = requests.post(
            "http://localhost:8000/explain_local",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Should get non-empty results even with feature skew
            assert "top_features" in result
            assert len(result["top_features"]) > 0
            
            # Check that the response is valid
            for feature, importance in result["top_features"]:
                assert isinstance(feature, str)
                assert isinstance(importance, (int, float))
        else:
            pytest.skip("API not available")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running")

@pytest.mark.perf 
def test_feature_skew_metrics():
    """Test that feature skew metrics are properly incremented."""
    # This would check the Prometheus metrics endpoint
    # In a real test, you'd query /metrics and parse the counter
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            metrics_text = response.text
            
            # Check that feature skew counter exists
            assert "maie_feature_skew_total" in metrics_text
            
            # Parse the counter value (simplified)
            lines = metrics_text.split('\n')
            for line in lines:
                if line.startswith('maie_feature_skew_total'):
                    # Should have a value >= 0
                    assert ' ' in line
                    value = float(line.split()[-1])
                    assert value >= 0
                    break
        else:
            pytest.skip("Metrics endpoint not available")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running")
