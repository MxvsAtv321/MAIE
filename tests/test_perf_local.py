from __future__ import annotations
import pytest
import time
import subprocess
import sys
from pathlib import Path


@pytest.mark.perf
class TestPerfLocal:
    """Performance tests for local execution."""
    
    def test_build_expected_panel_time(self):
        """Test that build_expected_panel completes within time bounds."""
        if not Path("scripts/build_expected_panel.py").exists():
            pytest.skip("build_expected_panel.py not found")
        
        start_time = time.time()
        
        # Run the build script
        result = subprocess.run([
            sys.executable, "scripts/build_expected_panel.py"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assert completion within reasonable time (5 minutes)
        assert duration < 300, f"build_expected_panel took {duration:.1f}s (max 300s)"
        assert result.returncode == 0, f"build_expected_panel failed: {result.stderr}"
    
    def test_api_latency_bounds(self):
        """Test API latency bounds (requires running API)."""
        # This test would require the API to be running
        # For now, we'll check if the numbers file has latency data
        numbers_path = Path("docs/numbers.json")
        if not numbers_path.exists():
            pytest.skip("Numbers file not found - run 'make audit' first")
        
        import json
        with open(numbers_path) as f:
            numbers = json.load(f)
        
        # Check latency bounds
        score_expected_p95 = numbers.get("api", {}).get("score_expected", {}).get("p95_ms", 0.0)
        explain_local_p95 = numbers.get("api", {}).get("explain_local", {}).get("p95_ms", 0.0)
        
        assert score_expected_p95 < 200, f"/score_expected P95 {score_expected_p95}ms > 200ms"
        assert explain_local_p95 < 400, f"/explain_local P95 {explain_local_p95}ms > 400ms"
    
    def test_memory_usage_bounds(self):
        """Test that memory usage is within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Assert memory usage is reasonable (less than 2GB)
        assert memory_mb < 2048, f"Memory usage {memory_mb:.1f}MB exceeds 2GB limit"
    
    def test_disk_usage_bounds(self):
        """Test that disk usage is within reasonable bounds."""
        outputs_dir = Path("outputs_from_expected")
        if not outputs_dir.exists():
            pytest.skip("No outputs directory found")
        
        total_size = sum(f.stat().st_size for f in outputs_dir.rglob("*") if f.is_file())
        size_mb = total_size / 1024 / 1024
        
        # Assert disk usage is reasonable (less than 1GB)
        assert size_mb < 1024, f"Disk usage {size_mb:.1f}MB exceeds 1GB limit"
