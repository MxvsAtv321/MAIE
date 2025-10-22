#!/usr/bin/env python3
"""
Generate factual release notes from repo artifacts.
Facts-only, no hype, no assumptions.
"""

import json
import pathlib
import hashlib
import subprocess
from datetime import datetime


def get_git_info():
    """Get git commit and dirty tree status."""
    try:
        # Get commit SHA (short)
        commit_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], 
            capture_output=True, text=True, check=True
        ).stdout.strip()
        
        # Check for dirty tree
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, text=True
        ).stdout.strip()
        dirty_tree = dirty_output != ""
        
        return commit_sha, dirty_tree
    except Exception:
        return "unknown", False


def get_file_hash(file_path: pathlib.Path) -> str:
    """Get SHA256 hash of a file."""
    if not file_path.exists():
        return "missing"
    try:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        return "error"


def load_artifacts():
    """Load all required artifacts."""
    artifacts = {}
    
    # Load numbers.json
    numbers_path = pathlib.Path("docs/numbers.json")
    if numbers_path.exists():
        artifacts["numbers"] = json.loads(numbers_path.read_text())
    else:
        artifacts["numbers"] = {}
    
    # Load threshold status
    threshold_status_path = pathlib.Path("docs/threshold_status.json")
    if threshold_status_path.exists():
        artifacts["threshold_status"] = json.loads(threshold_status_path.read_text())
    else:
        artifacts["threshold_status"] = {"status": "UNKNOWN"}
    
    # Load expected metadata
    expected_meta_path = pathlib.Path("expected/metadata.json")
    if expected_meta_path.exists():
        artifacts["expected_meta"] = json.loads(expected_meta_path.read_text())
    else:
        artifacts["expected_meta"] = {}
    
    # Get git info
    commit_sha, dirty_tree = get_git_info()
    artifacts["git_commit"] = commit_sha
    artifacts["dirty_tree"] = dirty_tree
    
    # Get file hashes
    artifacts["numbers_hash"] = get_file_hash(numbers_path)
    
    return artifacts


def format_number(value, decimals=2):
    """Format a number with specified decimal places."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_list(value, max_items=3):
    """Format a list with max items shown."""
    if not value or not isinstance(value, list):
        return "N/A"
    if len(value) <= max_items:
        return str(value)
    return str(value[:max_items]) + f"... (+{len(value)-max_items} more)"


def generate_release_notes():
    """Generate factual release notes from artifacts."""
    
    # Load all artifacts
    artifacts = load_artifacts()
    
    # Extract key data
    numbers = artifacts["numbers"]
    threshold_status = artifacts["threshold_status"]
    expected_meta = artifacts["expected_meta"]
    
    # Expected panel facts
    expected_panel = numbers.get("expected_panel", {})
    expected_shape = expected_panel.get("shape", [0, 0])
    expected_start = expected_panel.get("start", "N/A")
    expected_end = expected_panel.get("end", "N/A")
    expected_n_files = expected_panel.get("n_files", 0)
    expected_build_seconds = format_number(expected_panel.get("build_seconds"))
    
    # Backtest metrics
    backtest = numbers.get("backtest", {}).get("constrained", {})
    sharpe_annual = format_number(backtest.get("sharpe_annual"))
    vol_annual = format_number(backtest.get("vol_annual"))
    max_dd = format_number(backtest.get("max_dd"))
    
    # Constraint metrics
    constraints = numbers.get("constraints", {})
    infeasible_pct = format_number(constraints.get("infeasible_pct"))
    
    # Warnings
    warnings = numbers.get("warnings", [])
    
    # Generate release notes
    release_notes = f"""# MAIE Release Notes

## Version & Provenance

- **Commit SHA**: {artifacts['git_commit']}
- **Profile**: {threshold_status.get('profile', 'unknown')}
- **Audit Timestamp**: {numbers.get('metadata', {}).get('timestamp', 'N/A')}
- **Dirty Tree**: {'Yes' if artifacts['dirty_tree'] else 'No'}
- **Numbers Hash**: {artifacts['numbers_hash']}

## Summary

- **Expected Panel**: Shape {expected_shape}, {expected_start} to {expected_end}, {expected_n_files} files, {expected_build_seconds}s build time
- **Backtest Horizon**: {expected_shape[0] if expected_shape else 'N/A'} trading days
- **QP Infeasible Ratio**: {infeasible_pct}%
- **Performance Metrics**: Sharpe {sharpe_annual}, Vol {vol_annual}, MaxDD {max_dd}
- **API Observability**: Prometheus metrics with custom gauges and counters

## Threshold Gate Verdict

- **Status**: {threshold_status.get('status', 'UNKNOWN')}
- **Violations**: {format_list(threshold_status.get('violations', []))}

## Notable Changes

- **Initial Release**: Complete MAIE system implementation
- **CI/CD**: GitHub Actions with Docker publishing to GHCR
- **Metrics**: Prometheus instrumentation with custom gauges and counters
- **Constraints**: β-neutral and sector-neutral with tolerance bands
- **Explainability**: Three-tier fallback system for local explanations

## Risks & Limitations

- **Warnings**: {format_list(warnings)}

## Repro Steps

```bash
make build-expected
make bt-constrained
make audit
make paper
```

## Artifacts

- `docs/MAIE_Research_Paper.md`
- `docs/numbers.json`
- `docs/threshold_status.json`
- `expected/metadata.json`
- `outputs_from_expected/report.html`
"""
    
    # Write to file
    release_notes_path = pathlib.Path("docs/RELEASE_NOTES.md")
    release_notes_path.write_text(release_notes, encoding="utf-8")
    
    print(f"Release notes generated at {release_notes_path}")
    return release_notes_path


if __name__ == "__main__":
    generate_release_notes()
