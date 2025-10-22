from __future__ import annotations
import json
import re
import pathlib


def test_paper_has_no_missing_placeholders_when_thresholds_pass():
    """Ensure the research paper contains no unresolved [[MISSING:...]] placeholders."""
    status_path = pathlib.Path("docs/threshold_status.json")
    paper_path = pathlib.Path("docs/MAIE_Research_Paper.md")
    
    assert paper_path.exists(), "Research paper missing"
    text = paper_path.read_text(encoding="utf-8")

    # Always forbid dangling placeholders in the paper
    assert "[[MISSING:" not in text, "Paper contains unresolved [[MISSING:...]] placeholders"

    # If thresholds file exists and indicates fail, the paper must say so explicitly in Observability/Release Gates section.
    if status_path.exists():
        status = json.loads(status_path.read_text())
        failed = status.get("status") == "FAILED"
        if failed:
            assert "threshold" in text.lower() and "fail" in text.lower(), \
                "Thresholds failed but paper does not state the failure explicitly"


def test_paper_references_actual_file_paths():
    """Ensure paper references actual code paths that exist."""
    paper_path = pathlib.Path("docs/MAIE_Research_Paper.md")
    text = paper_path.read_text(encoding="utf-8")
    
    # Extract all file paths mentioned in the paper
    file_paths = re.findall(r'`([^`]+\.py)`|`([^`]+\.yaml)`|`([^`]+\.yml)`|`([^`]+\.json)`', text)
    all_paths = [path for match in file_paths for path in match if path]
    
    # Check that referenced files exist
    missing_files = []
    for path in all_paths:
        if not pathlib.Path(path).exists():
            missing_files.append(path)
    
    assert not missing_files, f"Paper references non-existent files: {missing_files}"


def test_paper_uses_numbers_json_data():
    """Ensure paper uses actual data from docs/numbers.json where available."""
    numbers_path = pathlib.Path("docs/numbers.json")
    paper_path = pathlib.Path("docs/MAIE_Research_Paper.md")
    
    if not numbers_path.exists():
        return  # No numbers to validate
    
    numbers = json.loads(numbers_path.read_text())
    text = paper_path.read_text(encoding="utf-8")
    
    # Check that expected panel facts are used
    expected_panel = numbers.get("expected_panel", {})
    if expected_panel.get("shape"):
        shape_str = str(expected_panel["shape"])
        assert shape_str in text, f"Expected panel shape {shape_str} not found in paper"
    
    # Check that backtest metrics are used if available
    backtest = numbers.get("backtest", {}).get("constrained", {})
    if backtest.get("sharpe_annual") is not None:
        sharpe = backtest["sharpe_annual"]
        assert str(sharpe) in text, f"Sharpe ratio {sharpe} not found in paper"
