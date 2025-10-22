#!/usr/bin/env python3
"""
Diff two docs/numbers.json files and show deltas in release notes.
Useful for comparing releases and generating change summaries.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_numbers(file_path: str) -> Dict[str, Any]:
    """Load numbers from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        sys.exit(1)


def compare_numbers(old_numbers: Dict[str, Any], new_numbers: Dict[str, Any]) -> List[str]:
    """Compare two numbers.json files and return list of changes."""
    changes = []
    
    # Compare expected panel
    old_expected = old_numbers.get("expected_panel", {})
    new_expected = new_numbers.get("expected_panel", {})
    
    if old_expected.get("shape") != new_expected.get("shape"):
        changes.append(f"Expected panel shape: {old_expected.get('shape')} → {new_expected.get('shape')}")
    
    if old_expected.get("n_files") != new_expected.get("n_files"):
        changes.append(f"Expected panel files: {old_expected.get('n_files')} → {new_expected.get('n_files')}")
    
    # Compare backtest metrics
    old_backtest = old_numbers.get("backtest", {}).get("constrained", {})
    new_backtest = new_numbers.get("backtest", {}).get("constrained", {})
    
    for metric in ["sharpe_annual", "vol_annual", "cagr", "max_dd"]:
        old_val = old_backtest.get(metric)
        new_val = new_backtest.get(metric)
        if old_val != new_val:
            changes.append(f"Backtest {metric}: {old_val} → {new_val}")
    
    # Compare constraint residuals
    old_constraints = old_numbers.get("constraints", {})
    new_constraints = new_numbers.get("constraints", {})
    
    for metric in ["infeasible_days", "infeasible_pct"]:
        old_val = old_constraints.get(metric)
        new_val = new_constraints.get(metric)
        if old_val != new_val:
            changes.append(f"Constraint {metric}: {old_val} → {new_val}")
    
    # Compare warnings
    old_warnings = old_numbers.get("warnings", [])
    new_warnings = new_numbers.get("warnings", [])
    
    if old_warnings != new_warnings:
        changes.append(f"Warnings: {len(old_warnings)} → {len(new_warnings)} items")
    
    return changes


def main():
    """Main function to diff two numbers.json files."""
    if len(sys.argv) != 3:
        print("Usage: python diff_numbers.py <old_numbers.json> <new_numbers.json>")
        sys.exit(1)
    
    old_file = sys.argv[1]
    new_file = sys.argv[2]
    
    # Load both files
    old_numbers = load_numbers(old_file)
    new_numbers = load_numbers(new_file)
    
    # Compare and get changes
    changes = compare_numbers(old_numbers, new_numbers)
    
    # Output changes
    if changes:
        print("Changes detected:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("No significant changes detected")


if __name__ == "__main__":
    main()
