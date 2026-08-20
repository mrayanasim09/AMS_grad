"""
Pytest configuration and margin reporting hooks for AMS-grad test suite.
"""

import pytest


@pytest.fixture(autouse=True)
def print_test_margin(request):
    """Fixture that allows tests to record margins and prints summary upon completion."""
    margins = []
    
    def record_margin(name: str, actual: float, threshold: float, condition: str = ">="):
        if condition in (">=", ">"):
            margin = actual - threshold
        else:
            margin = threshold - actual
        margins.append((name, actual, threshold, condition, margin))
        
    request.node.record_margin = record_margin
    yield
    
    if margins:
        print(f"\n--- Margins for {request.node.name} ---")
        for name, actual, thresh, cond, margin in margins:
            status = "PASS" if margin >= 0 else "FAIL"
            print(f"  [{status}] {name}: actual={actual:+.4f} (req {cond} {thresh:+.4f}, margin={margin:+.4f})")
