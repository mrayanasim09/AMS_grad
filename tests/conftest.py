"""
Pytest configuration and margin reporting hooks for AMS-grad test suite.
"""

import pytest


@pytest.fixture(autouse=True)
def print_test_margin(request):
    """Fixture that allows tests to record margins and prints clean summary upon completion."""
    margins = []

    def record_margin(name: str, actual: float, threshold, condition: str = ">="):
        if condition in (">=", ">"):
            margin = actual - threshold
            passed = actual >= threshold if condition == ">=" else actual > threshold
            cond_str = f"{condition} {threshold:+.4f}"
        elif condition in ("<=", "<"):
            margin = threshold - actual
            passed = actual <= threshold if condition == "<=" else actual < threshold
            cond_str = f"{condition} {threshold:+.4f}"
        elif condition == "within_3sigma":
            center, three_sigma = threshold
            margin = three_sigma - abs(actual - center)
            passed = abs(actual - center) <= three_sigma
            cond_str = f"in [{center - three_sigma:+.4f}, {center + three_sigma:+.4f}]"
        else:
            raise ValueError(f"Unknown condition: {condition}")

        margins.append((name, actual, cond_str, margin, passed))

    request.node.record_margin = record_margin
    yield

    if margins:
        print(f"\n--- Margins for {request.node.name} ---")
        for name, actual, cond_str, margin, passed in margins:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: actual={actual:+.4f} (req {cond_str}, margin={margin:+.4f})")
