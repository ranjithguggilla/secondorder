import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def toy_orders():
    """A small synthetic order log with a known structure."""
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("1997-01-01")
    for cid in range(400):
        first = start + pd.Timedelta(days=int(rng.integers(0, 80)))
        n = 1 + rng.poisson(1.2)
        days = np.sort(rng.integers(0, 520, n))
        dates = [first] + [first + pd.Timedelta(days=int(d)) for d in days[1:]]
        for d in dates:
            rows.append((f"{cid:05d}", d, int(rng.integers(1, 5)), float(rng.gamma(3, 12))))
    return pd.DataFrame(rows, columns=["customer_id", "date", "number_of_cds", "dollar_value"])
