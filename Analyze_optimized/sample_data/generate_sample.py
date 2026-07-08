"""Generate synthetic sample sales data with realistic messy characteristics."""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports"]
MANAGERS = ["Alice Chen", "Bob Martinez", "Carol Singh", "David Kim", "Eve Okafor", "Frank Dubois"]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%B %d, %Y"]


def random_date(start_dt, end_dt):
    delta = end_dt - start_dt
    dt = start_dt + timedelta(days=random.randint(0, delta.days))
    return dt.strftime(random.choice(DATE_FORMATS))


def generate(n_rows: int = 550) -> pd.DataFrame:
    rows = []
    start, end = datetime(2023, 1, 1), datetime(2024, 12, 31)

    for i in range(n_rows):
        base_revenue = np.random.lognormal(mean=7.5, sigma=1.2)
        if i % 7 == 0:
            base_revenue *= random.uniform(8, 15)
        revenue = f"${base_revenue:,.2f}" if i % 3 == 0 else round(base_revenue, 2)

        units = max(1, int(np.random.normal(120, 40)))
        if i % 9 == 0:
            units = None

        discount_val = max(0, round(np.random.normal(12, 8), 1))
        discount = f"{discount_val}%" if i % 4 == 0 else discount_val

        returns = None if random.random() < 0.25 else max(0, int(np.random.normal(8, 5)))
        sparse_col = None if random.random() < 0.85 else random.randint(1, 100)

        date_str = random_date(start, end)
        if i % 30 == 0:
            date_str = "not-a-date"

        rows.append({
            "Date": date_str, "Region": random.choice(REGIONS), "Category": random.choice(CATEGORIES),
            "Manager": random.choice(MANAGERS), "Revenue": revenue, "Units_Sold": units,
            "Discount_Pct": discount, "Returns": returns, "Sparse_Column": sparse_col,
        })

    for _ in range(5):
        rows.append({k: None for k in rows[0].keys()})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
    generate().to_csv(out_path, index=False)
    print(f"Generated -> {out_path}")
