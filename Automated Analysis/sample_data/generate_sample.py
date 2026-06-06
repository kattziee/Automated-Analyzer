"""
Generate synthetic sample sales data with realistic messy characteristics:
intentional nulls, currency strings, percentage strings, outliers, and
date format inconsistencies.
"""
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports"]
MANAGERS = ["Alice Chen", "Bob Martinez", "Carol Singh", "David Kim", "Eve Okafor", "Frank Dubois"]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%B %d, %Y"]

def random_date(start_dt, end_dt):
    delta = end_dt - start_dt
    rand_days = random.randint(0, delta.days)
    dt = start_dt + timedelta(days=rand_days)
    fmt = random.choice(DATE_FORMATS)
    return dt.strftime(fmt)

rows = []
start = datetime(2023, 1, 1)
end = datetime(2024, 12, 31)

for i in range(550):
    region = random.choice(REGIONS)
    category = random.choice(CATEGORIES)
    manager = random.choice(MANAGERS)

    # Revenue: sometimes as currency string
    base_revenue = np.random.lognormal(mean=7.5, sigma=1.2)
    if i % 7 == 0:                                  # inject outlier
        base_revenue *= random.uniform(8, 15)
    if i % 3 == 0:
        revenue = f"${base_revenue:,.2f}"
    else:
        revenue = round(base_revenue, 2)

    # Units
    units = max(1, int(np.random.normal(120, 40)))
    if i % 9 == 0:
        units = None                                # inject null

    # Discount: sometimes as percentage string
    discount_val = max(0, round(np.random.normal(12, 8), 1))
    if i % 4 == 0:
        discount = f"{discount_val}%"
    else:
        discount = discount_val

    # Returns: 25% null
    returns = None if random.random() < 0.25 else max(0, int(np.random.normal(8, 5)))

    # Sparse column (will be dropped by cleaner)
    sparse_col = None if random.random() < 0.85 else random.randint(1, 100)

    # Bad date injection
    date_str = random_date(start, end)
    if i % 30 == 0:
        date_str = "not-a-date"                    # inject unparseable date

    rows.append({
        "Date": date_str,
        "Region": region,
        "Category": category,
        "Manager": manager,
        "Revenue": revenue,
        "Units_Sold": units,
        "Discount_Pct": discount,
        "Returns": returns,
        "Sparse_Column": sparse_col,
    })

# Add a few fully empty rows
for _ in range(5):
    rows.append({k: None for k in rows[0].keys()})

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    pass
