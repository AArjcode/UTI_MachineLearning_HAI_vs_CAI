"""Create a small, entirely synthetic dataset with the input format of the scripts.

The values are random and are NOT derived from the study data. They exist only
so the code can be run end to end; results on this file mean nothing clinically.

    python example/make_synthetic_data.py          # writes example/synthetic_example.xlsx
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "PATIENT_AGE": rng.integers(18, 90, n).astype(float),
        "PATIENT_GENDER": rng.choice(["Female", "Male"], n),
        "PATIENT_ETHNICITY": rng.choice(["Hispanic", "Not Hispanic", "Unknown"], n),
        "PATIENT_EMPLOYMENT_STATUS": rng.choice(["Employed", "Unemployed", "Retired"], n),
        "PATIENT_MARITAL_STATUS": rng.choice(["Single", "Married", "Other"], n),
        "Poverty": rng.uniform(0, 40, n),             # % below poverty line (area level)
        "Median_Income": rng.normal(60000, 15000, n),  # area median household income
        "Graduate_degree": rng.uniform(0, 30, n),      # % with a graduate degree
        "population_density": rng.lognormal(6, 1, n),  # people per square mile
        "New Location": rng.choice(["Clinic", "Medical Center", "Hospital"], n),
        "New Nurse Unit": rng.choice(["Inpatient", "Mental Health", "Emergency", "Other"], n),
    })
    df.loc[rng.random(n) < 0.05, "PATIENT_AGE"] = np.nan   # some missing values
    risk = 0.03 + 0.25 * (df["New Nurse Unit"] == "Inpatient")
    df["H"] = (rng.random(n) < risk).astype(int)            # imbalanced binary outcome
    return df


if __name__ == "__main__":
    out = Path(__file__).with_name("synthetic_example.xlsx")
    make().to_excel(out, index=False)
    print(f"Wrote {out}")
