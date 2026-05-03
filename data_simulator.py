import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

os.makedirs("data", exist_ok=True)

while True:
    age = np.random.randint(18, 60)
    income = np.random.randint(20000, 120000)
    spending = np.random.randint(1, 100)

    if spending > 70:
        segment = "Premium"
    elif spending > 40:
        segment = "High Value"
    elif spending > 20:
        segment = "Mid Value"
    else:
        segment = "Low Value"

    row = {
        "age": age,
        "income": income,
        "spending_score": spending,
        "segment": segment,
        "timestamp": datetime.now()
    }

    pd.DataFrame([row]).to_csv(
        "data/live_data.csv",
        mode='a',
        index=False,
        header=not os.path.exists("data/live_data.csv")
    )

    print("Generated:", row)
    time.sleep(2)