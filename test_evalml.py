import numpy as np
import pandas as pd
from evalml import AutoMLSearch

df = pd.DataFrame([i for i in range(100)])
df["first"] = np.random.choice([1, 2, 3, 8, 23], 100)
y = pd.Series([np.random.choice([1, 2, 3, 8, 23], 100)])


aml = AutoMLSearch(df, y, "regression")
aml.search()
print(aml.results)
print(type(aml.results))
print(type(aml.results["pipeline_results"]))


