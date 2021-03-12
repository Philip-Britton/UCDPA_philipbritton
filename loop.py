import pandas as pd
import seaborn as sns
import numpy as np

# For Looping & iterrows



# Adding Company Name length as a column using for looping & itterows
for lab, row in forbes.iterrows():
    forbes.loc[lab, "name_length"] = len(row["Company"])
    print(forbes.head())