import pandas as pd
#from IdentifyOutliers import CustomZscoreScaler
from IdentifyOutliers.CustomZscoreScaler import CustomZscoreScaler

import requests as req



# Sample DataFrame
data = {
    'A': [1, 2, 3, 100, 5],
    'B': [5, 6, 7, 8, 500]
}
df = pd.DataFrame(data)

# Initialize the scaler with a z-score threshold (default is 3.0)
scaler = CustomZscoreScaler(threshold=1.9)

# Transform the data
df_no_outliers, df_scaled_no_outliers, df_outliers, df_scaled_outliers = scaler.transform(df)

print('*'*50)
print(df_no_outliers)
print('*'*50)
print(df_scaled_no_outliers)
print('*'*50)
print(df_outliers)
print('*'*50)
print(df_scaled_outliers)
print('*'*50)
