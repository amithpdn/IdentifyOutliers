# IdentifyOutliers

A Python package for efficient scaling and outlier handling of pandas DataFrames using the some of the most popular outlier elimination approaches.

`IdentifyOutliers` is designed to provide a seamless experience in preprocessing pandas DataFrames by ensuring data normalization and outlier handling in one step.

## Features

- **Data Scaling**: Utilizes the standard scaler, min_max scaler and robust scaler methods for data normalization.
- **Outlier Detection**: Provides an option to set thresholds for outlier detection.
- **Multiple Outputs**: Returns the original data, the scaled data without outliers, a separate DataFrame for detected outliers, and scaled outliers.

## Installation

Install the package using pip:

```bash
pip install IdentifyOutliers
```

## Usage

```python
import pandas as pd
from IdentifyOutliers.CustomZscoreScaler import CustomZscoreScaler
from IdentifyOutliers.CustomMinMaxScaler import CustomMinMaxScaler
from IdentifyOutliers.CustomRobustScaler import CustomRobustScaler
from IdentifyOutliers.CustomIQRScaler import CustomIQRScaler

# Sample DataFrame
data = {
    'A': [1, 2, 3, 100, 5],
    'B': [5, 6, 7, 8, 500]
}
df = pd.DataFrame(data)

# Initialize the scalers with attributes. The default values are shown below.
scaler_czs = CustomZscoreScaler(threshold=3.0)
scaler_cms = CustomMinMaxScaler(lower_bound=0.05, upper_bound=0.95)
scaler_crs = CustomRobustScaler(threshold=3.5, mad_multiplier=0.6745)
scaler_cis = CustomIQRScaler(lower_bound=1.5, upper_bound=1.5)


# Transform the data
df_no_outliers_czs, df_scaled_no_outliers_czs, df_outliers_czs, df_scaled_outliers_czs = scaler_czs.transform(df)
df_no_outliers_cms, df_scaled_no_outliers_cms, df_outliers_cms, df_scaled_outliers_cms = scaler_cms.transform(df)
df_no_outliers_crs, df_scaled_no_outliers_crs, df_outliers_crs, df_scaled_outliers_crs = scaler_crs.transform(df)
df_no_outliers_cis, df_scaled_no_outliers_cis, df_outliers_cis, df_scaled_outliers_cis = scaler_cis.transform(df)


# Print the results for CustomZscoreScaler
print(df_no_outliers_czs)
#    A  B
# 0  1  5
# 1  2  6
# 2  3  7

print(df_scaled_no_outliers_czs)
#           A         B
# 0 -0.544672 -0.507592
# 1 -0.518980 -0.502526
# 2 -0.493288 -0.497461

print(df_outliers_czs)
#      A    B
# 3  100    8
# 4    5  500

print(df_outliers_czs)
#      A    B
# 3  100    8
# 4    5  500

print(df_scaled_outliers_czs)
#           A         B
# 3  1.998845 -0.492395
# 4 -0.441904  1.999974

```

## Parameters

##### CustomZscoreScaler:

- `threshold`: The z-score threshold for outlier detection. Data points exceeding threshold standard deviations away from the mean are considered outliers. The default value is 3.0.

##### CustomMinMaxScaler:

- `lower_bound`: The lower bound for outlier detection. Data points below the lower bound are considered outliers. The default value is 0.05.
- `upper_bound`: The upper bound for outlier detection. Data points above the upper bound are considered outliers. The default value is 0.95.

##### CustomRobustScaler:

- `threshold`: The z-score threshold for outlier detection. Data points exceeding threshold standard deviations away from the mean are considered outliers. The default value is 3.5.
- `mad_multiplier`: The MAD multiplier for outlier detection. Data points exceeding the MAD multiplied by the threshold are considered outliers. The default value is 0.6745.

##### CustomIQRScaler:

- `lower_bound`: Multiplier applied to Interquartile Range (IQR) for identifying lower bound. The default value is 1.5 or Q1 - 1.5*IRQ.
- `upper_bound`: Multiplier applied to Interquartile Range (IQR) for identifying upper bound. The default value is 1.5 or Q3 + 1.5*IRQ.

## Contributions

Contributions are welcome! Please create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/amithpdn/IdentifyOutliers/blob/master/LICENSE.TXT).
