import pandas as pd
import psutil

# Load a subset of your data (e.g., 10,000 rows)
subset_data = pd.read_csv('cosmicclassifierTraining.csv', nrows=10000)

# Measure memory usage of the subset
memory_usage_subset = subset_data.memory_usage(deep=True).sum()

# Get available memory
available_memory = psutil.virtual_memory().available

# Estimate the maximum number of data points
max_data_points = (available_memory // memory_usage_subset) * 10000

print(f"Estimated maximum number of data points: {max_data_points}")