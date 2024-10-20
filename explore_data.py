import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/EV_Dataset.csv')

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check basic statistics
print("\nBasic Statistics:\n", df.describe(include='all'))
