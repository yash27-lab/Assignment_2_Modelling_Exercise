import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/EV_Dataset.csv')

# Aggregate EV sales by State and Year
aggregated_data = df.groupby(['State', 'Year'])['EV_Sales_Quantity'].sum().reset_index()

# Sort the aggregated data to identify top-performing states and years
aggregated_data = aggregated_data.sort_values(by='EV_Sales_Quantity', ascending=False)

# Display the top 10 results
print("\nTop 10 States by Total EV Sales:")
print(aggregated_data.head(10))
