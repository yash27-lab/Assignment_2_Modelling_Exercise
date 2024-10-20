import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/EV_Dataset.csv')

# Step 1: Aggregate EV sales by State and Year to avoid duplicate entries
yearly_sales = df.groupby(['State', 'Year'], as_index=False)['EV_Sales_Quantity'].sum()

# Step 2: Sort by State and Year to ensure the cumulative sum is calculated correctly
yearly_sales = yearly_sales.sort_values(by=['State', 'Year'])

# Step 3: Calculate Cumulative Sales for each state
yearly_sales['Cumulative_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].cumsum()

# Step 4: Calculate Previous Year Sales using shift(1)
yearly_sales['Previous_Year_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].shift(1)

# Step 5: Calculate Yearly Growth Rate and handle division by zero gracefully
yearly_sales['Yearly_Growth_Rate'] = (
    (yearly_sales['EV_Sales_Quantity'] - yearly_sales['Previous_Year_Sales']) /
    yearly_sales['Previous_Year_Sales'].replace(0, pd.NA)
).fillna(0) * 100

# Step 6: Display the first few rows to verify the results
print("\nYearly Aggregated Data with Growth Rate:")
print(yearly_sales.head(10))

