import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset and prepare features
df = pd.read_csv('EV_Dataset.csv')
yearly_sales = df.groupby(['State', 'Year'], as_index=False)['EV_Sales_Quantity'].sum()
yearly_sales['Cumulative_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].cumsum()
yearly_sales['Previous_Year_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].shift(1)
yearly_sales['Yearly_Growth_Rate'] = (
    (yearly_sales['EV_Sales_Quantity'] - yearly_sales['Previous_Year_Sales']) /
    yearly_sales['Previous_Year_Sales'].replace(0, pd.NA)
).fillna(0) * 100

# Prepare features and target
X = yearly_sales[['Year', 'Cumulative_Sales', 'Yearly_Growth_Rate']]
y = yearly_sales['EV_Sales_Quantity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)

# Print evaluation metrics
print("Random Forest - MAE:", mean_absolute_error(y_test, y_pred))
print("Random Forest - R² Score:", r2_score(y_test, y_pred))
