import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv('EV_Dataset.csv')

# Aggregate sales and compute the additional features
yearly_sales = df.groupby(['State', 'Year'], as_index=False)['EV_Sales_Quantity'].sum()
yearly_sales['Cumulative_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].cumsum()
yearly_sales['Previous_Year_Sales'] = yearly_sales.groupby('State')['EV_Sales_Quantity'].shift(1)
yearly_sales['Yearly_Growth_Rate'] = (
    (yearly_sales['EV_Sales_Quantity'] - yearly_sales['Previous_Year_Sales']) / 
    yearly_sales['Previous_Year_Sales'].replace(0, pd.NA)
).fillna(0) * 100

# Prepare the features and target variable
X = yearly_sales[['Year', 'Cumulative_Sales', 'Yearly_Growth_Rate']]
y = yearly_sales['EV_Sales_Quantity']

# Split the data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
