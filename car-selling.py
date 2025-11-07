import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("car data.csv")

data = pd.get_dummies(data, drop_first=True)

X = data.drop(['Selling_Price'], axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Car Price Prediction Results ")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R² Score:", round(r2, 2))

numeric_cols = [col for col in data.columns if any(key in col for key in ['Year', 'Present_Price', 'Kms', 'Owner', 'Selling_Price'])]
corr = data[numeric_cols].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred, color='orange')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(y_test - y_pred, kde=True, color='skyblue')
plt.title("Residual (Error) Distribution")
plt.show()

