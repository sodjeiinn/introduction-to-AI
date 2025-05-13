import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv('housing.csv')


imputer = SimpleImputer(strategy='mean')
df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']] = imputer.fit_transform(df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']])

X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
percentage_r2 = r2 * 100
print(f"Predicted Prices: {y_pred[:5]}")
print(f"Actual Prices: {y_test.values[:5]}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Accuracy (R-squared): {percentage_r2:.2f}%")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

new_data = pd.DataFrame({
        'longitude': [float(input("Введите долготу: "))],
        'latitude': [float(input("Введите широту: "))],
        'housing_median_age': [float(input("Введите медианный возраст дома: "))],
        'total_rooms': [float(input("Введите общее количество комнат: "))],
        'total_bedrooms': [float(input("Введите общее количество спален: "))],
        'population': [float(input("Введите население: "))],
        'households': [float(input("Введите количество домохозяйств: "))],
        'median_income': [float(input("Введите медианный доход: "))]
    })
predicted_prices = model.predict(new_data)
print(f"Предсказанные цены для новых данных: {predicted_prices}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.xlim(y_test.min() - 10000, y_test.max() + 10000)
plt.ylim(y_test.min() - 10000, y_test.max() + 10000)
plt.show()
