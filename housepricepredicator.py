import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("data.csv")
print("Dataset:")
print(data.head())
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict([[1500, 3, 2]])
print("\nPredicted price for (1500 sq ft, 3 bed, 2 bath):", prediction)

score = model.score(X_test, y_test)
print("Model Accuracy:", score)

plt.scatter(data['area'], data['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()