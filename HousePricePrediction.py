import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "area": [
        650, 800, 900, 1000, 1100, 1200, 1250, 1300, 1400, 1500,
        1550, 1600, 1700, 1800, 1850, 2000, 2100, 2200, 2400, 2600,
        750, 950, 1150, 1450, 1750, 1950, 2300, 2500, 1350, 1650
    ],
    "bedrooms": [
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        3, 3, 3, 3, 4, 4, 4, 4, 4, 5,
        1, 2, 2, 3, 3, 4, 4, 5, 3, 3
    ],
    "bathrooms": [
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 2, 3, 3, 3, 3, 3, 4, 4,
        1, 1, 2, 2, 3, 3, 4, 4, 2, 3
    ],
    "price_lakhs": [
        28, 32, 38, 45, 48, 52, 58, 60, 65, 70,
        72, 78, 76, 85, 92, 98, 105, 112, 125, 140,
        30, 40, 50, 68, 82, 95, 120, 135, 63, 80
    ]
}

df = pd.DataFrame(data)

print(df.head())

df = pd.DataFrame(data)

print("Dataset Preview:\n")
#print(df.head())

X = df[["area", "bedrooms", "bathrooms"]]
Y = df["price_lakhs"]
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =42)
#print(X_train)

model = LinearRegression()
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)
print("\nModel Performance:", round(score * 100, 2), "%")

new_house = [[1500, 3, 2]]
predicted_price = model.predict(new_house)
print("\nPredicted House Price:")
print("Rs.", round(predicted_price[0], 2), "Lakhs")