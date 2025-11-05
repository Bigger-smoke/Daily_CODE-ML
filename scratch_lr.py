import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("GPA_vs_Placement_Package.csv")
# print(df)
X=df["GPA"].values
print(X)
y=df["Package(LPA)"].values
print(y)

# means
x_mean = np.mean(X)
y_mean = np.mean(y)
# steps to calculate the m (numerator and denominator) and b
num = np.sum((X-x_mean)*(y-y_mean))
den = np.sum((X-x_mean)**2)

# compute slope and intercept ie. m and b

m = num/den
b=y_mean- m*(x_mean)
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

y_hat = m*X + b

#wrapping in class for my own linear regression
class myownlinearregression:
    def __init__(self):
        self.m= None
        self.b= None
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        num = np.sum((X - x_mean) * (y - y_mean))
        den = np.sum((X - x_mean) ** 2)
        self.m = num / den
        self.b = y_mean - self.m * x_mean

    def predict(self, X):
        X = np.array(X)
        return self.m * X + self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - y_mean) ** 2)
        return 1 - SSE / SST
model = myownlinearregression()
model.fit(X, y)

print("Slope (m):", model.m)
print("Intercept (b):", model.b)

# Predict new GPA values
print("Predicted package for GPA 8.0:", model.predict([8.0]))
print("R² Score:", model.score(X, y))





