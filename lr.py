import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("GPA_vs_Placement_Package.csv")
plt.scatter(data['GPA'], data['Package(LPA)'])
plt.xlabel("GPA")
plt.ylabel("Placement Package (LPA)")
plt.title("GPA vs Placement Package")
plt.show()
plt.figure(figsize=(8,5))
plt.scatter(data['GPA'], data['Package(LPA)'], color='dodgerblue', s=70, edgecolor='k')
plt.xlabel("GPA", fontsize=12)
plt.ylabel("Placement Package (LPA)", fontsize=12)
plt.title("GPA vs Placement Package", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

import numpy as np

x = data['GPA']
y = data['Package(LPA)']

# Best fit line (Linear Regression manually)
m, b = np.polyfit(x, y, 1)  # slope (m) and intercept (b)

plt.figure(figsize=(8,5))
plt.scatter(x, y, color='skyblue', label='Data points')
plt.plot(x, m*x + b, color='red', linewidth=2, label=f'Best Fit: y = {m:.2f}x + {b:.2f}')
plt.xlabel("GPA")
plt.ylabel("Placement Package (LPA)")
plt.title("GPA vs Placement Package (with Regression Line)")
plt.legend()
plt.show()

