"""
Air Quality Prediction using Machine Learning

This project predicts CO concentration using environmental sensor data.
Models used:
- Linear Regression
- Random Forest

Evaluation metrics:
- Mean Squared Error (MSE)
- R² Score
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor #تنبؤ بالقيم العدديه regression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv("AirQuality.csv")
df = df.drop(columns=["Unnamed: 14", "Unnamed: 15"], errors="ignore")

# استبدال الفاصلة بالنقطة
df = df.replace(",", ".", regex=True)

# تحويل الأعمدة الرقمية
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
df = df.drop(columns=["Date", "Time"])

# تقسيم البيانات إلى ميزات وهدف
x= df.drop(columns=["CO(GT)"])
y= df["CO(GT)"]

# تقسيم البيانات إلى مجموعة تدريب واختبار
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#حتى لا تؤثر القيم الكبيرة على النموذج
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#matrix of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")

plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.show()

# بناء نموذج الانحدار الخطي
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# التنبؤ باستخدام النموذج
predictions = model.predict(x_test_scaled)
print(predictions[:5])

# تقييم النموذج
mse= mean_squared_error(y_test, predictions)
r2= r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#vizualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs Predicted CO(GT)")

plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png")
plt.show()

rf= RandomForestRegressor(random_state=42)
rf.fit(x_train_scaled, y_train) #يفهم العلاقة بين x و y

importances = rf.feature_importances_
feature_names = x.columns 


importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
print(importance_df)

sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()

result = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MSE": [mse, mean_squared_error(y_test, rf.predict(x_test_scaled))],
    "R^2 Score": [r2, r2_score(y_test, rf.predict(x_test_scaled))]  
})
print(result)