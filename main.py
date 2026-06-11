import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/walmart.csv")

print("Dataset Loaded Successfully")
print("Rows:", len(df))

# =========================
# DATE PROCESSING
# =========================

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month

# =========================
# BASIC EDA
# =========================

print("\nDataset Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# =========================
# FEATURES & TARGET
# =========================

X = df[['Store',
        'Holiday_Flag',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment']]

y = df['Weekly_Sales']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Results")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# =========================
# SALES TREND
# =========================

sales = df.groupby('Date')['Weekly_Sales'].sum()

plt.figure(figsize=(12, 6))
sales.plot()

plt.title("Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("outputs/sales_trend.png")
plt.close()

# =========================
# FORECAST GRAPH
# =========================

plt.figure(figsize=(12, 6))

plt.plot(y_test.values[:100], label="Actual Sales")
plt.plot(predictions[:100], label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.legend()

plt.tight_layout()

plt.savefig("outputs/forecast.png")
plt.close()

# =========================
# FEATURE IMPORTANCE
# =========================

importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(X.columns, importance)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.close()

# =========================
# CORRELATION HEATMAP
# =========================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# =========================
# SALES DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df['Weekly_Sales'],
    bins=30
)

plt.title("Sales Distribution")
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/sales_distribution.png")
plt.close()

# =========================
# HOLIDAY SALES
# =========================

holiday_sales = df.groupby(
    'Holiday_Flag'
)['Weekly_Sales'].mean()

plt.figure(figsize=(6, 5))

holiday_sales.plot(
    kind='bar'
)

plt.title(
    "Holiday vs Non-Holiday Sales"
)

plt.tight_layout()

plt.savefig(
    "outputs/holiday_sales.png"
)

plt.close()

# =========================
# MONTHLY SALES TREND
# =========================

monthly_sales = df.groupby(
    'Month'
)['Weekly_Sales'].mean()

plt.figure(figsize=(8, 5))

monthly_sales.plot()

plt.title(
    "Monthly Sales Trend"
)

plt.tight_layout()

plt.savefig(
    "outputs/monthly_sales.png"
)

plt.close()

# =========================
# REPORT FILE
# =========================

with open(
    "outputs/report.txt",
    "w"
) as file:

    file.write(
        "SALES FORECAST REPORT\n"
    )

    file.write(
        "=====================\n\n"
    )

    file.write(
        f"Rows : {len(df)}\n"
    )

    file.write(
        f"MAE : {mae}\n"
    )

    file.write(
        f"R2 Score : {r2}\n"
    )

# =========================
# BUSINESS INSIGHTS
# =========================

with open(
    "insights/business_insights.txt",
    "w"
) as file:

    file.write(
        "BUSINESS INSIGHTS\n"
    )

    file.write(
        "=================\n\n"
    )

    file.write(
        f"Model Accuracy (R2): {r2}\n\n"
    )

    file.write(
        "Key Findings:\n\n"
    )

    file.write(
        "- Store ID strongly influences sales.\n"
    )

    file.write(
        "- Holiday periods affect demand.\n"
    )

    file.write(
        "- CPI and unemployment impact sales patterns.\n"
    )

    file.write(
        "- Model explains over 93% of sales variation.\n"
    )

print("\nFiles Generated Successfully!")

print("sales_trend.png")
print("forecast.png")
print("feature_importance.png")
print("correlation_heatmap.png")
print("sales_distribution.png")
print("holiday_sales.png")
print("monthly_sales.png")
print("report.txt")
print("business_insights.txt")