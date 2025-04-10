import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/Priteek Singh Manhas/Downloads/Sub_Division_IMD_2017.csv")

# Fill missing values in monthly and seasonal columns with the median of each column
df.fillna(df.median(numeric_only=True), inplace=True)

# Replace zero values with the mean of each column
numeric_cols = df.select_dtypes(include=['number']).columns  # Get numeric columns
for col in numeric_cols:
    mean_value = df[col].mean()
    df[col] = df[col].replace(0, mean_value)

# Drop rows where 'ANNUAL' is still missing 
df.dropna(subset=['ANNUAL'], inplace=True)

# Ensure 'YEAR' is integer
df['YEAR'] = df['YEAR'].astype(int)

'''Find the Total Rainfall Trend Over the Years for a Subdivision
This helps analyze whether rainfall is increasing or decreasing over time for a specific location.'''

# Select a specific subdivision
subdivision_name = "Jammu & Kashmir"
sub_df = df[df["SUBDIVISION"] == subdivision_name]

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(sub_df["YEAR"], sub_df["ANNUAL"], marker="o", linestyle="-", color="b")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.title(f"Annual Rainfall Trend in {subdivision_name}")
plt.grid(True)
plt.show()

''' Find the Subdivision with the Highest Rainfall in a Given Year
This will help identify regions with extreme rainfall.'''

year = 2015  # Change this year as needed
highest_rainfall = df[df["YEAR"] == year].sort_values(by="ANNUAL", ascending=False).head(1)

print("Subdivision with highest rainfall in", year)
print(highest_rainfall[["SUBDIVISION", "ANNUAL"]])

'''Calculate the Average Monthly Rainfall Across All Subdivisions
This shows which months receive the most rainfall on average.'''
monthly_avg = df.loc[:, "JAN":"DEC"].mean()

# Plot the results
plt.figure(figsize=(10, 5))
monthly_avg.plot(kind="bar", color="teal")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.title("Average Monthly Rainfall Across All Subdivisions")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

''' Identify the Top 5 Wettest and Driest Years for a Subdivision
This helps detect extreme weather years.'''
subdivision_name = "Andaman & Nicobar Islands"  # Change this as needed
sub_df = df[df["SUBDIVISION"] == subdivision_name]

# Find top 5 wettest years
print("Top 5 Wettest Years:")
print(sub_df.nlargest(5, "ANNUAL")[["YEAR", "ANNUAL"]])

# Find top 5 driest years
print("\nTop 5 Driest Years:")
print(sub_df.nsmallest(5, "ANNUAL")[["YEAR", "ANNUAL"]])

'''Compare Monsoon (JJAS) Rainfall Over Decades
This checks whether monsoon rainfall is changing over decades.'''

df["Decade"] = (df["YEAR"] // 10) * 10  # Group by decade
decadal_rainfall = df.groupby("Decade")["JJAS"].mean()

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(decadal_rainfall.index, decadal_rainfall.values, marker="o", linestyle="-", color="purple")
plt.xlabel("Decade")
plt.ylabel("Average Monsoon Rainfall (mm)")
plt.title("Decadal Trend of Monsoon (JJAS) Rainfall")
plt.grid(True)
plt.show()

''' Identify Subdivisions with Decreasing Rainfall Trends
This helps detect regions that are experiencing less rainfall over time, which can indicate drought-prone areas.'''

from scipy.stats import linregress

# Calculate the trend (slope) for each subdivision
trends = []

for sub in df["SUBDIVISION"].unique():
    sub_df = df[df["SUBDIVISION"] == sub]
    slope, _, _, _, _ = linregress(sub_df["YEAR"], sub_df["ANNUAL"])
    trends.append((sub, slope))

# Convert to DataFrame and sort
trend_df = pd.DataFrame(trends, columns=["SUBDIVISION", "Slope"])
trend_df_sorted = trend_df.sort_values(by="Slope")

# Get the top 5 subdivisions with decreasing rainfall trends
print("Top 5 Subdivisions with Decreasing Rainfall Trends:")
print(trend_df_sorted.head(5))

'''Find Correlation Between Monsoon (JJAS) and Annual Rainfall
This checks if monsoon rainfall is a strong predictor of annual rainfall.'''
# Calculate correlation
correlation = df["JJAS"].corr(df["ANNUAL"])
print(f"Correlation between Monsoon (JJAS) and Annual Rainfall: {correlation:.2f}")

# Plot scatter plot
plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["JJAS"], y=df["ANNUAL"], alpha=0.5, color="blue")
plt.xlabel("Monsoon Rainfall (JJAS)")
plt.ylabel("Annual Rainfall")
plt.title("Correlation Between Monsoon and Annual Rainfall")
plt.grid(True)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Group data by YEAR and calculate the mean annual rainfall
rainfall_series = df.groupby("YEAR")["ANNUAL"].mean()

# Fit an ARIMA model (Auto-Regressive Integrated Moving Average)
model = ARIMA(rainfall_series, order=(2, 1, 2))  # ARIMA (p, d, q)
model_fit = model.fit()

# Forecast for the next 10 years
forecast = model_fit.forecast(steps=10)

# Create future years for plotting
future_years = list(range(df["YEAR"].max() + 1, df["YEAR"].max() + 11))

# Plot historical and forecasted rainfall
plt.figure(figsize=(10, 5))
plt.plot(rainfall_series.index, rainfall_series.values, label="Historical Rainfall", marker="o")
plt.plot(future_years, forecast, label="Forecasted Rainfall", linestyle="dashed", marker="o", color="red")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.title("Annual Rainfall Forecast (Next 10 Years)")
plt.legend()
plt.grid(True)
plt.show()

# Print forecasted values
forecast_df = pd.DataFrame({"Year": future_years, "Predicted Rainfall (mm)": forecast})
print(forecast_df)

# Compute correlation matrix
corr_matrix = df.iloc[:, 2:].corr()

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Rainfall Data")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


print(df.info())
print(df.describe())

# Scatter plot
plt.scatter(df["YEAR"], df["JJAS"])
plt.xlabel("Year")
plt.ylabel("Monsoon Rainfall (JJAS) in mm")
plt.title("Year vs JJAS Rainfall")
plt.grid(True)
plt.show()

# Linear Regression model
X = df[["YEAR"]]  # Independent variable (2D)
y = df["JJAS"]    # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict JJAS rainfall for a specific year (e.g., 2025)
check_year = pd.DataFrame({'YEAR': [2025]})
result = model.predict(check_year)
print("Predicted JJAS Rainfall for 2025:", result[0])

# Plot regression line
plt.scatter(X, y, color='blue', label='Actual JJAS Data')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='Regression Line')
plt.xlabel("Year")
plt.ylabel("JJAS Rainfall (mm)")
plt.title("Linear Regression: Year vs JJAS Rainfall")
plt.legend()
plt.grid(True)
plt.show()

