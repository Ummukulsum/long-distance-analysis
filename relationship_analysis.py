import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("us_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

print("\n--- Relationship Timeline Overview ---\n")
print(df.sort_values("date")[["date","event_label","significance_score"]])

print("\n--- Descriptive Statistics ---\n")
print(df.describe())

print("\nTotal Major Milestones:",
      len(df[df["significance_score"] >= 9]))

print("Average Growth Index:",
      round(df["growth_index"].mean(), 2))

# Plot emotional trend
plt.figure()
plt.plot(
    df.sort_values("date")["significance_score"],
    color="#FC6A80",
    linewidth=2
)
plt.title("Emotional Significance Trend")
plt.xlabel("Event Index")
plt.ylabel("Significance Score")
plt.grid(alpha=0.3)
plt.show()


# Correlation
print("\n--- Correlation Matrix ---\n")
print(df[["significance_score","growth_index"]].corr())

# Long-term projection
def long_term_projection(years):
    base = 100
    growth_rate = 0.15
    return base * (1 + growth_rate) ** years

print("\nProjected Stability in 10 Years:",
      round(long_term_projection(10), 2))
