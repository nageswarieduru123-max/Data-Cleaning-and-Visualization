import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sample_dataset.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove outliers using IQR
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Save cleaned data
df.to_csv("cleaned_dataset.csv", index=False)

# Histogram
sns.histplot(df[numeric_cols[0]], kde=True)
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
