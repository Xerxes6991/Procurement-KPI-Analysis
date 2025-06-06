import pandas as pd

# Loading the dataset
df = pd.read_csv("C:/Users/USER/Downloads/Procurement KPI Analysis Dataset.csv")

# Displaying the head of the dataframe to understand its structure
print(df.head(5))

# Checking for missing values and data types in the dataframe
missing_values =df.isnull().sum()
data_types = df.dtypes

# Displaying the missing values and data types
print(missing_values)
print(data_types)

# Filling missing values in the 'Delivery_Date' and 'Defective_Units' columns
# For 'Delivery_Date', we can fill with the next available date (forward fill)
# For 'Defective_Units', we can fill with 0 as a reasonable assumption

df['Delivery_Date'].fillna(method='ffill', inplace=True)
df['Defective_Units'].fillna(0, inplace=True)

# Checking the dataframe again to confirm that missing values have been filled
missing_values_after = df.isnull().sum()
print(missing_values_after)

# Clean Compliance column
df['Compliance'] = df['Compliance'].str.strip().str.capitalize()


# Convert to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

# Temporary column for existing delivery days
df['Delivery_Days_Calculated'] = (df['Delivery_Date'] - df['Order_Date']).dt.days
print(df['Delivery_Days_Calculated'])

# Compute median delivery days per Item_Category
category_medians = df[df['Delivery_Days_Calculated'].notnull()].groupby('Item_Category')['Delivery_Days_Calculated'].median()
print(category_medians)

# Impute missing Delivery_Date based on category median
def impute_delivery_date(row):
    if pd.isnull(row['Delivery_Date']):
        median_days = category_medians.get(row['Item_Category'], 10)
        return row['Order_Date'] + pd.Timedelta(days=median_days)
    return row['Delivery_Date']

df['Delivery_Date'] = df.apply(impute_delivery_date, axis=1)

# Recalculate delivery days
df['Delivery_Days'] = (df['Delivery_Date'] - df['Order_Date']).dt.days
print(df['Delivery_Days'])

# Feature engineering
df['Total_Cost'] = df['Quantity'] * df['Unit_Price']
df['Negotiated_Cost'] = df['Quantity'] * df['Negotiated_Price']
df['Cost_Savings'] = df['Total_Cost'] - df['Negotiated_Cost']
df['Defect_Rate'] = df['Defective_Units'] / df['Quantity']

print(df['Total_Cost'])
print(df['Negotiated_Cost'])
print(df['Negotiated_Price'])
print(df['Cost_Savings'])
print(df['Defect_Rate'])

# Drop temporary column
df.drop(columns='Delivery_Days_Calculated', inplace=True)

# Display cleaned and imputed dataset
print(df.head())           # For a quick look at the first few rows
print(df.info())           # Summary of columns and non-null counts
print(df.describe())       # Summary statistics for numeric columns

# SUPPLIER RISK ANALYSIS

# Define thresholds for "high risk"
defect_threshold = 0.1  # 10% or more defective
delay_threshold = 13    # more than 13 days to deliver

# Identify high-risk records
df['Is_High_Defect'] = df['Defect_Rate'] >= defect_threshold
df['Is_Delayed'] = df['Delivery_Days'] > delay_threshold

print(df['Is_High_Defect'])
print(df['Is_Delayed'])

# Aggregate risk per supplier
risk_summary = df.groupby('Supplier').agg(
    Total_Orders=('PO_ID', 'count'),
    High_Defect_Count=('Is_High_Defect', 'sum'),
    Delayed_Count=('Is_Delayed', 'sum')
)

# Add risk rates
risk_summary['Defect_Rate_%'] = (risk_summary['High_Defect_Count'] / risk_summary['Total_Orders']) * 100
risk_summary['Delay_Rate_%'] = (risk_summary['Delayed_Count'] / risk_summary['Total_Orders']) * 100

# Sort to show high-risk vendors first
risk_summary = risk_summary.sort_values(by=['Defect_Rate_%', 'Delay_Rate_%'], ascending=False)
print(risk_summary)

from tabulate import tabulate

print(tabulate(risk_summary, headers='keys', tablefmt='psql'))

# COST OPTIMISATION

# Summarise Cost Savings
supplier_savings = df.groupby('Supplier').agg(
    Total_Orders=('PO_ID', 'count'),
    Total_Quantity=('Quantity', 'sum'),
    Total_Cost=('Total_Cost', 'sum'),
    Negotiated_Cost=('Negotiated_Cost', 'sum'),
    Total_Savings=('Cost_Savings', 'sum')
).reset_index()

# Add % savings column
supplier_savings['Savings_%'] = (supplier_savings['Total_Savings'] / supplier_savings['Total_Cost']) * 100

# Sort by highest savings
supplier_savings = supplier_savings.sort_values(by='Total_Savings', ascending=False)

print(supplier_savings)

# Sort by Total Savings
top_savings = supplier_savings.sort_values(by='Total_Savings', ascending=False).head(10)
top_savings_percent = supplier_savings.sort_values(by='Savings_%', ascending=False).head(10)

print(top_savings)
print(top_savings_percent)
print(top_savings.columns)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Top 10 Suppliers by Total Savings
plt.figure(figsize=(12, 6))
sns.barplot(x='Total_Savings', y='Supplier', data=top_savings, palette='Blues_d')
plt.title('Top 10 Suppliers by Total Cost Savings')
plt.xlabel('Total Savings (£)')
plt.ylabel('Supplier')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Suppliers by Savings Percentage
plt.figure(figsize=(12, 6))
sns.barplot(x='Savings_%', y='Supplier', data=top_savings_percent, palette='Oranges_d')
plt.title('Top 10 Suppliers by Percentage Savings')
plt.xlabel('Savings (%)')
plt.ylabel('Supplier')
plt.tight_layout()
plt.show()

# TREND FORECAST

# Create time features
df['Order_YearMonth'] = df['Order_Date'].dt.to_period('M')  # e.g., 2023-01

print(df["Order_YearMonth"])

# Group and analyze prices over time
price_trends = df.groupby('Order_YearMonth').agg(
    Avg_Unit_Price=('Unit_Price', 'mean'),
    Avg_Negotiated_Price=('Negotiated_Price', 'mean'),
    Order_Count=('PO_ID', 'count')
).reset_index()

# Convert Period to Timestamp for plotting
price_trends['Order_YearMonth'] = price_trends['Order_YearMonth'].dt.to_timestamp()

import matplotlib.pyplot as plt

# Plot trends
plt.figure(figsize=(12, 6))
plt.plot(price_trends['Order_YearMonth'], price_trends['Avg_Unit_Price'], label='Unit Price', marker='o')
plt.plot(price_trends['Order_YearMonth'], price_trends['Avg_Negotiated_Price'], label='Negotiated Price', marker='o')
plt.title("Monthly Average Price Trends")
plt.xlabel("Order Month")
plt.ylabel("Price (£)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

