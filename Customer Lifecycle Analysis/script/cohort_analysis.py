import pandas as pd
import numpy as np
import os

# --- Data Generation (for reproducibility) ---
# This part ensures the script can run even if the CSV is not present.
if not os.path.exists('synthetic_orders.csv'):
    print("Generating 'synthetic_orders.csv' as it was not found...")
    num_users = 500
    num_orders = 5000
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-05-31')

    # Create users with different cohort start dates
    user_ids = range(1, num_users + 1)
    first_purchase_dates = np.random.choice(pd.to_datetime(pd.date_range(start_date, end_date, freq='D')), num_users)
    user_cohorts = {user_id: date for user_id, date in zip(user_ids, first_purchase_dates)}

    # Generate orders
    order_user_ids = np.random.choice(user_ids, num_orders)
    order_dates = []
    for user_id in order_user_ids:
        user_start_date = user_cohorts[user_id]
        random_day_offset = np.random.geometric(p=0.2, size=1)[0] * 7
        order_date = user_start_date + pd.to_timedelta(random_day_offset, unit='D')
        if order_date <= end_date:
            order_dates.append(order_date)

    order_user_ids = order_user_ids[:len(order_dates)]

    df_gen = pd.DataFrame({
        'order_id': range(1, len(order_dates) + 1),
        'user_id': order_user_ids,
        'order_date': order_dates,
        'order_value': np.random.uniform(10, 200, len(order_dates)).round(2)
    })
    df_gen.to_csv('synthetic_orders.csv', index=False)
    print("✅ Data generation complete.")


# --- Part 1: Data Preparation ---

# Load the dataset
try:
    df = pd.read_csv('synthetic_orders.csv')
except FileNotFoundError:
    print("Error: 'synthetic_orders.csv' not found. Please ensure it is in the same directory.")
    exit()

# Ensure 'order_date' is a datetime object
df['order_date'] = pd.to_datetime(df['order_date'])

# Step 1: Get the first purchase month (cohort) for each customer
df['order_month'] = df['order_date'].dt.to_period('M')
df['first_purchase_month'] = df.groupby('user_id')['order_month'].transform('min')

# *** THIS IS THE CORRECTED LINE ***
# Step 2: Calculate the cohort index (number of months since the first purchase)
# We must use the .dt accessor to get .year and .month from a Series.
df['cohort_index'] = (df['order_month'].dt.year - df['first_purchase_month'].dt.year) * 12 + \
                     (df['order_month'].dt.month - df['first_purchase_month'].dt.month)

# Step 3: Create the retention count matrix
cohort_data = df.groupby(['first_purchase_month', 'cohort_index'])['user_id'].nunique().reset_index()
cohort_counts = cohort_data.pivot_table(index='first_purchase_month',
                                        columns='cohort_index',
                                        values='user_id')

# Step 4: Calculate the retention rate matrix (as floats 0.0 to 1.0)
cohort_size = cohort_counts.iloc[:, 0]
retention_matrix = cohort_counts.divide(cohort_size, axis=0)


# --- Part 2: Enhanced Churn Risk Reporting (Integrated and Corrected) ---

print("\n✅ Starting Enhanced Churn Risk Reporting...")

# Rename columns for clarity
retention_matrix.columns = [f'Month_{col}' for col in retention_matrix.columns]
retention_rates = retention_matrix.copy()

# Ensure 'Month_2' exists for calculations
if 'Month_2' not in retention_rates.columns:
    print("Warning: Not enough historical data (less than 3 months) to calculate Month 2 churn.")
    retention_rates['Month_2'] = np.nan
    cohort_counts[2] = np.nan # Use integer key for original counts df

# Identify new cohorts BEFORE filling NaN values
# A NaN in the original cohort_counts for month 2 means the cohort is too new.
is_new_cohort = cohort_counts[2].isna()

# Fill NaN values to avoid errors in math
retention_rates.fillna(0, inplace=True)

# Calculate % drop from Month 0 to Month 2
retention_rates['PctDrop_M0_M2'] = (retention_rates['Month_0'] - retention_rates['Month_2']) * 100

# Flag churn risk using NumPy conditions, checking 'New Cohort' first
conditions = [
    (is_new_cohort),
    (retention_rates['PctDrop_M0_M2'] > 70),
    (retention_rates['PctDrop_M0_M2'] > 40),
]
choices = [
    'New Cohort',
    'High Risk',
    'Moderate Risk',
]
retention_rates['ChurnFlag'] = np.select(conditions, choices, default='Stable')

# Create the final report
risk_report = retention_rates[['Month_0', 'Month_2', 'PctDrop_M0_M2', 'ChurnFlag']].copy()
risk_report['Alert'] = np.where(
    risk_report['ChurnFlag'] == 'High Risk', '⚠️ Immediate Action Needed',
    np.where(
        risk_report['ChurnFlag'] == 'Moderate Risk', '🔍 Investigate Cohort',
        np.where(
            risk_report['ChurnFlag'] == 'New Cohort', '⏳ Too Early to Tell',
            '✅ Healthy Retention'
        )
    )
)

# Format for printing
risk_report['Month_0'] *= 100
risk_report['Month_2'] *= 100

# Present the final report
print("\n--- Churn Risk Analysis Report ---")
print(risk_report.sort_index(ascending=False))

# Export to CSV
risk_report.to_csv('cohort_churn_risk_report.csv')
print("\n✅ Churn risk report saved to 'cohort_churn_risk_report.csv'")

# Bonus: Format for executive presentation
try:
    styled_report = risk_report.sort_index(ascending=False).style.format({
        'Month_0': '{:.0f}%',
        'Month_2': '{:.0f}%',
        'PctDrop_M0_M2': '{:.1f}%'
    }).apply(lambda x: [
        'background-color: #ffcccc' if v == 'High Risk'
        else 'background-color: #fff3cd' if v == 'Moderate Risk'
        else 'background-color: #d4edda' if v == 'Stable'
        else 'background-color: #f8f9fa'
        for v in x
    ], subset=['ChurnFlag'])

    styled_report.to_excel('formatted_churn_risk_report.xlsx', engine='openpyxl', index=True)
    print("✅ Beautifully formatted Excel report saved to 'formatted_churn_risk_report.xlsx'")
except ImportError:
    print("\nWarning: 'openpyxl' is not installed. Cannot create styled Excel report. Run 'pip install openpyxl'.")
except Exception as e:
    print(f"\nAn error occurred while creating the Excel report: {e}")