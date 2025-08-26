import pandas as pd
import numpy as np


df = pd.read_csv('iris_1.csv')

print("Original Dataset:")
print(df)
print(f"\nOriginal shape: {df.shape}")

print("\n=== MISSING VALUES ANALYSIS ===")
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)


print("\n=== HANDLING MISSING VALUES ===")


for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().any():
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
            print(f"Filled missing values in '{col}' with mode: {mode_val}")
        else:
            df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in '{col}' with 'Unknown'")

print("\n=== REMOVING DUPLICATES ===")
original_count = len(df)
df_before_dedup = df.copy()


df = df.drop_duplicates()
after_dedup_count = len(df)

print(f"Removed {original_count - after_dedup_count} duplicate rows")
print(f"Rows remaining: {after_dedup_count}")


print("\n=== STANDARDIZING DATA FORMATS ===")


text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    if col not in ['Join_Date']: 
        original_unique = df[col].nunique()
        df[col] = df[col].astype(str).str.strip().str.lower()
        new_unique = df[col].nunique()
        print(f"Standardized '{col}': {original_unique} -> {new_unique} unique values")


date_columns = ['Join_Date']  
for col in date_columns:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted '{col}' to datetime format")
        except Exception as e:
            print(f"Error converting '{col}' to datetime: {e}")


print("\n=== FINAL DUPLICATE REMOVAL ===")
before_final_dedup = len(df)
df = df.drop_duplicates()
after_final_dedup = len(df)

if before_final_dedup != after_final_dedup:
    print(f"Removed {before_final_dedup - after_final_dedup} additional duplicates after standardization")


print("\n=== CLEANING SUMMARY ===")
print(f"Original rows: {original_count}")
print(f"Final rows: {len(df)}")
print(f"Rows removed: {original_count - len(df)}")
print(f"Missing values remaining: {df.isnull().sum().sum()}")

print("\nCleaned Dataset:")
print(df)

df.to_csv('cleaned_dataset_iris.csv', index=False)
print("\n✅ Data cleaning completed!")
print("Cleaned dataset saved as 'cleaned_dataset_iris.csv'")


print("\n=== FINAL DATA INFO ===")
print(df.info())
print("\nFirst few rows of cleaned data:")
print(df.head())
