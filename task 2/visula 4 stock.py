import warnings


warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.ioff()


df = pd.read_csv('cleaned_dataset_stock.csv')


print("Dataset Shape:", df.shape)
print("\nColumn Information:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())


summary_stats = df.describe(include='all')
print("\nSummary Statistics:")
print(summary_stats)


numerical_df = df.select_dtypes(include=['number'])
if not numerical_df.empty:
    mean_values = numerical_df.mean()
    median_values = numerical_df.median()
    std_dev = numerical_df.std()
    
    print("\nMean Values:")
    print(mean_values)
    print("\nMedian Values:")
    print(median_values)
    print("\nStandard Deviation:")
    print(std_dev)


print("\nMode Values:")
for col in df.columns:
    mode_val = df[col].mode()
    if not mode_val.empty:
        print(f"{col}: {mode_val[0]}")




numerical_columns = df.select_dtypes(include=['number']).columns
n_cols = len(numerical_columns)

if n_cols > 0:
    fig, axes = plt.subplots(nrows=(n_cols+2)//3, ncols=3, figsize=(15, 5*((n_cols+2)//3)))
    if n_cols == 1:
        axes = [axes]
    elif n_cols <= 3:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
    
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if n_cols > 0:
    fig, axes = plt.subplots(nrows=(n_cols+2)//3, ncols=3, figsize=(15, 5*((n_cols+2)//3)))
    if n_cols == 1:
        axes = [axes]
    elif n_cols <= 3:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')
    
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if n_cols > 1:
    try:
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            cat_col = categorical_columns[0]
            unique_count = df[cat_col].nunique()
            
            if unique_count <= 10:
                sns.pairplot(df, vars=numerical_columns, hue=cat_col)
            else:
                print(f"Skipping hue for '{cat_col}' - too many unique values ({unique_count})")
                sns.pairplot(df[numerical_columns])
        else:
            sns.pairplot(df[numerical_columns])
        plt.show()
    except KeyboardInterrupt:
        print("Pairplot execution was interrupted by user")
    except Exception as e:
        print(f"Pairplot failed: {e}")
        print("Creating individual scatter plots instead...")
        
        
        import itertools
        num_cols_list = list(numerical_columns)
        pairs = list(itertools.combinations(num_cols_list, 2))[:6]
        
        if pairs:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (x_col, y_col) in enumerate(pairs):
                if i < len(axes):
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=axes[i])
                    axes[i].set_title(f'{x_col} vs {y_col}')
            
            for j in range(len(pairs), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()


categorical_columns = df.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    n_cat_cols = len(categorical_columns)
    fig, axes = plt.subplots(nrows=(n_cat_cols+2)//3, ncols=3, figsize=(15, 5*((n_cat_cols+2)//3)))
    
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()  
    else:
        axes = [axes]  
    
    for i, column in enumerate(categorical_columns):
        sns.countplot(data=df, x=column, ax=axes[i])
        axes[i].set_title(f'Count Plot of {column}')
        axes[i].tick_params(axis='x', rotation=45)
    
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if n_cols > 1:
   
    correlation_matrix = numerical_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    
    
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  
                strong_correlations.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if strong_correlations:
        print("\nStrong Correlations (|r| > 0.7):")
        for corr in strong_correlations:
            print(f"{corr['Feature 1']} - {corr['Feature 2']}: {corr['Correlation']:.3f}")
    else:
        print("\nNo strong correlations found (|r| > 0.7)")


missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values per Column:")
    print(missing_values[missing_values > 0])
else:
    print("\nNo missing values found in the dataset.")


print("\nData Types:")
print(df.dtypes)


if len(categorical_columns) > 0:
    print("\nUnique Values in Categorical Columns:")
    for col in categorical_columns:
        unique_vals = df[col].unique()
        print(f"{col}: {len(unique_vals)} unique values -> {list(unique_vals)}")


if len(categorical_columns) > 0:
    print("\nFrequency Distribution for Categorical Columns:")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

print("\n=== EDA Analysis Complete ===")
