import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


house_df = pd.read_csv('cleaned_dataset_house.csv', delim_whitespace=True, header=None)
house_df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

iris_df = pd.read_csv('cleaned_dataset_iris.csv')
sentiment_df = pd.read_csv('cleaned_dataset_sentiment.csv')
stock_df = pd.read_csv('cleaned_dataset_stock.csv')

# ========== HOUSE DATASET VISUALIZATIONS ==========


plt.figure(figsize=(8,5))
sns.barplot(x='CHAS', y='MEDV', data=house_df)
plt.title('Average Median Value by CHAS', fontsize=14, fontweight='bold')
plt.xlabel('CHAS (Charles River dummy variable)')
plt.ylabel('Median Value (MEDV)')
plt.savefig('house_barplot_medv_by_chas.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,5))
plt.plot(house_df.index[:30], house_df['MEDV'][:30], marker='o', linestyle='-', linewidth=2)
plt.title('Median Value Trend (First 30 Houses)', fontsize=14, fontweight='bold')
plt.xlabel('House Index')
plt.ylabel('Median Value (MEDV)')
plt.grid(True, alpha=0.3)
plt.savefig('house_linechart_medv.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,5))
plt.scatter(house_df['RM'], house_df['MEDV'], alpha=0.6, color='darkblue')
plt.title('Scatter Plot: Average Rooms vs Median Value', fontsize=14, fontweight='bold')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median Value (MEDV)')
plt.savefig('house_scatter_rm_medv.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== IRIS DATASET VISUALIZATIONS ==========


plt.figure(figsize=(7,5))
sns.countplot(x='species', data=iris_df, palette='viridis')
plt.title('Count of Iris Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig('iris_species_countbar.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,5))
plt.plot(iris_df.index[:50], iris_df['sepal_length'][:50], marker='o', linestyle='-', color='green', linewidth=2)
plt.title('Sepal Length Trend (First 50 Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True, alpha=0.3)
plt.savefig('iris_sepal_length_line.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris_df, s=60)
plt.title('Petal Length vs Petal Width by Species', fontsize=14, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.savefig('iris_scatter_petal.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== SENTIMENT DATASET VISUALIZATIONS ==========


plt.figure(figsize=(8,5))
sns.countplot(x='Sentiment', data=sentiment_df, order=sentiment_df['Sentiment'].value_counts().index, palette='coolwarm')
plt.title('Distribution of Sentiments in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('sentiment_countbar.png', dpi=300, bbox_inches='tight')
plt.show()


sentiment_df['Timestamp'] = pd.to_datetime(sentiment_df['Timestamp'], dayfirst=True)
sentiment_sorted = sentiment_df.sort_values('Timestamp')

plt.figure(figsize=(12,5))
plt.plot(sentiment_sorted['Timestamp'], sentiment_sorted['Likes'], marker='o', linestyle='-', color='purple', linewidth=2)
plt.title('Likes Trend Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Timestamp')
plt.ylabel('Number of Likes')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sentiment_line_likes_time.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x='Retweets', y='Likes', hue='Sentiment', data=sentiment_df, alpha=0.7)
plt.title('Retweets vs Likes by Sentiment', fontsize=14, fontweight='bold')
plt.xlabel('Number of Retweets')
plt.ylabel('Number of Likes')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('sentiment_scatter_retweets_likes.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== STOCK DATASET VISUALIZATIONS ==========


top_symbols = stock_df['symbol'].value_counts().index[:10]
avg_close = stock_df[stock_df['symbol'].isin(top_symbols)].groupby('symbol')['close'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(x='symbol', y='close', data=avg_close, order=top_symbols, palette='plasma')
plt.title('Average Closing Price for Top 10 Stock Symbols', fontsize=14, fontweight='bold')
plt.xlabel('Stock Symbol')
plt.ylabel('Average Closing Price ($)')
plt.xticks(rotation=45)
plt.savefig('stock_bar_avg_close.png', dpi=300, bbox_inches='tight')
plt.show()


stock_df['date'] = pd.to_datetime(stock_df['date'], dayfirst=True, errors='coerce')
aapl_jan2014 = stock_df[(stock_df['symbol']=='aapl') & (stock_df['date'].dt.month == 1) & (stock_df['date'].dt.year == 2014)]

plt.figure(figsize=(12,6))
plt.plot(aapl_jan2014['date'], aapl_jan2014['close'], marker='o', linestyle='-', linewidth=2, color='red')
plt.title('AAPL Closing Price Trend - January 2014', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_line_aapl_jan2014.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,5))
plt.scatter(aapl_jan2014['volume'], aapl_jan2014['close'], alpha=0.7, color='orange', s=60)
plt.title('AAPL: Trading Volume vs Closing Price', fontsize=14, fontweight='bold')
plt.xlabel('Trading Volume')
plt.ylabel('Closing Price ($)')
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
plt.savefig('stock_scatter_aapl_vol_close.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All visualizations completed and saved as high-quality PNG files!")
