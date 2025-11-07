import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Unemployment in India.csv")
data.columns = data.columns.str.strip()
data.dropna(inplace=True)
data['Region'] = data['Region'].str.strip()

plt.figure(figsize=(10,6))
sns.histplot(data['Estimated Unemployment Rate (%)'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Unemployment Rate in India')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data, estimator='mean', errorbar=None)
plt.title('Average Unemployment Rate by Region')
plt.xticks(rotation=90)
plt.ylabel('Average Unemployment Rate (%)')
plt.show()

plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='Region', data=data)
plt.title('Unemployment Rate Trend Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(bbox_to_anchor=(1,1))
plt.show()
