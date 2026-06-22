import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("movie_dataset.csv", encoding="latin1")

# Rating Distribution
sns.histplot(df['Rating'].dropna(), bins=20)
plt.title("Movie Rating Distribution")
plt.show()

# Top Genres
plt.figure(figsize=(10,5))
df['Genre'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Genres")
plt.show()