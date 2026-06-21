import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic-dataset.csv")

# Survival Count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.show()

# Class vs Survival
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.show()
