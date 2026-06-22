import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("movie_dataset.csv", encoding="latin1")
print(df.head())
print(df.shape)
print(df.columns)

# Required columns
df = df[['Genre','Director','Actor 1','Actor 2','Actor 3','Rating']]

# Remove missing values
df = df.dropna()

# Encoding categorical columns
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor1 = LabelEncoder()
le_actor2 = LabelEncoder()
le_actor3 = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor1.fit_transform(df['Actor 1'])
df['Actor 2'] = le_actor2.fit_transform(df['Actor 2'])
df['Actor 3'] = le_actor3.fit_transform(df['Actor 3'])

# Features and Target
X = df[['Genre','Director','Actor 1','Actor 2','Actor 3']]
y = df['Rating']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save Model
joblib.dump(model, "movie_rating_model.pkl")

print("Model Saved Successfully")