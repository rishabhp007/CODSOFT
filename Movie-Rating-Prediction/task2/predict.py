import joblib
import numpy as np

# Load trained model
model = joblib.load("movie_rating_model.pkl")

# Example movie features
sample_movie = np.array([[10, 50, 100, 120, 130]])

# Predict rating
prediction = model.predict(sample_movie)

print("Predicted Rating:", prediction[0])