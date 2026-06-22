import joblib

model = joblib.load("iris_model.pkl")

prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

flowers = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

print("Predicted Flower:", flowers[prediction[0]])