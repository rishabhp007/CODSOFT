import joblib
import numpy as np

model = joblib.load("model.pkl")

# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
passenger = np.array([[1, 0, 25, 0, 0, 80, 2]])

prediction = model.predict(passenger)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")