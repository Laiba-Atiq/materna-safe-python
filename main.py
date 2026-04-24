from cleaning import dataCleaning 
from visuals import visualisation
from logistic_regression import logisticRegression,predictRisk

df = dataCleaning("Dataset.csv")
visualisation(df)

logisticRegression(df)

example = {
        "Age": 25, "Systolic BP": 140, "Diastolic": 90, "BS": 11,
        "Body Temp": 98, "BMI": 25.0, "Previous Complications": 1,
        "Preexisting Diabetes": 1, "Gestational Diabetes": 0,
        "Mental Health": 1, "Heart Rate": 82
    }

result = predictRisk(example)
print("\nExample Prediction:")
print(f"Risk Level : {result['label']}")
print(f"P(High) : {result['prob_high']*100:.1f}%")
print(f"P(Low) : {result['prob_low']*100:.1f}%")