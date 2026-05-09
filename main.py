from cleaning import dataCleaning 
from visuals import visualisation
from logistic_regression import logisticRegression,predictRisk as pr1
from random_forest import randomForest,predictRisk as pr2
from xgBoost import xgBoost,predictRisk as pr3
from data_description import dataDescription
from xgBoost import xgBoost

orignal_df = dataDescription("Dataset.csv")
df = dataCleaning(orignal_df)
#visualisation(df)

logisticRegression(df)
randomForest(df)
xgBoost(df)

example = {
        "Age": 22,"Systolic BP": 90,"Diastolic": 60,"BS": 9,"Body Temp": 100,
        "BMI": 18,"Previous Complications": 1,"Preexisting Diabetes": 1,"Gestational Diabetes": 0,
        "Mental Health": 1,"Heart Rate": 80,
    }

result_first = pr1(example)

print("\nExample Prediction:")
print(f"Risk Level : {result_first['label']}")
print(f"P(High) : {result_first['prob_high']*100:.1f}%")
print(f"P(Low) : {result_first['prob_low']*100:.1f}%")

result_sec = pr2(example)

print("\nExample Prediction:")
print(f"Risk Level : {result_sec['label']}")
print(f"P(High) : {result_sec['prob_high']*100:.1f}%")
print(f"P(Low) : {result_sec['prob_low']*100:.1f}%")

result_third = pr3(example)

print("\nExample Prediction:")
print(f"Risk Level : {result_third['label']}")
print(f"P(High) : {result_third['prob_high']*100:.1f}%")
print(f"P(Low) : {result_third['prob_low']*100:.1f}%")