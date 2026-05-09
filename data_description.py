import pandas as pd
import matplotlib.pyplot as plt

def dataDescription(filePath):

    df = pd.read_csv(filePath)

    print("=" * 60)
    print("DATA DESCRIPTION: ")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n" + "-" * 60)
    print("Shape and basic Info:")
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\n" + "-" * 60)
    print("Descriptive Statistics:")
    print(df.describe())

    print("\n" + "-" * 60)
    print("Class Distribution (Target Variable):")
    counts = df["Risk Level"].value_counts()
    print(counts)
    print(f"\nHigh Risk : {counts['High']} ({counts['High']/len(df)*100:.1f}%)")
    print(f"Low Risk  : {counts['Low']}  ({counts['Low']/len(df)*100:.1f}%)")

    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color=["tomato", "steelblue"], width=0.4)
    plt.title("Class Distribution: Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print("\n" + "-" * 60)
    print("Feature Summary Table:")
    feature_info = {
        "Feature": [
            "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
            "Heart Rate", "Previous Complications", "Preexisting Diabetes",
            "Gestational Diabetes", "Mental Health"
        ],
        "Type": [
            "Continuous", "Continuous", "Continuous", "Continuous", "Continuous",
            "Continuous", "Continuous", "Binary (0/1)", "Binary (0/1)",
            "Binary (0/1)", "Binary (0/1)"
        ],
        "Description": [
            "Patient age in years",
            "Systolic blood pressure (mmHg)",
            "Diastolic blood pressure (mmHg)",
            "Blood glucose level (mmol/L)",
            "Body temperature in Fahrenheit",
            "Body Mass Index",
            "Pulse rate (bpm)",
            "History of obstetric complications",
            "Type 1/2 diabetes prior to pregnancy",
            "Diabetes developed during pregnancy",
            "Presence of mental health conditions"
        ]
    }
    feature_df = pd.DataFrame(feature_info)
    print(feature_df.to_string(index=False))

    return df