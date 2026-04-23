import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataCleaning(filePath):
    df = pd.read_csv(filePath)

    #missing values
    print("Missing values:")

    missingVal = df.isnull().sum()
    missingValPct = (missingVal / len(df)) * 100
    missingValDf = pd.DataFrame({
        'missingValCount': missingVal,
        'missingValPct': missingValPct.round(2).astype(str) + ' %'
    })

    print(missingValDf)
    print()

    medicalParameters = ['Age','Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Heart Rate']
    for col in medicalParameters:
        if col in df.columns:
            medianVal = df[col].median()
            df[col] = df[col].fillna(medianVal)
            print(f"{col} column filled with median value: {medianVal}")

    binaryParameters = ['Previous Complications', 'Preexisting Diabetes','Gestational Diabetes', 'Mental Health']
    for col in binaryParameters:
        if col in df.columns:
            df.loc[~df[col].isin([0,1]), col] = np.nan    
            modeSeries = df[col].mode()
            modeVal = modeSeries[0].astype(int) if not modeSeries.empty else 0
            df[col] = df[col].fillna(modeVal)
            print(f"{col} column filled with mode value: {modeVal}")

    beforeLength = len(df)
    df.loc[~df['Risk Level'].isin(['Low','High']), 'Risk Level'] = np.nan
    df.dropna(subset=['Risk Level'], inplace=True)
    print(f"\nNumber of rows eliminated: {beforeLength - len(df)}")  
    print(f"Number of remaining rows: {len(df)}\n")


    #outliers:
    plt.figure(figsize=(14, 8))
    plt.suptitle("Before Outlier Handling", fontsize=16)

    for i, col in enumerate(medicalParameters, 1):
        plt.subplot(2, 4, i)
        sns.boxplot(y=df[col])
        plt.title(col)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("Count of outliers:")
    for col in medicalParameters:
        Q1 =df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lowerBound = Q1-1.5*IQR
        upperBound = Q3+1.5*IQR

        outliers = df[(df[col] < lowerBound) | (df[col] > upperBound)]
        print(f"{col}: {len(outliers)} outliers")
        df[col] = df[col].clip(lowerBound, upperBound)   

    #covert datatype
    df[medicalParameters] = df[medicalParameters].astype(float)
    df[binaryParameters] = df[binaryParameters].astype(int)
    df['Risk Level'] = df['Risk Level'].astype('category')
    print("\nData types after conversion:")
    print(df.dtypes)

    rows, cols = df.shape
    print("Shape of dataset:")
    print(f"Number of rows (samples): {rows}")
    print(f"Number of columns (attributes): {cols}")

    print("Descriptive Statistics")
    print(df.describe())

    return df