import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pickle

FEATURE_COLS = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes",
    "Gestational Diabetes", "Mental Health", "Heart Rate"
]

def randomForest(df):

    # encode target variable
    le = LabelEncoder()
    y  = le.fit_transform(df["Risk Level"])

    # split data
    X = df[FEATURE_COLS].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("----------------------------MODEL STATISTICS-----------------------------")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':22} {'Pred High':>10} {'Pred Low':>10}")
    print(f"{'Actual High':22} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'Actual Low':22} {cm[1][0]:>10} {cm[1][1]:>10}")

    # performance metrics
    acc      = accuracy_score(y_test, y_pred)
    prec_cls = precision_score(y_test, y_pred, average=None)
    rec_cls  = recall_score(y_test, y_pred, average=None)
    f1_cls   = f1_score(y_test, y_pred, average=None)

    print("\nPerformance Metrics:")
    print(f"{'Metric':<20} {'High':>10} {'Low':>10}")
    print(f"{'-'*52}")
    print(f"{'Accuracy':<20} {acc*100:>9.2f}%")
    print(f"{'Precision':<20} {prec_cls[0]*100:>9.2f}% {prec_cls[1]*100:>9.2f}%")
    print(f"{'Recall':<20} {rec_cls[0]*100:>9.2f}% {rec_cls[1]*100:>9.2f}%")
    print(f"{'F1 Score':<20} {f1_cls[0]*100:>9.2f}% {f1_cls[1]*100:>9.2f}%")

    # classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # feature importances
    print("Feature Importances:")
    imp_sorted = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in imp_sorted:
        print(f"  {feat:<26} {imp:.4f}  ({imp*100:.2f}%)")

    # save model
    with open("random_forest_model.pkl", "wb") as f: pickle.dump(model, f)
    with open("label_encoder_rf.pkl",    "wb") as f: pickle.dump(le,    f)


def predictRisk(sample: dict):

    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder_rf.pkl", "rb") as f:
        le = pickle.load(f)

    row  = np.array([[sample[f] for f in FEATURE_COLS]])
    prob = model.predict_proba(row)[0]
    pred = le.inverse_transform([np.argmax(prob)])[0]

    result = {
        "label":     pred,
        "prob_high": round(prob[0], 4),
        "prob_low":  round(prob[1], 4)
    }

    return result