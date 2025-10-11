import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

scaled_df = pd.read_csv("scaled_data.csv")
if "fail" not in scaled_df.columns:
    raise KeyError("'fail' column missing from scaled_data.csv. Rerun cleaning.py")

X = scaled_df[["CS", "RP", "Temperature", "IP", "USS"]]
Y = scaled_df["fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [8, 10, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best hyperparams:", grid_search.best_params_)

predictions = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(best_model, "random_forest_failure_model.pkl")

results = X_test.copy()
results["actual_fail"] = y_test
results["predicted_fail"] = predictions
results.to_csv("supervised_predictions.csv", index=False)

