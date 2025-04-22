from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import numpy as np

def train_limit_regressor(X, y, model_path="models/xgb_loan_limit.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("💰 MAE:", mean_absolute_error(y_test, preds))
    print("📈 R² Score:", r2_score(y_test, preds))
    joblib.dump(model, model_path)
