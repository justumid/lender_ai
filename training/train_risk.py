from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import numpy as np

def train_risk_classifier(X, y, model_path="models/xgb_risk.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = XGBClassifier(eval_metric='logloss')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    print("✅ Risk Accuracy:", accuracy_score(y_test, preds))
    print("✅ Risk AUC:", roc_auc_score(y_test, proba))
    joblib.dump(clf, model_path)
