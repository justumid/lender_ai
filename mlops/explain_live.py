import shap
import numpy as np
from xgboost import XGBClassifier
import joblib

class LiveExplainer:
    def __init__(self, model_path, feature_names):
        self.model = joblib.load(model_path)
        self.feature_names = feature_names
        self.explainer = shap.Explainer(self.model, feature_names=feature_names)

    def explain_instance(self, instance: np.ndarray, top_n=5) -> dict:
        if instance.shape[0] != 1:
            raise ValueError("Instance must be shaped (1, n_features)")
        shap_values = self.explainer(instance)
        importance = dict(zip(self.feature_names, shap_values.values[0]))
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        return {k: round(v, 4) for k, v in sorted_items[:top_n]}
