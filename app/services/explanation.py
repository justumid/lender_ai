# app/services/explanation.py

import shap
import numpy as np
from typing import Dict, List


class ShapExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.Explainer(self._predict, algorithm="permutation")

    def _predict(self, x: np.ndarray):
        import torch
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            return self.model(tensor).detach().numpy()

    def explain(self, x: np.ndarray, top_n: int = 5) -> Dict[str, float]:
        try:
            shap_values = self.explainer(x)
            scores = dict(zip(self.feature_names, shap_values.values[0]))
            ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            return {k: round(v, 4) for k, v in ranked}
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
            return {k: 0.0 for k in self.feature_names[:top_n]}

