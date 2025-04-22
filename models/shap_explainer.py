import shap
import numpy as np
from typing import List, Dict
from torch import nn


class ShapExplainer:
    def __init__(self, model: nn.Module, background_data: np.ndarray, feature_names: List[str]):
        """
        model: A PyTorch model (must have forward(x) → single value)
        background_data: Numpy array used for background SHAP context (e.g. sample 100 embeddings)
        feature_names: List of original feature names
        """
        self.model = model
        self.feature_names = feature_names

        # Use Kernel SHAP (model-agnostic)
        self.explainer = shap.KernelExplainer(self._predict_wrapper, background_data)

    def _predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs).detach().cpu().numpy()
        return outputs

    def explain(self, instance: np.ndarray, top_n: int = 5) -> Dict[str, float]:
        """
        instance: single input vector (1, D)
        """
        try:
            shap_values = self.explainer.shap_values(instance)
            shap_scores = dict(zip(self.feature_names, shap_values[0]))
            sorted_features = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            return {k: round(v, 4) for k, v in sorted_features}
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
            return {k: 0.0 for k in self.feature_names[:top_n]}
