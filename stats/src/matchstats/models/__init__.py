"""Trained models: expected goals, pass difficulty and win probability.

Weights live in ``models/weights/*.json`` (written by ``matchstats train``). If a
weights file is missing, a documented fallback is used so the analyzer always
works, and the snapshot reports which models are trained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import (
    PASS_FEATURES,
    SHOT_FEATURES,
    WINPROB_FEATURES,
    pass_features,
    shot_features,
    winprob_features,
)
from .logistic import predict_logistic, predict_softmax

WEIGHTS_DIR = Path(__file__).parent / "weights"


@dataclass
class LinearModel:
    name: str
    features: tuple[str, ...]
    coef: np.ndarray
    trained: bool
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls, name: str, features: tuple[str, ...], fallback: np.ndarray, directory: Path = WEIGHTS_DIR
    ) -> LinearModel:
        path = directory / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if tuple(data["features"]) != features:
                raise ValueError(f"{path} was trained on different features; retrain with `matchstats train`")
            return cls(name, features, np.array(data["coef"]), True, data.get("meta", {}))
        return cls(name, features, fallback, False)

    def save(self, directory: Path = WEIGHTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        coef = self.coef.tolist()
        path.write_text(
            json.dumps(
                {"name": self.name, "features": list(self.features), "coef": coef, "meta": self.meta},
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


# Hand-set fallbacks: sensible shapes, used only until the models are trained.
_XG_FALLBACK = np.array([-0.2, -1.1, 0.0, 1.2, -0.9, 2.6, -0.3, -0.4, -0.35, 0.0])
_PASS_FALLBACK = np.array(
    [3.4, -0.6, 0.0, -0.8, -0.6, -1.0, -0.6, 0.6, 0.5, -0.4, -0.8, -0.1, -0.6, -0.4, 0.0]
)
_WP_FALLBACK = np.zeros((len(WINPROB_FEATURES), 3))
_WP_FALLBACK[0] = [0.25, 0.0, -0.1]
_WP_FALLBACK[3] = [0.6, 0.0, -0.6]
_WP_FALLBACK[4] = [4.0, 0.0, -4.0]
_WP_FALLBACK[6, 1] = -4.0


class Models:
    def __init__(self, directory: Path = WEIGHTS_DIR):
        self.xg = LinearModel.load("xg", SHOT_FEATURES, _XG_FALLBACK, directory)
        self.pass_difficulty = LinearModel.load("pass_completion", PASS_FEATURES, _PASS_FALLBACK, directory)
        self.win_probability = LinearModel.load("win_probability", WINPROB_FEATURES, _WP_FALLBACK, directory)

    def status(self) -> dict:
        return {
            m.name: ("trained" if m.trained else "fallback")
            for m in (self.xg, self.pass_difficulty, self.win_probability)
        }

    def shot_xg(self, details: dict, x_tf: float, y_tf: float) -> float:
        x = np.array([shot_features(details, x_tf, y_tf)])
        return float(predict_logistic(self.xg.coef, x)[0])

    def pass_completion(self, details: dict, x_tf: float, y_tf: float, ex: float, ey: float) -> float:
        x = np.array([pass_features(details, x_tf, y_tf, ex, ey)])
        return float(predict_logistic(self.pass_difficulty.coef, x)[0])

    def win_probabilities(
        self, minute: float, goal_diff: int, xg_diff: float, red_diff: int
    ) -> tuple[float, float, float]:
        x = np.array([winprob_features(minute, goal_diff, xg_diff, red_diff)])
        p = predict_softmax(self.win_probability.coef, x)[0]
        return float(p[0]), float(p[1]), float(p[2])


_DEFAULT: Models | None = None


def default_models() -> Models:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Models()
    return _DEFAULT
