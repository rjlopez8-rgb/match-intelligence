import numpy as np

from matchstats.models import default_models
from matchstats.models.features import SHOT_FEATURES, shot_features
from matchstats.models.logistic import fit_logistic, fit_softmax, predict_logistic, predict_softmax
from matchstats.pitch import attacking_frame, in_box, threat
from matchstats.training import auc, brier


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(size=(20000, 2))])
    w = np.array([-0.5, 1.5, -2.0])
    y = (rng.random(20000) < predict_logistic(w, X)).astype(float)
    assert np.allclose(fit_logistic(X, y, l2=0.0), w, atol=0.1)


def test_softmax_outputs_distributions():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(3000), rng.normal(size=(3000, 2))])
    y = rng.integers(0, 3, 3000)
    P = predict_softmax(fit_softmax(X, y, 3, iters=200), X)
    assert np.allclose(P.sum(axis=1), 1)


def test_xg_behaves_like_football():
    m = default_models()
    base = dict(shot_kind="open_play", body_part="foot", under_pressure=False, defenders_in_lane=0)
    close, far = m.shot_xg(dict(base), 99.0, 34.0), m.shot_xg(dict(base), 80.0, 34.0)
    header = m.shot_xg(dict(base, body_part="head"), 99.0, 34.0)
    pen = m.shot_xg(dict(base, shot_kind="penalty"), 94.0, 34.0)
    assert close > far and close > header and pen > 0.6
    assert len(shot_features(base, 99.0, 34.0)) == len(SHOT_FEATURES)


def test_pitch_helpers():
    assert attacking_frame(10.0, 20.0, True) == (10.0, 20.0)
    assert attacking_frame(10.0, 20.0, False) == (95.0, 48.0)
    assert in_box(100.0, 34.0) and not in_box(60.0, 34.0)
    assert threat(100.0, 34.0) > threat(50.0, 34.0)


def test_evaluation_metrics():
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert brier(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
