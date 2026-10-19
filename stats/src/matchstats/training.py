"""Train and evaluate the stats layer's models on simulated matches.

Inputs are raw event files produced by ``matchengine`` (one ``*.jsonl.gz`` per
match, first line ``{"meta": {...}}``). Matches are split by match into train
(80%) and test (20%) sets, so evaluation is always on unseen matches.

Models only ever see observable feed fields. The simulator's ground truth
(``details.truth``) is used exclusively to *evaluate* them, which is what lets us
publish honest accuracy numbers.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics.control import CHAOTIC
from .models import WEIGHTS_DIR, LinearModel
from .models.features import (
    PASS_FEATURES,
    SHOT_FEATURES,
    WINPROB_FEATURES,
    pass_features,
    shot_features,
    winprob_features,
)
from .models.logistic import fit_logistic, fit_softmax, predict_logistic, predict_softmax
from .pitch import attacking_frame


@dataclass
class Match:
    match_id: str
    home: str
    away: str
    events: list[dict]

    def tf(self, loc: dict, team_id: str, period: int) -> tuple[float, float]:
        attacks_right = (team_id == self.home) == (period == 1)
        return attacking_frame(loc["x"], loc["y"], attacks_right)

    def result(self) -> int:
        end = self.events[-1]["score"]
        return 0 if end["home"] > end["away"] else (1 if end["home"] == end["away"] else 2)


def load_matches(directory: str | Path) -> list[Match]:
    matches = []
    for path in sorted(Path(directory).glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            meta = json.loads(fh.readline())["meta"]
            events = [json.loads(line) for line in fh if line.strip()]
        if events and events[-1]["type"] == "match_end":
            matches.append(Match(meta["match_id"], meta["home"], meta["away"], events))
    return matches


@dataclass
class Dataset:
    X: list[list[float]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    truth: list[float] = field(default_factory=list)

    def arrays(self):
        return np.array(self.X), np.array(self.y), np.array(self.truth)


def shot_dataset(matches: list[Match]) -> Dataset:
    ds = Dataset()
    for m in matches:
        for e in m.events:
            if e["type"] == "shot":
                x, y = m.tf(e["location"], e["team_id"], e["period"])
                ds.X.append(shot_features(e["details"], x, y))
                ds.y.append(1.0 if e["details"]["outcome"] == "goal" else 0.0)
                ds.truth.append(e["details"]["truth"]["xg"])
    return ds


def pass_dataset(matches: list[Match]) -> Dataset:
    ds = Dataset()
    for m in matches:
        for e in m.events:
            if e["type"] == "pass":
                d = e["details"]
                x, y = m.tf(e["location"], e["team_id"], e["period"])
                ex, ey = m.tf(d["end_location"], e["team_id"], e["period"])
                ds.X.append(pass_features(d, x, y, ex, ey))
                ds.y.append(1.0 if d["outcome"] == "complete" else 0.0)
                ds.truth.append(d["truth"]["p_success_intended"])
    return ds


def match_states(m: Match, xg_coef: np.ndarray) -> list[tuple[float, int, float, int]]:
    """(minute, goal_diff, xg_diff, red_diff) sampled every match minute and after every goal/red card."""
    states = []
    xg = {m.home: 0.0, m.away: 0.0}
    reds = {m.home: 0, m.away: 0}
    next_minute = 0
    gd_before = 0
    for e in m.events:
        minute = (e["period"] - 1) * 45.0 + min(e["t"], 2700.0) / 60.0
        while next_minute <= minute and next_minute <= 95:
            gd = gd_before
            states.append((float(next_minute), gd, xg[m.home] - xg[m.away], reds[m.home] - reds[m.away]))
            next_minute += 1
        if e["type"] == "shot":
            x, y = m.tf(e["location"], e["team_id"], e["period"])
            xg[e["team_id"]] += float(
                predict_logistic(xg_coef, np.array([shot_features(e["details"], x, y)]))[0]
            )
        elif e["type"] == "card" and e["details"].get("card") in ("red", "second_yellow"):
            reds[e["team_id"]] += 1
        if e["type"] in ("goal", "card", "shot"):
            gd = e["score"]["home"] - e["score"]["away"]
            states.append((minute, gd, xg[m.home] - xg[m.away], reds[m.home] - reds[m.away]))
        gd_before = e["score"]["home"] - e["score"]["away"]
    return states


def chaos_rates(matches: list[Match]) -> list[float]:
    rates = []
    for m in matches:
        per_minute: dict[int, int] = {}
        for e in m.events:
            if e["type"] in CHAOTIC:
                minute = int((e["period"] - 1) * 45 + e["t"] / 60)
                per_minute[minute] = per_minute.get(minute, 0) + 1
        for minute in range(5, 90):
            rates.append(sum(per_minute.get(k, 0) for k in range(minute - 4, minute + 1)) / 5.0)
    return rates


# -- metrics ---------------------------------------------------------------------------
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            out.append(
                {
                    "predicted": round(float(p[mask].mean()), 4),
                    "observed": round(float(y[mask].mean()), 4),
                    "n": int(mask.sum()),
                }
            )
    return out


def train_all(
    directory: str | Path, out_dir: Path = WEIGHTS_DIR, seed: int = 7, test_share: float = 0.2
) -> dict:
    matches = load_matches(directory)
    if len(matches) < 10:
        raise SystemExit(f"need at least 10 matches in {directory}, found {len(matches)}")
    rng = random.Random(seed)
    rng.shuffle(matches)
    n_test = max(2, int(len(matches) * test_share))
    test, train = matches[:n_test], matches[n_test:]
    report: dict = {"matches": {"train": len(train), "test": len(test)}}

    # Expected goals
    Xs, ys, ts = shot_dataset(train).arrays()
    w_xg = fit_logistic(Xs, ys, l2=1e-3)
    Xt, yt, tt = shot_dataset(test).arrays()
    pt = predict_logistic(w_xg, Xt)
    report["xg"] = {
        "train_shots": len(ys),
        "test_shots": len(yt),
        "test_goals": int(yt.sum()),
        "test_predicted_goals": round(float(pt.sum()), 1),
        "true_xg_sum": round(float(tt.sum()), 1),
        "mae_vs_true_xg": round(float(np.mean(np.abs(pt - tt))), 4),
        "correlation_vs_true_xg": round(float(np.corrcoef(pt, tt)[0, 1]), 3),
        "brier": round(brier(pt, yt), 4),
        "brier_true_xg": round(brier(tt, yt), 4),
        "auc": round(auc(pt, yt), 3),
        "calibration": calibration_bins(pt, yt, 8),
    }
    LinearModel("xg", SHOT_FEATURES, w_xg, True, {"trained_on_shots": len(ys)}).save(out_dir)

    # Pass completion / difficulty
    Xp, yp, tp = pass_dataset(train).arrays()
    w_pass = fit_logistic(Xp, yp, l2=1e-4)
    Xpt, ypt, tpt = pass_dataset(test).arrays()
    ppt = predict_logistic(w_pass, Xpt)
    modelled = tpt < 0.999  # the engine reports 1.0 for restarts it does not model probabilistically
    report["pass_completion"] = {
        "train_passes": len(yp),
        "test_passes": len(ypt),
        "observed_completion": round(float(ypt.mean()), 4),
        "predicted_completion": round(float(ppt.mean()), 4),
        "passes_with_engine_probability": int(modelled.sum()),
        "mae_vs_true_probability": round(float(np.mean(np.abs(ppt[modelled] - tpt[modelled]))), 4),
        "correlation_vs_true_probability": round(float(np.corrcoef(ppt[modelled], tpt[modelled])[0, 1]), 3),
        "auc": round(auc(ppt, ypt), 3),
        "auc_modelled": round(auc(ppt[modelled], ypt[modelled]), 3),
        "auc_true_probability": round(auc(tpt[modelled], ypt[modelled]), 3),
        "brier": round(brier(ppt, ypt), 4),
        "calibration": calibration_bins(ppt, ypt, 10),
    }
    LinearModel("pass_completion", PASS_FEATURES, w_pass, True, {"trained_on_passes": len(yp)}).save(out_dir)

    # Win probability
    def wp_data(ms):
        X, y = [], []
        for m in ms:
            res = m.result()
            for minute, gd, xgd, red in match_states(m, w_xg):
                X.append(winprob_features(minute, gd, xgd, red))
                y.append(res)
        return np.array(X), np.array(y, dtype=int)

    Xw, yw = wp_data(train)
    W = fit_softmax(Xw, yw, 3, l2=1e-4, iters=4000, lr=0.05)
    Xwt, ywt = wp_data(test)
    Pt = predict_softmax(W, Xwt)
    onehot = np.eye(3)[ywt]
    base = np.bincount(yw, minlength=3) / len(yw)
    by_minute = {}
    minutes = Xwt[:, 1] * 95.0
    for lo, hi, label in (
        (0, 1, "kick-off"),
        (14, 16, "15'"),
        (44, 46, "45'"),
        (59, 61, "60'"),
        (74, 76, "75'"),
        (84, 86, "85'"),
    ):
        mask = (minutes >= lo) & (minutes < hi)
        if mask.sum():
            by_minute[label] = {
                "brier": round(float(np.mean(np.sum((Pt[mask] - onehot[mask]) ** 2, axis=1))), 4),
                "accuracy": round(float(np.mean(Pt[mask].argmax(1) == ywt[mask])), 3),
            }
    report["win_probability"] = {
        "train_states": len(yw),
        "test_states": len(ywt),
        "brier": round(float(np.mean(np.sum((Pt - onehot) ** 2, axis=1))), 4),
        "brier_baseline": round(float(np.mean(np.sum((base - onehot) ** 2, axis=1))), 4),
        "log_loss": round(float(-np.mean(np.log(np.clip(Pt[np.arange(len(ywt)), ywt], 1e-6, 1)))), 4),
        "by_minute": by_minute,
        "result_rates_train": {
            "home": round(float(base[0]), 3),
            "draw": round(float(base[1]), 3),
            "away": round(float(base[2]), 3),
        },
    }
    LinearModel("win_probability", WINPROB_FEATURES, W, True, {"trained_on_states": len(yw)}).save(out_dir)

    # Control-vs-chaos reference levels
    rates = np.array(chaos_rates(train))
    ref = {
        "chaos_rate_p10": round(float(np.quantile(rates, 0.10)), 3),
        "chaos_rate_p90": round(float(np.quantile(rates, 0.90)), 3),
    }
    (out_dir / "control_reference.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
    report["control_reference"] = ref
    report["generated_from"] = str(directory)
    return report


def report_markdown(r: dict, image: str | None = None) -> str:
    xg, pc, wp = r["xg"], r["pass_completion"], r["win_probability"]
    lines = [
        "# Model accuracy report",
        "",
        f"Trained on {r['matches']['train']} simulated matches, evaluated on {r['matches']['test']} **unseen** matches.",
        "Ground truth comes from the match engine's own probabilities, which the models never see.",
        "",
        *([f"![Model accuracy]({image})", ""] if image else []),
        "## Expected goals (xG)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test shots | {xg['test_shots']} |",
        f"| Goals scored vs. our xG total | {xg['test_goals']} vs. {xg['test_predicted_goals']} |",
        f"| Correlation with true chance quality | {xg['correlation_vs_true_xg']} |",
        f"| Mean absolute error vs. true xG | {xg['mae_vs_true_xg']} |",
        f"| Brier score (ours / true xG) | {xg['brier']} / {xg['brier_true_xg']} |",
        f"| AUC (goal vs. no goal) | {xg['auc']} |",
        "",
        "## Pass difficulty (completion model)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test passes | {pc['test_passes']:,} |",
        f"| Observed vs. predicted completion | {pc['observed_completion']:.1%} vs. {pc['predicted_completion']:.1%} |",
        f"| Correlation with true completion probability | {pc['correlation_vs_true_probability']} |",
        f"| Mean absolute error vs. true probability | {pc['mae_vs_true_probability']} |",
        f"| AUC on passes the engine models (ours / engine) | {pc['auc_modelled']} / {pc['auc_true_probability']} |",
        f"| AUC on all passes incl. set pieces | {pc['auc']} |",
        "",
        "## Win probability",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test game states | {wp['test_states']:,} |",
        f"| Brier score (ours / always-predict-league-average) | {wp['brier']} / {wp['brier_baseline']} |",
        f"| Log loss | {wp['log_loss']} |",
    ]
    for label, v in wp["by_minute"].items():
        lines.append(f"| At {label}: Brier / accuracy | {v['brier']} / {v['accuracy']:.0%} |")
    lines += [
        "",
        "Lower Brier and log loss are better. AUC of 0.5 is random and 1.0 is perfect. "
        "The engine's probabilities are its decision-time estimates, made before execution "
        "error and defender reactions, so a model trained on outcomes can outrank them.",
        "",
    ]
    return "\n".join(lines)


def plot_report(r: dict, path: str | Path) -> Path:
    """Three-panel accuracy chart: xG calibration, pass-model calibration, win-probability Brier by minute."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, accent, accent2, grid = "#1b1b1f", "#6b6b76", "#0f6cbd", "#8a5cf6", "#e6e6ea"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": grid,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=160)

    for ax, key, title, colour in (
        (axes[0], "xg", "Expected goals", accent),
        (axes[1], "pass_completion", "Pass completion", accent2),
    ):
        bins = r[key]["calibration"]
        lo = min(min(b["predicted"] for b in bins), min(b["observed"] for b in bins))
        hi = max(max(b["predicted"] for b in bins), max(b["observed"] for b in bins))
        lo, hi = max(0.0, lo - 0.05), min(1.0, hi + 0.05)
        ax.plot([lo, hi], [lo, hi], color=grid, lw=1.5, zorder=1, label="perfect calibration")
        ax.plot(
            [b["predicted"] for b in bins],
            [b["observed"] for b in bins],
            "-o",
            color=colour,
            lw=2,
            ms=5,
            zorder=2,
            label="our model",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed rate (unseen matches)")
        ax.set_title(title, loc="left", color=ink, fontweight="bold")
        ax.grid(color=grid, lw=0.6)
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    xg, pc = r["xg"], r["pass_completion"]
    axes[0].text(
        0.97,
        0.04,
        f"corr. with true xG {xg['correlation_vs_true_xg']:.3f}\nAUC {xg['auc']:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color=muted,
        fontsize=8.5,
    )
    axes[1].text(
        0.97,
        0.04,
        f"AUC {pc['auc']:.2f}\n{pc['test_passes']:,} test passes",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=muted,
        fontsize=8.5,
    )

    ax = axes[2]
    wp = r["win_probability"]
    labels = list(wp["by_minute"])
    vals = [wp["by_minute"][k]["brier"] for k in labels]
    ax.bar(labels, vals, color=accent, width=0.6, zorder=2)
    ax.axhline(wp["brier_baseline"], color=muted, lw=1.2, ls="--", zorder=3)
    ax.set_ylim(0, max([*vals, wp["brier_baseline"]]) * 1.22)
    ax.text(
        len(labels) - 0.6,
        wp["brier_baseline"] * 1.03,
        "guessing league averages",
        va="bottom",
        ha="right",
        color=muted,
        fontsize=8.5,
    )
    ax.set_title("Win probability: error by minute", loc="left", color=ink, fontweight="bold")
    ax.set_ylabel("Brier score (lower is better)")
    ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
    fig.suptitle(
        f"Model accuracy on {r['matches']['test']} unseen simulated matches",
        x=0.01,
        ha="left",
        color=ink,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
