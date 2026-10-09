"""Model training, evaluation, prediction and explainability helpers."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_sample_weight

from .config import (CLASSES, CLASS_TO_INT, EARLY_EXCLUDE_RAW, ENGINEERED, GROUP, LABEL,
                     LEAKY_ENGINEERED, METRICS_PATH, MODEL_PATH, MODES, RAW_NUMERIC, RULE, TARGET)
from .features import engineer

warnings.filterwarnings("ignore")


# =============================================================== predictor ===
@dataclass
class Predictor:
    """sklearn pipeline + feature selection; accepts RAW dataframes only."""
    name: str
    estimator: object
    feature_set: str = "full"            # "full" | "early"
    num_cols: list = field(default_factory=list)
    pipe: Pipeline | None = None
    needs_weights: bool = False

    def __post_init__(self):
        if not self.num_cols:
            cols = list(RAW_NUMERIC) + list(ENGINEERED)
            if self.feature_set == "early":
                cols = [c for c in cols if c not in EARLY_EXCLUDE_RAW and c not in LEAKY_ENGINEERED]
            self.num_cols = cols
        pre = ColumnTransformer([
            ("num", StandardScaler(), self.num_cols),
            ("cat", OneHotEncoder(categories=[MODES], handle_unknown="ignore"), ["Operation_Mode"]),
        ])
        self.pipe = Pipeline([("prep", pre), ("clf", self.estimator)])

    def fit(self, raw: pd.DataFrame, y: np.ndarray):
        X = engineer(raw)
        kw = {"clf__sample_weight": compute_sample_weight("balanced", y)} if self.needs_weights else {}
        self.pipe.fit(X, y, **kw)
        return self

    def proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(engineer(raw))

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return np.array(CLASSES)[self.proba(raw).argmax(1)]


def efficiency_index(p: np.ndarray) -> np.ndarray:
    """0-100 score: Low=0, Medium=50, High=100 weighted by class probability."""
    p = np.asarray(p)
    return 100 * (0.5 * p[..., 1] + p[..., 2])


def _candidates() -> list[Predictor]:
    c = [
        Predictor("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        Predictor("Random Forest", RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=8,
                                                          class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
    ]
    try:
        from xgboost import XGBClassifier
        c.append(Predictor("XGBoost", XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                                    n_jobs=-1, random_state=42, eval_metric="mlogloss"), needs_weights=True))
    except Exception:
        c.append(Predictor("Gradient Boosting", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=42),
                           needs_weights=True))
    return c


INTERPRETABILITY = {"Logistic Regression": "High", "Random Forest": "Medium", "XGBoost": "Low-Medium", "Gradient Boosting": "Low-Medium"}


# ============================================================== importance ===
def raw_permutation_importance(model: Predictor, raw: pd.DataFrame, y: np.ndarray, repeats: int = 3,
                               seed: int = 0, sample: int = 20000) -> pd.DataFrame:
    """Permute each RAW column (engineered columns are rebuilt) and measure the macro-F1 drop."""
    rng = np.random.default_rng(seed)
    if len(raw) > sample:
        idx = rng.choice(len(raw), sample, replace=False)
        raw, y = raw.iloc[idx].reset_index(drop=True), y[idx]
    base = f1_score(y, model.proba(raw).argmax(1), average="macro")
    cols = [c for c in RAW_NUMERIC if (model.feature_set == "full" or c not in EARLY_EXCLUDE_RAW)] + ["Operation_Mode"]
    rows = []
    for c in cols:
        drops = []
        for _ in range(repeats):
            tmp = raw.copy()
            tmp[c] = rng.permutation(tmp[c].values)
            drops.append(base - f1_score(y, model.proba(tmp).argmax(1), average="macro"))
        rows.append((c, LABEL.get(c, "Operation mode"), GROUP.get(c, "Mode"), float(np.mean(drops))))
    out = pd.DataFrame(rows, columns=["feature", "label", "group", "importance"])
    out["importance"] = out["importance"].clip(lower=0)
    tot = out["importance"].sum()
    out["share"] = out["importance"] / tot if tot > 0 else 0.0
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_local(model: Predictor, row: pd.DataFrame, background: pd.DataFrame, target: str | None = None,
                  n_bg: int = 150, seed: int = 0) -> tuple[str, pd.DataFrame]:
    """Occlusion attribution: how much does each metric push the probability of *target* up or down?"""
    p0 = model.proba(row)[0]
    target = target or CLASSES[int(p0.argmax())]
    t = CLASS_TO_INT[target]
    bg = background.sample(min(n_bg, len(background)), random_state=seed)
    cols = [c for c in RAW_NUMERIC if (model.feature_set == "full" or c not in EARLY_EXCLUDE_RAW)] + ["Operation_Mode"]
    out = []
    for c in cols:
        rep = pd.concat([row] * len(bg), ignore_index=True)
        rep[c] = bg[c].values
        out.append((c, LABEL.get(c, "Operation mode"), float(p0[t] - model.proba(rep)[:, t].mean()),
                    row[c].iloc[0]))
    df = pd.DataFrame(out, columns=["feature", "label", "contribution", "value"])
    return target, df.sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)


def attribute_shift(model: Predictor, before: pd.DataFrame, after: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Which metric moved the average efficiency index between two windows? (counterfactual swap)"""
    rng = np.random.default_rng(seed)
    idx_after = efficiency_index(model.proba(after)).mean()
    cols = [c for c in RAW_NUMERIC if (model.feature_set == "full" or c not in EARLY_EXCLUDE_RAW)] + ["Operation_Mode"]
    rows = []
    for c in cols:
        tmp = after.copy()
        tmp[c] = rng.choice(before[c].values, len(tmp))
        rows.append((c, LABEL.get(c, "Operation mode"), idx_after - efficiency_index(model.proba(tmp)).mean()))
    return pd.DataFrame(rows, columns=["feature", "label", "impact"]).sort_values("impact", key=abs, ascending=False)


# ================================================================= training ===
def _prep(df: pd.DataFrame):
    y = df[TARGET].map(CLASS_TO_INT).values
    return df.reset_index(drop=True), y


def train_all(df: pd.DataFrame, progress=lambda msg, frac: None, save: bool = True) -> dict:
    """Train/compare candidate models chronologically, pick the best, and persist everything."""
    df = df.sort_values("Datetime", kind="stable")
    df, y = _prep(df)
    split = int(len(df) * 0.8)                                       # chronological hold-out
    tr, te, ytr, yte = df.iloc[:split], df.iloc[split:], y[:split], y[split:]
    tscv = TimeSeriesSplit(n_splits=3)

    results, fitted = [], {}
    cands = _candidates()
    for i, m in enumerate(cands):
        progress(f"Training {m.name}", 0.05 + 0.5 * i / len(cands))
        t0 = time.time()
        cv = []
        for a, b in tscv.split(tr):
            tmp = Predictor(m.name, type(m.estimator)(**m.estimator.get_params()), needs_weights=m.needs_weights)
            tmp.fit(tr.iloc[a], ytr[a])
            cv.append(f1_score(ytr[b], tmp.proba(tr.iloc[b]).argmax(1), average="macro"))
        m.fit(tr, ytr)
        P = m.proba(te)
        pred = P.argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2], zero_division=0)
        try:
            auc = float(roc_auc_score(yte, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        except Exception:
            auc = float("nan")
        results.append(dict(
            model=m.name, accuracy=float(accuracy_score(yte, pred)), macro_f1=float(f1_score(yte, pred, average="macro")),
            weighted_f1=float(f1_score(yte, pred, average="weighted")), roc_auc=auc,
            cv_mean=float(np.mean(cv)), cv_std=float(np.std(cv)), train_seconds=round(time.time() - t0, 1),
            interpretability=INTERPRETABILITY.get(m.name, "Medium"),
            per_class={c: dict(precision=float(pr[k]), recall=float(rc[k]), f1=float(f1[k]), support=int((yte == k).sum()))
                       for k, c in enumerate(CLASSES)},
            confusion=confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist()))
        fitted[m.name] = m

    # best = highest macro-F1 minus instability penalty; prefer the more interpretable model on ties
    rank = {"High": 0, "Medium": 1, "Low-Medium": 2}
    for r in results:
        r["score"] = r["macro_f1"] - r["cv_std"]
    best = sorted(results, key=lambda r: (-round(r["score"], 3), rank.get(r["interpretability"], 3)))[0]["model"]
    primary = fitted[best]
    primary.name = best

    progress("Fitting early-warning model (sensor + network only)", 0.65)
    early = Predictor("Early-warning Random Forest", RandomForestClassifier(
        n_estimators=60, max_depth=8, min_samples_leaf=60, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        feature_set="early").fit(tr, ytr)
    Pe = early.proba(te)
    majority = float((yte == np.bincount(ytr).argmax()).mean())
    early_res = dict(accuracy=float(accuracy_score(yte, Pe.argmax(1))), macro_f1=float(f1_score(yte, Pe.argmax(1), average="macro")),
                     roc_auc=float(roc_auc_score(yte, Pe, multi_class="ovr", average="macro", labels=[0, 1, 2])),
                     majority_baseline=majority)

    progress("Computing feature importance", 0.78)
    imp = raw_permutation_importance(primary, te, yte)
    imp_early = raw_permutation_importance(early, te, yte)

    progress("Extracting plain-language rules", 0.9)
    X = engineer(tr)[["Error_Rate_%", "Production_Speed_units_per_hr"]]
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=0).fit(X, ytr)
    rules = export_text(tree, feature_names=["Error_Rate_%", "Production_Speed_units_per_hr"])
    e, s = df["Error_Rate_%"], df["Production_Speed_units_per_hr"]
    rule_pred = np.where((e > RULE["error_low"]) | (s <= RULE["speed_low"]), "Low",
                         np.where((s > RULE["speed_high"]) & (e <= RULE["error_high"]), "High", "Medium"))
    agreement = float((rule_pred == df[TARGET].values).mean())

    metrics = dict(
        trained_rows=int(len(df)), train_rows=int(len(tr)), test_rows=int(len(te)), primary=best,
        class_distribution=df[TARGET].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        models=results, early=early_res, importance=imp.to_dict("records"), importance_early=imp_early.to_dict("records"),
        surrogate_rules=rules, rule_agreement=agreement, trained_at=pd.Timestamp.now().isoformat(timespec="seconds"))
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"primary": primary, "early": early, "all": fitted}, MODEL_PATH, compress=3)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    progress("Done", 1.0)
    return dict(primary=primary, early=early, metrics=metrics)


def load_artifacts() -> dict | None:
    """Load persisted models; return None if missing or incompatible with the installed scikit-learn."""
    try:
        bundle = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text())
        bundle["primary"].proba(pd.DataFrame({**{c: [1.0] for c in RAW_NUMERIC}, "Operation_Mode": ["Active"]}))
        return dict(primary=bundle["primary"], early=bundle["early"], metrics=metrics)
    except Exception:
        return None
