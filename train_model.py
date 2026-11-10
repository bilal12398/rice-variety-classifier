"""
Rice Variety Classifier  -  Training & Evaluation Pipeline
CT-361 Artificial Intelligence & Expert System (AIES)

Classifies rice grains as Cammeo or Osmancik from 7 morphological features
(UCI "Rice (Cammeo and Osmancik)" dataset) using Decision Tree and KNN.

Steps:
  1. Load data and inspect quality (missing values, duplicates, outliers)
  2. Pre-process (label encoding, stratified split, imputation, scaling)
  3. Baseline models  (Majority-class, default Decision Tree, default KNN)
  4. Improved models  (GridSearchCV hyper-parameter tuning)
  5. Evaluation       (Accuracy, Precision, Recall, F1, ROC-AUC, CV, confusion matrix)
  6. Save models, metrics and figures for the interface and the report

Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "Rice_Cammeo_Osmancik.csv"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "outputs"
FIG_DIR = OUT_DIR / "figures"
for d in (MODEL_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]
TARGET = "Class"
CLASS_NAMES = ["Cammeo", "Osmancik"]  # 0 = Cammeo, 1 = Osmancik
PALETTE = {"Cammeo": "#d9822b", "Osmancik": "#2b7bd9"}

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# 1. Load + inspect
# ----------------------------------------------------------------------------
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")
print(df.head())

quality = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "class_counts": df[TARGET].value_counts().to_dict(),
}
# Outliers by the 1.5 x IQR rule (reported; kept because they are real grains)
outliers = {}
for col in FEATURES:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
quality["iqr_outliers_per_feature"] = outliers
print("Data quality:", json.dumps(quality, indent=2))

# ----------------------------------------------------------------------------
# 2. Exploratory plots
# ----------------------------------------------------------------------------
plt.figure(figsize=(5.5, 4))
ax = sns.countplot(data=df, x=TARGET, order=CLASS_NAMES, palette=PALETTE, hue=TARGET, legend=False)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.title("Class Distribution (3810 grains)")
plt.ylabel("Number of grains")
save_fig("01_class_distribution.png")

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.histplot(data=df, x=col, hue=TARGET, palette=PALETTE, kde=True, bins=30, ax=ax,
                 element="step", stat="density", common_norm=False)
    ax.set_title(col)
    ax.set_ylabel("")
axes.ravel()[-1].axis("off")
plt.suptitle("Feature distributions by rice variety", fontsize=14, y=1.01)
save_fig("02_feature_distributions.png")

plt.figure(figsize=(7.5, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Feature correlation matrix")
save_fig("03_correlation_heatmap.png")

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.boxplot(data=df, x=TARGET, y=col, order=CLASS_NAMES, palette=PALETTE, hue=TARGET,
                legend=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
axes.ravel()[-1].axis("off")
plt.suptitle("Box plots (outlier check) by rice variety", fontsize=14, y=1.01)
save_fig("04_boxplots.png")

# ----------------------------------------------------------------------------
# 3. Pre-processing
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. PRE-PROCESSING")
print("=" * 70)
df = df.drop_duplicates().reset_index(drop=True)          # duplicates (none expected)
X = df[FEATURES].astype(float)
y = (df[TARGET] == "Osmancik").astype(int)                # Cammeo=0, Osmancik=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print("Train class balance:", y_train.value_counts(normalize=True).round(3).to_dict())

# The imputer is part of every pipeline (no missing values here, but the
# system stays robust to missing data). Scaling is needed only for KNN.
def tree_pipe(**kw):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", DecisionTreeClassifier(random_state=SEED, **kw))])


def knn_pipe(**kw):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier(**kw))])


# ----------------------------------------------------------------------------
# 4. Baseline models
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. BASELINE MODELS")
print("=" * 70)
baselines = {
    "Majority Class (Dummy)": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
    "Decision Tree (Baseline)": tree_pipe(),
    "KNN (Baseline)": knn_pipe(n_neighbors=5),
}
for m in baselines.values():
    m.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 5. Hyper-parameter tuning (improvement)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. HYPER-PARAMETER TUNING (GridSearchCV, 5-fold stratified)")
print("=" * 70)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

dt_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5, 10],
    "clf__class_weight": [None, "balanced"],
}
knn_grid = {
    "clf__n_neighbors": list(range(1, 32, 2)),
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"],
}

dt_search = GridSearchCV(tree_pipe(), dt_grid, cv=cv, scoring="accuracy", n_jobs=-1)
dt_search.fit(X_train, y_train)
print("Best Decision Tree params:", dt_search.best_params_, f"CV acc = {dt_search.best_score_:.4f}")

knn_search = GridSearchCV(knn_pipe(), knn_grid, cv=cv, scoring="accuracy", n_jobs=-1)
knn_search.fit(X_train, y_train)
print("Best KNN params:", knn_search.best_params_, f"CV acc = {knn_search.best_score_:.4f}")

tuned = {
    "Decision Tree (Tuned)": dt_search.best_estimator_,
    "KNN (Tuned)": knn_search.best_estimator_,
}
all_models = {**baselines, **tuned}

# ----------------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)
rows, cms, rocs = [], {}, {}
for name, model in all_models.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    })
    cms[name] = confusion_matrix(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, proba)
    rocs[name] = (fpr, tpr, roc_auc_score(y_test, proba))

results = pd.DataFrame(rows).set_index("Model")
print(results.round(4).to_string())
results.round(4).to_csv(OUT_DIR / "results_table.csv")

# ----------------------------------------------------------------------------
# 7. Result figures
# ----------------------------------------------------------------------------
# 7a. model comparison bar chart
plot_df = results[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].reset_index().melt(
    id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(11, 5))
ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="viridis")
plt.ylim(0.2, 1.02)
plt.title("Model comparison on the test set")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
save_fig("05_model_comparison.png")

# 7b. confusion matrices (tuned + baselines)
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, name in zip(axes, ["Decision Tree (Baseline)", "Decision Tree (Tuned)",
                           "KNN (Baseline)", "KNN (Tuned)"]):
    sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
save_fig("06_confusion_matrices.png")

# 7c. ROC curves
plt.figure(figsize=(6.5, 5.5))
for name, (fpr, tpr, auc) in rocs.items():
    if "Dummy" in name:
        continue
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
save_fig("07_roc_curves.png")

# 7d. Decision-tree feature importance
best_tree = tuned["Decision Tree (Tuned)"].named_steps["clf"]
importance = pd.Series(best_tree.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(7, 4.5))
importance.plot(kind="barh", color="#2b7bd9")
plt.title("Decision Tree feature importance (tuned)")
plt.xlabel("Importance")
save_fig("08_feature_importance.png")

# 7e. KNN accuracy vs k (cross-validated, with best settings otherwise fixed)
ks = list(range(1, 32, 2))
bp = knn_search.best_params_
k_scores = []
for k in ks:
    m = knn_pipe(n_neighbors=k, weights=bp["clf__weights"], metric=bp["clf__metric"])
    k_scores.append(cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy").mean())
plt.figure(figsize=(7, 4.5))
plt.plot(ks, k_scores, marker="o", color="#d9822b")
plt.axvline(bp["clf__n_neighbors"], color="grey", ls="--", label=f"best k = {bp['clf__n_neighbors']}")
plt.xlabel("k (number of neighbours)")
plt.ylabel("5-fold CV accuracy")
plt.title("KNN: accuracy vs k")
plt.legend()
save_fig("09_knn_accuracy_vs_k.png")

# 7f. Decision-tree accuracy vs max_depth (train vs CV)
depths = list(range(1, 16))
tr_acc, cv_acc = [], []
for d in depths:
    m = tree_pipe(max_depth=d, criterion=dt_search.best_params_["clf__criterion"])
    m.fit(X_train, y_train)
    tr_acc.append(m.score(X_train, y_train))
    cv_acc.append(cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy").mean())
plt.figure(figsize=(7, 4.5))
plt.plot(depths, tr_acc, marker="o", label="Training accuracy")
plt.plot(depths, cv_acc, marker="s", label="5-fold CV accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: effect of depth (over-fitting check)")
plt.legend()
save_fig("10_tree_depth_curve.png")

# 7g. Top of the tuned tree (first 3 levels) for transparency
plt.figure(figsize=(16, 7))
plot_tree(best_tree, max_depth=3, feature_names=FEATURES, class_names=CLASS_NAMES,
          filled=True, rounded=True, fontsize=8, proportion=False)
plt.title("Tuned Decision Tree - first 3 levels")
save_fig("11_decision_tree_top.png")

# ----------------------------------------------------------------------------
# 8. Save artefacts
# ----------------------------------------------------------------------------
summary = {
    "quality": quality,
    "split": {"train": int(len(X_train)), "test": int(len(X_test)), "seed": SEED, "cv_folds": CV_FOLDS},
    "best_params": {
        "decision_tree": {k.replace("clf__", ""): v for k, v in dt_search.best_params_.items()},
        "knn": {k.replace("clf__", ""): v for k, v in knn_search.best_params_.items()},
    },
    "grid_sizes": {
        "decision_tree": int(np.prod([len(v) for v in dt_grid.values()])),
        "knn": int(np.prod([len(v) for v in knn_grid.values()])),
    },
    "grid_cv_accuracy": {"decision_tree": float(dt_search.best_score_), "knn": float(knn_search.best_score_)},
    "results": results.round(4).reset_index().to_dict(orient="records"),
    "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
    "feature_importance": importance.sort_values(ascending=False).round(4).to_dict(),
    "feature_stats": {
        c: {"min": float(df[c].min()), "max": float(df[c].max()),
            "median_cammeo": float(df.loc[df[TARGET] == "Cammeo", c].median()),
            "median_osmancik": float(df.loc[df[TARGET] == "Osmancik", c].median())}
        for c in FEATURES
    },
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

joblib.dump({
    "decision_tree": tuned["Decision Tree (Tuned)"],
    "knn": tuned["KNN (Tuned)"],
    "features": FEATURES,
    "class_names": CLASS_NAMES,
}, MODEL_DIR / "rice_models.joblib")

print("\nSaved: models/rice_models.joblib, outputs/metrics.json, outputs/results_table.csv, outputs/figures/*.png")
print("DONE.")
