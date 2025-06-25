# %%
import matplotlib.pyplot as plt
import numpy as np
from fairlearn.datasets import fetch_acs_income
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skada import LinearOTMapping, source_target_split

# %%
# Load the dataset
df = fetch_acs_income(as_frame=True)
X_df = df["frame"]
y = df["target"].to_numpy()
sample_domain = X_df["SEX"].to_numpy()  # 1: male, 2: female

# Drop target and sensitive attribute to get feature matrix
X = X_df.drop(columns=["PINCP", "SEX"]).to_numpy()

# Take 10% of data while preserving the distribution
X, _, y, _, sample_domain, _ = train_test_split(
    X, y, sample_domain, test_size=0.9, stratify=sample_domain, random_state=42
)

# Normalize features
X = StandardScaler().fit_transform(X)

# Re-label domains: source=1 (e.g. male), target=2 (e.g. female)
sample_domain = np.where(sample_domain == 1, 1, -1)


# %%

X_source, X_target, y_source, y_target = source_target_split(
    X, y, sample_domain=sample_domain
)
print(f"Source domain size: {X_source.shape[0]}")
print(f"Target domain size: {X_target.shape[0]}")
print(f"Source domain income mean: {y_source.mean()}")
print(f"Target domain income mean: {y_target.mean()}")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(y_source, bins=50, alpha=0.5, label="Male (source)")
plt.hist(y_target, bins=50, alpha=0.5, label="Female (target)")
plt.legend()
plt.title("Income distribution before adaptation")
plt.xlabel("PINCP")

# %%

# Train regressor on source

clf = RandomForestRegressor(n_estimators=5, random_state=31415)
clf.fit(X_source, y_source)
# Evaluate performance using MAE
mae_source = mean_absolute_error(y_source, clf.predict(X_source)) / y_source.mean()
mae_target = mean_absolute_error(y_target, clf.predict(X_target)) / y_target.mean()

# Scatterplot of predicted vs true (no decision boundary in regression)
y_pred_source = clf.predict(X_source)
y_pred_target = clf.predict(X_target)

plt.figure(2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_source, y_pred_source, alpha=0.5, label="Source")
plt.plot([y_source.min(), y_source.max()], [y_source.min(), y_source.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"Source (MAE={mae_source:.2f})")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_target, y_pred_target, alpha=0.5, label="Target")
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"Target (MAE={mae_target:.2f})")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# ----------------------------------
# Build OTDA pipeline for regression
clf_otda = LinearOTMapping(RandomForestRegressor(n_estimators=5, random_state=31415))

# modify y such that for the target domain there are only nan
y_for_fit = np.where(sample_domain == 1, y, np.nan)
clf_otda.fit(X, y_for_fit, sample_domain=sample_domain)


# Evaluate Mean Absolute Error (MAE) scores
mae_source = mean_absolute_error(y_source, clf_otda.predict(X_source)) / y_source.mean()
mae_target = mean_absolute_error(y_target, clf_otda.predict(X_target)) / y_target.mean()

print(f"Mean Absolute Error (MAE) - Source: {mae_source:.2f}")
print(f"Mean Absolute Error (MAE) - Target: {mae_target:.2f}")

# Predict
y_pred_source_ot = clf_otda.predict(X_source)
y_pred_target_ot = clf_otda.predict(X_target)

# Plot predictions
plt.figure(3, figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_source, y_pred_source_ot, alpha=0.5, label="Source")
plt.plot([y_source.min(), y_source.max()], [y_source.min(), y_source.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"OTDA Source (MAE={mae_source:.2f})")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_target, y_pred_target_ot, alpha=0.5, label="Target")
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"OTDA Target (MAE={mae_target:.2f})")
plt.legend()

plt.tight_layout()
plt.show()

# %%


def compute_demographic_parity_difference(y_pred, sensitive_attr):
    """Compute the demographic parity difference between two groups."""
    group_1 = y_pred[sensitive_attr == 1]
    group_2 = y_pred[sensitive_attr == -1]

    p1 = np.mean(group_1)
    p2 = np.mean(group_2)

    # Scale the difference by the overall mean prediction
    return np.abs(p1 - p2)


def compute_error_difference(y_pred, y_true, sensitive_attr):
    group_1 = y_pred[sensitive_attr == 1]
    group_2 = y_pred[sensitive_attr == -1]

    group_1_true = y_true[sensitive_attr == 1]
    group_2_true = y_true[sensitive_attr == -1]

    error_1 = np.linalg.norm(group_1 - group_1_true)
    error_2 = np.linalg.norm(group_2 - group_2_true)

    return np.abs(error_1 - error_2)


# %%
# Concatenate source predictions (before OTDA) with target predictions (before OTDA)
y_stacked_no_ot = np.concatenate([y_pred_source, y_pred_target])

# Concatenate source predictions (before OTDA) with target predictions (after OTDA)
y_stacked_ot = np.concatenate([y_pred_source, y_pred_target_ot])

# %%

print(
    "Demographic parity difference before OTDA:",
    compute_demographic_parity_difference(
        y_stacked_no_ot, sensitive_attr=sample_domain
    ),
)
print(
    "Demographic parity difference after OTDA:",
    compute_demographic_parity_difference(y_stacked_ot, sensitive_attr=sample_domain),
)

# compute percrentage improvement in demographic parity difference
improvement_dp = (
    compute_demographic_parity_difference(y_stacked_no_ot, sensitive_attr=sample_domain)
    - compute_demographic_parity_difference(y_stacked_ot, sensitive_attr=sample_domain)
) / compute_demographic_parity_difference(y_stacked_no_ot, sensitive_attr=sample_domain)

print(f"Percentage improvement in demographic parity difference: {improvement_dp:.2%}")
# %%

print(
    "Error difference before OTDA:",
    compute_error_difference(y_stacked_no_ot, y, sensitive_attr=sample_domain),
)
print(
    "Error difference after OTDA:",
    compute_error_difference(y_stacked_ot, y, sensitive_attr=sample_domain),
)
# %%
