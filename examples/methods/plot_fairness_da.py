# %%
import matplotlib.pyplot as plt
import numpy as np
from fairlearn.datasets import fetch_acs_income
from sklearn.ensemble import RandomForestRegressor
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

# take 10% of data
n_samples = int(0.1 * X.shape[0])
X = X[:n_samples]
y = y[:n_samples]
sample_domain = sample_domain[:n_samples]

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

# Evaluate performance
R2_source = clf.score(X_source, y_source)
R2_target = clf.score(X_target, y_target)

# Scatterplot of predicted vs true (no decision boundary in regression)
y_pred_source = clf.predict(X_source)
y_pred_target = clf.predict(X_target)

plt.figure(2, figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_source, y_pred_source, alpha=0.5, label="Source")
plt.plot([y_source.min(), y_source.max()], [y_source.min(), y_source.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"Source (R²={R2_source:.2f})")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_target, y_pred_target, alpha=0.5, label="Target")
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"Target (R²={R2_target:.2f})")
plt.legend()

plt.tight_layout()
plt.show()


# %%
# ----------------------------------
# Build OTDA pipeline for regression
clf_otda = LinearOTMapping(RandomForestRegressor(n_estimators=5, random_state=31415))

# modify y such that for the target domain there are only nan
y = np.where(sample_domain == 1, y, np.nan)
clf_otda.fit(X, y, sample_domain=sample_domain)

# Evaluate R² scores
R2_source = clf_otda.score(X_source, y_source)
R2_target = clf_otda.score(X_target, y_target)

# Predict
y_pred_source = clf_otda.predict(X_source)
y_pred_target = clf_otda.predict(X_target)

# Plot predictions
plt.figure(3, figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_source, y_pred_source, alpha=0.5, label="Source")
plt.plot([y_source.min(), y_source.max()], [y_source.min(), y_source.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"OTDA Source (R²={R2_source:.2f})")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_target, y_pred_target, alpha=0.5, label="Target")
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], "k--")
plt.xlabel("True income")
plt.ylabel("Predicted income")
plt.title(f"OTDA Target (R²={R2_target:.2f})")
plt.legend()

plt.tight_layout()
plt.show()

# %%
