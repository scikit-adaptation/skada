"""
====================================================
Plot dataset source domain and shifted target domain
====================================================
This illustrates the :func:`~skada.datasets.make_variable_frequency_dataset`
dataset generator. Each method consists of generating source data
and shifted target data.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

from skada.datasets import make_variable_frequency_dataset

# Use same random seed for multiple calls to make_datasets to
# ensure same distributions
RANDOM_SEED = np.random.randint(2**10)


X_source, y_source, X_target, y_target = make_variable_frequency_dataset(
    n_samples_source=2,
    n_samples_target=2,
    n_channels=1,
    n_classes=2,
    delta_f=2,
    band_size=1,
    noise=0.2,
    random_state=RANDOM_SEED
)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
plt.subplots_adjust(bottom=0.15)
fig.suptitle('Signal visualisation')
ax1.plot(
    X_source[0, 0, 1000:1100],
    alpha=0.7,
    label='source'
)
ax1.plot(
    X_target[0, 0, 1000:1100],
    alpha=0.7,
    label='target'
)
ax1.set_title("Class 1")
ax1.legend()

ax2.plot(
    X_source[1, 0, 1000:1100],
    alpha=0.7
)
ax2.plot(
    X_target[1, 0, 1000:1100],
    alpha=0.7
)
ax2.set_title("Class 2")

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex="row", sharey="row", figsize=(8, 4))
plt.subplots_adjust(bottom=0.15)
fig.suptitle('PSD shift')
ax1.psd(
    X_source[0, 0],
    Fs=100,
    alpha=0.7,
    label='source'
)
ax1.psd(
    X_target[0, 0],
    Fs=100,
    alpha=0.7,
    label='target'
)
ax1.set_title("Class 1")
ax1.legend()

ax2.psd(
    X_source[1, 0],
    Fs=100,
    alpha=0.7
)
ax2.psd(
    X_target[1, 0],
    Fs=100,
    alpha=0.7
)
ax2.set_title("Class 2")

plt.show()


print("The data was generated from (random_state=%d):" % RANDOM_SEED)

# %%
