"""
=====================
SKADA logo generation
=====================


In this example we plot the logos of the SKADA toolbox.

This logo is that it is done 100% in Python and generated using
matplotlib and plotting the solution of the EMD solver from POT.
"""

# Author: Théo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %%
from math import factorial

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

color_S = "black"
color_1 = "#2364aa"
color_0 = "#c84630"
color_unlabel = "#999c81"


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def bezier_curve(t, points):
    n = len(points) - 1
    return sum(comb(n, i) * (1 - t) ** (n - i) * t**i * points[i] for i in range(n + 1))


rng = np.random.RandomState(42)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.2)

# Define the control points for the Bézier curve
points = np.array([[2, 8], [4, 10], [6, 0], [8, 2]])

# Generate the S shape
t_values = np.linspace(-0.3, 1.3, 100)
curve = np.array([bezier_curve(t, points) for t in t_values])

start_length = 10
for i, angle in enumerate(range(0, 50, 10)):
    rotation_center = np.mean(curve, axis=0)
    rotation_angle = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    rotated_curve = np.dot(curve - rotation_center, rotation_matrix) + rotation_center

    # Plot the S shape
    ax.plot(
        rotated_curve[:, 1],
        rotated_curve[:, 0],
        color=color_S,
        linewidth=start_length - 2 * i,
        alpha=1 - i / 8,
    )

# Plot the S shape
ax.plot(curve[:, 1], curve[:, 0], color=color_S, linewidth=2)
# Draw point in the square 0 10 and choose the class of the point depending on the curve
n_dots = 200
max = 2
min = 1
dots = rng.rand(n_dots, 2) * 10
dots_class = []
dots_keep = []
for i, dot in enumerate(dots):
    # get the closest x axis point of the curve of dot
    closest_point = curve[np.argmin(np.abs(curve[:, 0] - dot[0]))]
    if (abs(closest_point[1] - dot[1]) < max) & (abs(closest_point[1] - dot[1]) > min):
        dots_keep.append(dot)
        if closest_point[1] > dot[1]:
            dots_class.append(1)
        else:
            dots_class.append(0)

dots_keep = np.array(dots_keep)
dots_class = np.array(dots_class)
# Plot the dots
# choose the cmap with two colors
s = 150
alpha = 0.9
linewidths = 1

# plot the dots with different color and shape depending of the class
ax.scatter(
    dots_keep[dots_class == 0, 1],
    dots_keep[dots_class == 0, 0],
    color=color_1,
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_keep[dots_class == 1, 1],
    dots_keep[dots_class == 1, 0],
    color=color_0,
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

dots_rotated = np.dot(dots_keep - rotation_center, rotation_matrix) + rotation_center

ax.scatter(
    dots_rotated[dots_class == 0, 1],
    dots_rotated[dots_class == 0, 0],
    color=color_unlabel,
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_rotated[dots_class == 1, 1],
    dots_rotated[dots_class == 1, 0],
    color=color_unlabel,
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

# Hide axes
ax.axis("off")

fontsize = 250
y_axis = 1.5
ax.text(
    10,
    y_axis,
    r"\bf\textsf{K}",
    usetex=True,
    fontsize=fontsize,
    color="black",
)
ax.text(
    16,
    y_axis,
    r"\bf\textsf{A}",
    usetex=True,
    fontsize=fontsize,
    color="black",
)
ax.text(
    22.5,
    y_axis,
    r"\bf\textsf{D}",
    usetex=True,
    fontsize=fontsize,
    color=color_1,
)
ax.text(
    28,
    y_axis,
    r"\bf\textsf{A}",
    usetex=True,
    fontsize=fontsize,
    color=color_0,
)


# Save the figure
plt.savefig(
    "skada_logo_full.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "skada_logo_full.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.2)

# Define the control points for the Bézier curve
points = np.array([[2, 8], [4, 10], [6, 0], [8, 2]])


# Generate the S shape
t_values = np.linspace(-0.3, 1.3, 100)
curve = np.array([bezier_curve(t, points) for t in t_values])

start_length = 10
for i, angle in enumerate(range(0, 50, 10)):
    rotation_center = np.mean(curve, axis=0)
    rotation_angle = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    rotated_curve = np.dot(curve - rotation_center, rotation_matrix) + rotation_center

    # Plot the S shape
    ax.plot(
        rotated_curve[:, 1],
        rotated_curve[:, 0],
        color=color_S,
        linewidth=start_length - 2 * i,
        alpha=1 - i / 8,
    )

# Plot the S shape
ax.plot(curve[:, 1], curve[:, 0], color=color_S, linewidth=start_length)
# Draw point in the square 0 10 and choose the class of the point depending on the curve

# Plot the dots

# plot the dots with different color and shape depending of the class
ax.scatter(
    dots_keep[dots_class == 0, 1],
    dots_keep[dots_class == 0, 0],
    color=color_1,
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_keep[dots_class == 1, 1],
    dots_keep[dots_class == 1, 0],
    color=color_0,
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

dots_rotated = np.dot(dots_keep - rotation_center, rotation_matrix) + rotation_center

ax.scatter(
    dots_rotated[dots_class == 0, 1],
    dots_rotated[dots_class == 0, 0],
    color=color_unlabel,
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_rotated[dots_class == 1, 1],
    dots_rotated[dots_class == 1, 0],
    color=color_unlabel,
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

# Hide axes
ax.axis("off")


# Save the figure
plt.savefig(
    "skada_logo.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "skada_logo.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.2)

# Define the control points for the Bézier curve
points = np.array([[2, 8], [4, 10], [6, 0], [8, 2]])


# Function to calculate a point on a Bézier curve given t (0 <= t <= 1)
def bezier_curve(t, points):
    n = len(points) - 1
    return sum(comb(n, i) * (1 - t) ** (n - i) * t**i * points[i] for i in range(n + 1))


# Generate the S shape
t_values = np.linspace(-0.3, 1.3, 100)
curve = np.array([bezier_curve(t, points) for t in t_values])

start_length = 10
for i, angle in enumerate(range(0, 50, 10)):
    rotation_center = np.mean(curve, axis=0)
    rotation_angle = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    rotated_curve = np.dot(curve - rotation_center, rotation_matrix) + rotation_center

    # Plot the S shape
    ax.plot(
        rotated_curve[:, 1],
        rotated_curve[:, 0],
        color="w",
        linewidth=start_length - 2 * i,
        alpha=1 - i / 8,
    )

# Plot the S shape
ax.plot(curve[:, 1], curve[:, 0], color="w", linewidth=2)

ax.scatter(
    dots_keep[dots_class == 0, 1],
    dots_keep[dots_class == 0, 0],
    color="w",
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_keep[dots_class == 1, 1],
    dots_keep[dots_class == 1, 0],
    color="w",
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

dots_rotated = np.dot(dots_keep - rotation_center, rotation_matrix) + rotation_center

ax.scatter(
    dots_rotated[dots_class == 0, 1],
    dots_rotated[dots_class == 0, 0],
    color="w",
    marker="o",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

ax.scatter(
    dots_rotated[dots_class == 1, 1],
    dots_rotated[dots_class == 1, 0],
    color="w",
    marker="s",
    s=s,
    alpha=alpha,
    linewidths=linewidths,
)

# Hide axes
ax.axis("off")

fontsize = 250
y_axis = 1.5
ax.text(
    10,
    y_axis,
    r"\bf\textsf{K}",
    usetex=True,
    fontsize=fontsize,
    color="w",
)
ax.text(
    16,
    y_axis,
    r"\bf\textsf{A}",
    usetex=True,
    fontsize=fontsize,
    color="w",
)
ax.text(
    22.5,
    y_axis,
    r"\bf\textsf{D}",
    usetex=True,
    fontsize=fontsize,
    color="w",
)
ax.text(
    28,
    y_axis,
    r"\bf\textsf{A}",
    usetex=True,
    fontsize=fontsize,
    color="w",
)


# Save the figure
plt.savefig("skada_logo_full_white.svg", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig("skada_logo_full_white.pdf", dpi=300, bbox_inches="tight", transparent=True)
