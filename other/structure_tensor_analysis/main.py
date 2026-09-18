"""
Fiber-orientation analysis for HO vs VasHO fluorescence images.

Method:
Structure-tensor analysis (the same math used by ImageJ's
OrientationJ / FibrilTool / scikit-image).

For every pixel we compute:
    - Local orientation
    - Coherency (0 = isotropic, 1 = perfectly aligned)

We then aggregate across the image to obtain:

    Mean coherency
        Weighted by gradient energy -> local alignment

    Orientational Order Parameter (OOP)
        OOP = |<exp(i * 2 * theta)>|
        -> global alignment

    Dominant orientation
        Circular mean of 2 * theta, then divided by 2

    Circular standard deviation of orientation

    Angular histogram
        0–180 degrees

Higher OOP / coherency means more organised / better aligned fibres.
"""

import os
import json
import math
import csv

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mc


# ============================================================================
# Paths
# ============================================================================

IN_DIR = "~/in"
OUT_DIR = "~/out"

os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    [
        f
        for f in os.listdir(IN_DIR)
        if f.lower().endswith(".tif")
    ]
)

if not FILES:
    raise RuntimeError(f"No .tif files found in: {IN_DIR}")


# ============================================================================
# Helper functions
# ============================================================================

def gaussian_kernel_1d(sigma, radius=None):
    """Create a normalized 1-D Gaussian kernel."""

    if radius is None:
        radius = int(max(1, round(3.0 * sigma)))

    x = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32,
    )

    k = np.exp(
        -(x * x) / (2.0 * sigma * sigma)
    )

    k /= k.sum()

    return k


def conv1d(arr, k, axis):
    """
    Apply 1-D convolution along one axis.

    Reflect padding is used at the image boundary.
    """

    pad = len(k) // 2

    pads = [(0, 0)] * arr.ndim
    pads[axis] = (pad, pad)

    a = np.pad(
        arr,
        pads,
        mode="reflect",
    )

    out = np.zeros_like(
        arr,
        dtype=np.float32,
    )

    for i, weight in enumerate(k):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(
            i,
            i + arr.shape[axis],
        )

        out += weight * a[tuple(sl)]

    return out


def gaussian_blur(arr, sigma):
    """Apply separable 2-D Gaussian blur."""

    k = gaussian_kernel_1d(sigma)

    return conv1d(
        conv1d(arr, k, axis=0),
        k,
        axis=1,
    )


def sobel_xy(arr):
    """
    Calculate Sobel-like x/y image gradients.

    Sobel_x:
        smoothing along y
        derivative along x

    Sobel_y:
        derivative along y
        smoothing along x
    """

    smooth = np.array(
        [1, 2, 1],
        dtype=np.float32,
    ) / 4.0

    deriv = np.array(
        [-1, 0, 1],
        dtype=np.float32,
    ) / 2.0

    gx = conv1d(
        conv1d(arr, smooth, axis=0),
        deriv,
        axis=1,
    )

    gy = conv1d(
        conv1d(arr, deriv, axis=0),
        smooth,
        axis=1,
    )

    return gx, gy


# ============================================================================
# Structure tensor analysis
# ============================================================================

def structure_tensor_metrics(
    img,
    sigma_grad=1.5,
    sigma_tensor=8.0,
    energy_percentile=60,
):
    """
    Calculate fibre orientation metrics for a 2-D image.

    Returns
    -------
    dict
        OOP
        mean_coherency
        dominant_deg
        circ_std_deg
        hist
        edges
        theta
        coh
        energy
        foreground
    """

    im = img.astype(np.float32)

    if im.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got shape {im.shape}"
        )

    # ------------------------------------------------------------------
    # Mild pre-smoothing at the derivative scale
    # ------------------------------------------------------------------

    im = gaussian_blur(
        im,
        sigma_grad,
    )

    gx, gy = sobel_xy(im)

    # ------------------------------------------------------------------
    # Structure tensor components
    # ------------------------------------------------------------------

    Jxx = gx * gx
    Jyy = gy * gy
    Jxy = gx * gy

    # Integration window for structure tensor
    Jxx = gaussian_blur(
        Jxx,
        sigma_tensor,
    )

    Jyy = gaussian_blur(
        Jyy,
        sigma_tensor,
    )

    Jxy = gaussian_blur(
        Jxy,
        sigma_tensor,
    )

    # ------------------------------------------------------------------
    # Eigenvalues of the 2x2 symmetric structure tensor
    # ------------------------------------------------------------------

    tr = Jxx + Jyy
    diff = Jxx - Jyy

    s = np.sqrt(
        diff * diff
        + 4.0 * Jxy * Jxy
    )

    # Larger eigenvalue: across fibre
    lam1 = 0.5 * (tr + s)

    # Smaller eigenvalue: along fibre
    lam2 = 0.5 * (tr - s)

    # ------------------------------------------------------------------
    # Coherency
    #
    # 0 = isotropic
    # 1 = perfectly aligned
    # ------------------------------------------------------------------

    coh = np.zeros_like(tr)

    valid = tr > 1e-12

    coh[valid] = (
        (lam1[valid] - lam2[valid])
        / (lam1[valid] + lam2[valid])
    )

    # ------------------------------------------------------------------
    # Fibre orientation
    #
    # Orientation follows the smallest eigenvector.
    # theta is in approximately [-pi/2, pi/2).
    #
    # theta = 0.5 * atan2(2*Jxy, Jyy-Jxx)
    # ------------------------------------------------------------------

    theta = 0.5 * np.arctan2(
        2.0 * Jxy,
        Jyy - Jxx,
    )

    # ------------------------------------------------------------------
    # Gradient energy / foreground
    # ------------------------------------------------------------------

    energy = tr

    threshold = np.percentile(
        energy,
        energy_percentile,
    )

    # Keep high-energy pixels only
    w = np.where(
        energy >= threshold,
        energy,
        0.0,
    )

    # Weight again by local coherency
    w = w * coh

    wsum = w.sum()

    if wsum <= 0:
        wsum = 1.0

    # ------------------------------------------------------------------
    # Orientational Order Parameter
    #
    # OOP = | < exp(i * 2 * theta) > |
    #
    # 0 = random orientation
    # 1 = perfectly aligned
    # ------------------------------------------------------------------

    c2 = (
        w * np.cos(2.0 * theta)
    ).sum() / wsum

    s2 = (
        w * np.sin(2.0 * theta)
    ).sum() / wsum

    OOP = float(
        np.sqrt(
            c2 * c2
            + s2 * s2
        )
    )

    # ------------------------------------------------------------------
    # Dominant orientation
    # ------------------------------------------------------------------

    dom_theta = 0.5 * math.atan2(
        s2,
        c2,
    )

    dom_deg = (
        math.degrees(dom_theta)
        + 180.0
    ) % 180.0

    # ------------------------------------------------------------------
    # Circular standard deviation
    #
    # Calculated in doubled-angle space,
    # then divided by 2.
    # ------------------------------------------------------------------

    R = math.sqrt(
        c2 * c2
        + s2 * s2
    )

    circ_std_deg = math.degrees(
        0.5
        * math.sqrt(
            max(
                0.0,
                -2.0
                * math.log(
                    max(R, 1e-9)
                ),
            )
        )
    )

    # ------------------------------------------------------------------
    # Mean coherency
    # ------------------------------------------------------------------

    mean_coh = float(
        (w * coh).sum() / wsum
    )

    # ------------------------------------------------------------------
    # Angular histogram
    # ------------------------------------------------------------------

    foreground = energy >= threshold

    ang = (
        np.degrees(theta[foreground])
        + 180.0
    ) % 180.0

    hist, edges = np.histogram(
        ang,
        bins=36,
        range=(0, 180),
        weights=(
            coh[foreground]
            * energy[foreground]
        ),
    )

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return {
        "OOP": OOP,
        "mean_coherency": mean_coh,
        "dominant_deg": dom_deg,
        "circ_std_deg": circ_std_deg,
        "hist": hist.tolist(),
        "edges": edges.tolist(),
        "theta": theta,
        "coh": coh,
        "energy": energy,
        "foreground": foreground,
    }


# ============================================================================
# Run analysis
# ============================================================================

rows = []
per_image = {}

for f in FILES:

    path = os.path.join(
        IN_DIR,
        f,
    )

    img = np.array(
        Image.open(path)
    )

    print(
        f"Processing {f}: "
        f"shape={img.shape}, "
        f"dtype={img.dtype}"
    )

    metrics = structure_tensor_metrics(img)

    per_image[f] = metrics

    rows.append(
        {
            "file": f,
            "group": (
                "VasHO"
                if f.startswith("VasHO")
                else "HO"
            ),
            "OOP": metrics["OOP"],
            "mean_coherency": metrics["mean_coherency"],
            "dominant_deg": metrics["dominant_deg"],
            "circ_std_deg": metrics["circ_std_deg"],
        }
    )


# ============================================================================
# Save CSV
# ============================================================================

csv_path = os.path.join(
    OUT_DIR,
    "orientation_metrics.csv",
)

with open(
    csv_path,
    "w",
    newline="",
) as fh:

    writer = csv.DictWriter(
        fh,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

print("Wrote:", csv_path)


# ============================================================================
# Group statistics
# ============================================================================

def group_stats(group):
    """Calculate group-level summary statistics."""

    vals = [
        row
        for row in rows
        if row["group"] == group
    ]

    out = {}

    for key in (
        "OOP",
        "mean_coherency",
        "circ_std_deg",
    ):

        xs = [
            value[key]
            for value in vals
        ]

        if len(xs) == 0:
            out[key] = (
                float("nan"),
                float("nan"),
            )
            continue

        mean = float(
            np.mean(xs)
        )

        sd = (
            float(
                np.std(
                    xs,
                    ddof=1,
                )
            )
            if len(xs) > 1
            else 0.0
        )

        out[key] = (
            mean,
            sd,
        )

    return out, vals


ho_stats, ho = group_stats("HO")
vh_stats, vh = group_stats("VasHO")

print("HO:", ho_stats)
print("VasHO:", vh_stats)


# ============================================================================
# Save JSON summary
# ============================================================================

json_path = os.path.join(
    OUT_DIR,
    "group_summary.json",
)

with open(
    json_path,
    "w",
) as fh:

    json.dump(
        {
            "HO": ho_stats,
            "VasHO": vh_stats,
            "per_image": rows,
        },
        fh,
        indent=2,
    )

print("Wrote:", json_path)


# ============================================================================
# Visualization helper
# ============================================================================

def orient_rgb(
    img,
    theta,
    coh,
    energy,
    thresh_pct=60,
):
    """
    Convert local fibre orientation into HSV/RGB visualization.

    Hue       = orientation
    Saturation = coherency
    Value      = image intensity
    """

    threshold = np.percentile(
        energy,
        thresh_pct,
    )

    value = np.clip(
        (
            img - img.min()
        )
        / (
            img.max()
            - img.min()
            + 1e-9
        ),
        0,
        1,
    )

    hue = (
        (
            np.degrees(theta)
            + 180.0
        )
        % 180.0
    ) / 180.0

    saturation = np.clip(
        coh,
        0,
        1,
    )

    # Remove orientation colour from low-energy background
    saturation = np.where(
        energy >= threshold,
        saturation,
        0.0,
    )

    hsv = np.stack(
        [
            hue,
            saturation,
            value,
        ],
        axis=-1,
    ).astype(np.float32)

    rgb = mc.hsv_to_rgb(hsv)

    return rgb


# ============================================================================
# Figure 1:
# Original + orientation map + angular histogram
# ============================================================================

n = len(FILES)

fig, axes = plt.subplots(
    n,
    3,
    figsize=(14, 4.5 * n),
    squeeze=False,
)

for i, f in enumerate(FILES):

    metrics = per_image[f]

    img = np.array(
        Image.open(
            os.path.join(
                IN_DIR,
                f,
            )
        )
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Raw image
    # ------------------------------------------------------------------

    axes[i, 0].imshow(
        np.log1p(img),
        cmap="gray",
    )

    axes[i, 0].set_title(
        f"{f}\nraw (log)"
    )

    axes[i, 0].axis("off")

    # ------------------------------------------------------------------
    # Orientation-coloured image
    # ------------------------------------------------------------------

    rgb = orient_rgb(
        img,
        metrics["theta"],
        metrics["coh"],
        metrics["energy"],
    )

    axes[i, 1].imshow(rgb)

    axes[i, 1].set_title(
        "orientation (hue) × coherency (sat)\n"
        f"OOP={metrics['OOP']:.3f}, "
        f"dom={metrics['dominant_deg']:.1f}°"
    )

    axes[i, 1].axis("off")

    # ------------------------------------------------------------------
    # Angular histogram
    # ------------------------------------------------------------------

    bin_centres = (
        np.array(metrics["edges"][:-1])
        + 2.5
    )

    axes[i, 2].bar(
        bin_centres,
        metrics["hist"],
        width=4.5,
        color="steelblue",
        edgecolor="k",
    )

    axes[i, 2].set_xlabel(
        "angle (°)"
    )

    axes[i, 2].set_ylabel(
        "weighted fraction"
    )

    axes[i, 2].set_title(
        "angular histogram\n"
        f"circ-SD={metrics['circ_std_deg']:.1f}°"
    )


fig.tight_layout()

fig_path = os.path.join(
    OUT_DIR,
    "per_image_orientation.png",
)

fig.savefig(
    fig_path,
    dpi=110,
)

plt.close(fig)

print("Wrote:", fig_path)


# ============================================================================
# Figure 2:
# Group comparison
# ============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
)

groups = [
    "HO",
    "VasHO",
]

colors = [
    "#4C72B0",
    "#DD8452",
]

metrics_to_plot = [
    "OOP",
    "mean_coherency",
    "circ_std_deg",
]

labels = [
    "Orientational Order Parameter\n(↑ = more aligned)",
    "Mean coherency\n(↑ = more aligned)",
    "Circular SD of orientation\n(↓ = more aligned)",
]

for ax, key, label in zip(
    axes,
    metrics_to_plot,
    labels,
):

    for j, group in enumerate(groups):

        xs = [
            row[key]
            for row in rows
            if row["group"] == group
        ]

        if not xs:
            continue

        ax.scatter(
            [j] * len(xs),
            xs,
            color=colors[j],
            s=60,
            zorder=3,
        )

        ax.hlines(
            np.mean(xs),
            j - 0.2,
            j + 0.2,
            color="k",
            lw=2,
        )

    ax.set_xticks(
        range(len(groups))
    )

    ax.set_xticklabels(groups)

    ax.set_title(label)

    ax.grid(alpha=0.3)


fig.tight_layout()

cmp_path = os.path.join(
    OUT_DIR,
    "group_comparison.png",
)

fig.savefig(
    cmp_path,
    dpi=120,
)

plt.close(fig)

print("Wrote:", cmp_path)


# ============================================================================
# Figure 3:
# Overlaid polar / rose plots
# ============================================================================

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6.5, 6.5),
    subplot_kw={
        "projection": "polar"
    },
)

for group, color in zip(
    groups,
    colors,
):

    group_metrics = [
        per_image[row["file"]]
        for row in rows
        if row["group"] == group
    ]

    if not group_metrics:
        continue

    # Mean angular histogram across images
    h = np.mean(
        [
            np.array(x["hist"])
            for x in group_metrics
        ],
        axis=0,
    )

    edges = np.array(
        group_metrics[0]["edges"]
    )

    centres_deg = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    # Orientation is modulo 180 degrees,
    # so mirror into the other half of polar coordinates.
    theta_plot = np.deg2rad(
        np.concatenate(
            [
                centres_deg,
                centres_deg + 180,
            ]
        )
    )

    r = np.concatenate(
        [
            h,
            h,
        ]
    )

    # Close the curve
    theta_closed = np.append(
        theta_plot,
        theta_plot[0],
    )

    r_closed = np.append(
        r,
        r[0],
    )

    ax.plot(
        theta_closed,
        r_closed,
        color=color,
        lw=2,
        label=group,
    )

    ax.fill(
        theta_plot,
        r,
        color=color,
        alpha=0.2,
    )


ax.set_theta_zero_location("E")
ax.set_theta_direction(1)

ax.set_title(
    "Mean angular distribution per group"
)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.25, 1.1),
)

fig.tight_layout()

rose_path = os.path.join(
    OUT_DIR,
    "group_rose.png",
)

fig.savefig(
    rose_path,
    dpi=120,
)

plt.close(fig)

print("Wrote:", rose_path)
