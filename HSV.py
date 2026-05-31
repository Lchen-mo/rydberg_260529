import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ============================================================
# Example wavefunction
# Replace psi with your own wavefunction
# ============================================================

N = 3000

x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)

X, Y = np.meshgrid(x, y)

m = 4

sigma = 1.5

r = np.sqrt(X**2 + Y**2)

theta = np.arctan2(Y,X)

psi = (
    np.exp(-r)*np.exp( 1j*r  )
)

# ============================================================
# Domain Coloring
# ============================================================

phase = np.angle(psi)
amp = np.abs(psi)

# ------------------------------------------------------------
# Phase -> Hue
# ------------------------------------------------------------

h = (phase + np.pi) / (2 * np.pi)

# Full saturation
s = np.ones_like(h)

# ------------------------------------------------------------
# Amplitude compression
# ------------------------------------------------------------

amp_log = np.log1p(amp)
#amp_log = amp
amp_log /= amp_log.max()

# Gamma correction
gamma = 1

brightness = amp_log ** gamma

# ------------------------------------------------------------
# Modulus contours
# log-spaced magnitude rings
# ------------------------------------------------------------

log_amp = np.log2(amp + 1e-14)

modulus_lines = (
    0.5
    + 0.5 * np.cos(
        2 * np.pi * log_amp
    )
)

# ------------------------------------------------------------
# Phase contours
# highlight phase structure
# ------------------------------------------------------------

phase_lines = (
    0.5
    + 5 * np.cos(
        4 * phase
    )
)

# ------------------------------------------------------------
# Combine brightness
# ------------------------------------------------------------

v = brightness 


v = np.clip(v, 0, 1)

# ------------------------------------------------------------
# HSV -> RGB
# ------------------------------------------------------------

hsv = np.stack(
    (
        h,
        s,
        v
    ),
    axis=-1
)

rgb = hsv_to_rgb(hsv)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 10),
    dpi=100
)

ax.imshow(
    rgb,
    origin="lower",
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    interpolation="nearest"
)

ax.set_xlabel("Re(z)", fontsize=14)
ax.set_ylabel("Im(z)", fontsize=14)

ax.set_title(
    r"Domain Coloring ",
    fontsize=16
)

plt.tight_layout()

plt.show()