import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Grid
# ============================================================

N = 3000

x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)

X, Y = np.meshgrid(x, y)

# ============================================================
# Test wavefunction
# ============================================================

r = np.sqrt(X**2 + Y**2)

#psi = np.exp(-r) * np.exp(1j * r)
m = 3

theta = np.arctan2(Y,X)

psi = (
    r**m
    *
    np.exp(-r**2/4)
    *
    np.exp(1j*m*theta)
)
# ============================================================
# Extract phase & amplitude
# ============================================================

phase = np.angle(psi)

amp = np.abs(psi)

# ============================================================
# Phase -> Twilight colormap
# ============================================================

phase_norm = (phase + np.pi) / (2*np.pi)

rgb = plt.cm.twilight_shifted(
    phase_norm
)[..., :3]

# ============================================================
# Amplitude -> Brightness
# ============================================================

amp_scaled = np.log1p(15 * amp)

amp_scaled /= amp_scaled.max()

# Gamma correction
gamma = 0.55

brightness = amp_scaled**gamma

# ============================================================
# Apply brightness
# ============================================================

rgb = rgb * brightness[..., None]

# ============================================================
# Optional: enhance contrast
# ============================================================

rgb = np.clip(rgb, 0, 1)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(10,10),
    dpi=100
)

ax.imshow(
    rgb,
    origin='lower',
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    interpolation='nearest'
)

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

ax.set_title(
    r"$\psi=e^{-r}e^{ir}$",
    fontsize=18
)

plt.tight_layout()
plt.show()