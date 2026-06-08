
import matplotlib
matplotlib.use("TkAgg")
from arc import *

atom = Rubidium87()

wf = Wavefunction(
    atom,
    [[70,0,0.5,0.5]],
    [1.0]
)

radial = wf.basisWavefunctions[0]

print(type(radial))

print(radial.x.shape)
print(radial.y.shape)

print(radial.x[:10])
print(radial.y[:10])

print(radial.x[-10:])
print(radial.y[-10:])

import matplotlib.pyplot as plt

r = radial.x
u = radial.y

plt.figure(figsize=(10,5))
plt.plot(r,u)
plt.xlabel("r")
plt.ylabel("wavefunction")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(r,u*u)
plt.xlim(0,10000)
plt.show()

import numpy as np

idx = np.argmax(np.abs(u))

print(r[idx])