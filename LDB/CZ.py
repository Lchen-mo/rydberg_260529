import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pulser import Register
from pulser import Sequence
from pulser import Pulse

from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform

from pulser_simulation import QutipEmulator

from qutip import (
    ptrace,
    sigmax,
    sigmay,
    sigmaz,
    Bloch
)

##############################################################################
# Register
##############################################################################

reg = Register.from_coordinates(
    [
        (0.0, 0.0),
        (5.0, 0.0)
    ],
    prefix="q"
)

##############################################################################
# Sequence
##############################################################################

seq = Sequence(
    reg,
    DigitalAnalogDevice
)

seq.declare_channel(
    "ryd",
    "rydberg_global"
)

##############################################################################
# Pulse
##############################################################################

amp = 2 * np.pi      # 6.283 rad/us

pulse = Pulse(
    ConstantWaveform(500, amp),
    ConstantWaveform(500, 0.0),
    0.0
)

seq.add(
    pulse,
    "ryd"
)

##############################################################################
# Draw sequence
##############################################################################

seq.draw()

##############################################################################
# Emulator
##############################################################################

sim = QutipEmulator.from_sequence(
    seq
)

##############################################################################
# Run
##############################################################################

result = sim.run()

##############################################################################
# States
##############################################################################

states = result.states

print("Number of states =", len(states))

##############################################################################
# Bloch trajectory
##############################################################################

xs = []
ys = []
zs = []

for psi in states:

    rho = psi.proj()

    rho0 = ptrace(rho, 0)

    xs.append(
        np.real(
            (rho0 * sigmax()).tr()
        )
    )

    ys.append(
        np.real(
            (rho0 * sigmay()).tr()
        )
    )

    zs.append(
        np.real(
            (rho0 * sigmaz()).tr()
        )
    )

##############################################################################
# Bloch sphere
##############################################################################

b = Bloch()

b.add_points(
    [xs, ys, zs]
)

b.render()

plt.show()

##############################################################################
# Final state
##############################################################################

print()
print("Final state:")
print(states[-1])