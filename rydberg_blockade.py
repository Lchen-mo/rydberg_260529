import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ==========================
# 参数
# ==========================

Omega = 2*np.pi*1.0

# 修改这里观察效果
V = 20*Omega

T = 5
N = 400

times = np.linspace(0,T,N)

# ==========================
# 基底:
#
# |gg> = [1,0,0,0]
# |gr> = [0,1,0,0]
# |rg> = [0,0,1,0]
# |rr> = [0,0,0,1]
#
# ==========================

# drive Hamiltonian

H_drive = 0.5*Omega*np.array(

[
    [0,1,1,0],
    [1,0,0,1],
    [1,0,0,1],
    [0,1,1,0]
],

dtype=complex

)

# interaction

H_int = np.diag([0,0,0,V])

H = H_drive + H_int

# ==========================
# 初态 |gg>
# ==========================

psi0 = np.array(
    [1,0,0,0],
    dtype=complex
)

Pgg=[]
Pgr=[]
Prg=[]
Prr=[]

for t in times:

    U = expm(-1j*H*t)

    psi = U @ psi0

    Pgg.append(abs(psi[0])**2)
    Pgr.append(abs(psi[1])**2)
    Prg.append(abs(psi[2])**2)
    Prr.append(abs(psi[3])**2)

# ==========================
# plot
# ==========================

plt.figure(figsize=(10,6))

plt.plot(times,Pgg,label="|gg>")
plt.plot(times,Pgr,label="|gr>")
plt.plot(times,Prg,label="|rg>")
plt.plot(times,Prr,label="|rr>")

plt.xlabel("Time")
plt.ylabel("Probability")

plt.title(f"Rydberg Blockade   V={V}")

plt.legend()
plt.grid()

plt.show()