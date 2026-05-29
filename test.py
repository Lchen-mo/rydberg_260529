import sys
import numpy as np

print("="*50)
print("STEP 1 : Python 检查")
print("="*50)

print("Python version:")
print(sys.version)

print("\nPython OK\n")

# =====================================================

print("="*50)
print("STEP 2 : numpy 检查")
print("="*50)

try:
    a = np.array([1,2,3])
    print("numpy OK")
    print("array =", a)
except Exception as e:
    print("numpy ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 3 : matplotlib 检查")
print("="*50)

try:
    import matplotlib

    # 强制使用 TkAgg 后端
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt

    print("matplotlib OK")
    print("backend =", matplotlib.get_backend())

    # 测试简单绘图
    x = np.linspace(0,10,100)
    y = np.sin(x)

    plt.figure(figsize=(6,4))
    plt.plot(x,y)

    plt.title("Matplotlib Test")
    plt.xlabel("x")
    plt.ylabel("sin(x)")

    print("\n现在应该弹出一个 sin 曲线窗口")
    print("如果窗口一闪而过，说明 matplotlib 后端有问题")

    plt.show()

    print("matplotlib show() 执行完成")

except Exception as e:
    print("matplotlib ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 4 : QuTiP 检查")
print("="*50)

try:
    import qutip

    print("QuTiP OK")
    print("QuTiP version =", qutip.__version__)

except Exception as e:
    print("QuTiP ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 5 : basis/state 检查")
print("="*50)

try:
    from qutip import *

    g = basis(2,0)
    r = basis(2,1)

    print("basis OK")
    print("g =")
    print(g)

    print("r =")
    print(r)

except Exception as e:
    print("basis ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 6 : Hamiltonian 检查")
print("="*50)

try:
    Omega = 2*np.pi*1.0

    H = 0.5 * Omega * sigmax()

    print("Hamiltonian OK")
    print(H)

except Exception as e:
    print("Hamiltonian ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 7 : sesolve 检查")
print("="*50)

try:
    times = np.linspace(0,5,100)

    result = sesolve(H, g, times)

    print("sesolve OK")

    print("result type =", type(result))

    print("Number of states =", len(result.states))

except Exception as e:
    print("sesolve ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 8 : expectation value 检查")
print("="*50)

try:
    P_g = [expect(g*g.dag(), state) for state in result.states]
    P_r = [expect(r*r.dag(), state) for state in result.states]

    print("expectation value OK")

    print("P_g first 5 =", P_g[:5])
    print("P_r first 5 =", P_r[:5])

except Exception as e:
    print("expectation ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("STEP 9 : 最终绘图检查")
print("="*50)

try:
    plt.figure(figsize=(8,5))

    plt.plot(times, P_g, label='Ground')
    plt.plot(times, P_r, label='Rydberg')

    plt.xlabel("Time")
    plt.ylabel("Population")

    plt.title("Rabi Oscillation Test")

    plt.legend()
    plt.grid()

    print("现在应该弹出 Rabi Oscillation 图")

    plt.show()

    print("最终绘图成功")

except Exception as e:
    print("Final Plot ERROR")
    print(e)

print()

# =====================================================

print("="*50)
print("TEST FINISHED")
print("="*50)

input("Press Enter to exit...")
