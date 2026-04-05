import numpy as np
import matplotlib.pyplot as plt
from mat_model import sigmoid

a1 = 5
a2 = 5
k1 = 2
k2 = 2
p1 = 3
p2 = 3

xs = np.linspace(-10, 10, 500)

ys1 = (sigmoid(k1 * xs) * sigmoid(k1 * (a1 - xs)))**p2

ys2 = (sigmoid(k2 * (xs + a2)) * sigmoid(k2 * (a2 - xs)))**p1

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(xs, ys1)
axs[0].set_title(
    rf"$g_{{\parallel}}^{{p_1}}\ [0, a_1]$",
    fontsize=16
)
axs[0].axvline(0, linestyle='--')
axs[0].axvline(a1, linestyle='--')
axs[0].grid()

axs[1].plot(xs, ys2)
axs[1].set_title(
    rf"$g_{{\perp}}^{{p_2}}\ [-a_2, a_2]$",
    fontsize=16
)
axs[1].axvline(a2, linestyle='--')
axs[1].axvline(-a2, linestyle='--')
axs[1].grid()

plt.suptitle(
    f"Indicators smoothed by sigmoid functions\n"
    f"$a_1={a1}$, $a_2={a2}$, $k_1={k1}$, $k_2={k2}$, $p_1={p1}$, $p_2={p2}$",
    fontsize=20,
)
plt.tight_layout()
plt.savefig("smoothed_indicators.png")
plt.show()