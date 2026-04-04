import numpy as np
import matplotlib.pyplot as plt


def generate_points(func, x_range=(-3, 3), n=30):
    x = np.linspace(x_range[0], x_range[1], n)
    y = func(x)
    return np.stack((x, y), axis=1)

points = generate_points(lambda x: x)

a1, a2 = 10.0, 2.0
k1, k2, k3, k4 = 1.0, 1.0, 1.0, 1.0
p1, p2 = 4.0, 4.0
eps = 1.0
delta = 1e-3

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softplus function
def softplus(x):
    return np.log(1 + np.exp(x))

# Compute objective and its gradient
def compute(x0, y0, phi):
    cos, sin = np.cos(phi), np.sin(phi)

    total = 0
    grad_x, grad_y, grad_phi = 0, 0, 0

    for xi, yi in points:
        dx, dy = xi - x0, yi - y0

        u = cos * dx + sin * dy
        n = -sin * dx + cos * dy

        A = sigmoid(k1 * u)
        B = sigmoid(k1 * (a1 - u))
        C = sigmoid(k2 * (n + a2))
        D = sigmoid(k2 * (a2 - n))

        g_par = A * B
        g_perp = C * D

        g_par_p = k1 * g_par * (B - A)
        g_perp_p = k2 * g_perp * (D - C)

        g_par_pow = g_par**p1
        g_perp_pow = g_perp**p2

        total += g_par_pow * g_perp_pow

        grad_x += (
            -p1 * cos * g_par_p * (g_par**(p1 - 1)) * g_perp_pow
            + p2 * sin * g_perp_p * g_par_pow * (g_perp**(p2 - 1))
        )

        grad_y += (
            -p1 * sin * g_par_p * (g_par**(p1 - 1)) * g_perp_pow
            - p2 * cos * g_perp_p * g_par_pow * (g_perp**(p2 - 1))
        )

        grad_phi += (
            p1 * n * g_par_p * (g_par**(p1 - 1)) * g_perp_pow
            - p2 * u * g_perp_p * g_par_pow * (g_perp**(p2 - 1))
        )

        d = np.sqrt(dx**2 + dy**2 + delta**2)
        sp = sigmoid(k3 * (eps - d))

        total -= k4 * softplus(k3 * (eps - d))

        grad_x += k3 * k4 * sp * (x0 - xi) / d
        grad_y += k3 * k4 * sp * (y0 - yi) / d

    return total, grad_x, grad_y, grad_phi

X, Y = np.meshgrid(np.linspace(-10, 10, 30),
                   np.linspace(-10, 10, 30))

U = np.zeros_like(X)
V = np.zeros_like(Y)
Z = np.zeros_like(X)

phis = np.linspace(0, 2*np.pi, 360)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        val_sum = 0
        gx_sum = 0
        gy_sum = 0
        for phi in phis:
            val, gx, gy, _ = compute(X[i,j], Y[i,j], phi)
            val_sum += val
            gx_sum += gx
            gy_sum += gy
        Z[i,j] = val_sum / len(phis)
        U[i,j] = gx_sum / len(phis)
        V[i,j] = gy_sum / len(phis)

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1)
magnitude = np.sqrt(U**2 + V**2)
U_norm = U / (magnitude + 1e-6)
V_norm = V / (magnitude + 1e-6)

ax1.quiver(X, Y, U_norm, V_norm)
ax1.quiver(X, Y, U, V)
ax1.scatter(points[:,0], points[:,1], color='red')
ax1.set_title("Average Gradient field", fontsize=16)

ax2 = fig.add_subplot(1, 3, 2)
contour = ax2.contourf(X, Y, Z, levels=150)
fig.colorbar(contour, ax=ax2)
ax2.scatter(points[:,0], points[:,1], color='red')
ax2.set_title("Average Objective function", fontsize=16)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', vmin=-50, vmax=np.max(Z))
ax3.set_title("3D surface", fontsize=16)

plt.suptitle("Visualization of the objective function and its gradient (linear points, averaged over 360 angles)", fontsize=18)
plt.tight_layout()
plt.savefig("gradient_visualization.png")
plt.show()