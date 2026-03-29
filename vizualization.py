import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
points = np.random.randn(30, 2)

a1, a2 = 10.0, 2.0
k1, k2 = 1.0, 2.0
k3, k4 = 1.0, 10.0
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

        total += g_par * g_perp

        grad_x += -cos * g_par_p * g_perp + sin * g_par * g_perp_p
        grad_y += -sin * g_par_p * g_perp - cos * g_par * g_perp_p
        grad_phi += n * g_par_p * g_perp - u * g_par * g_perp_p

        d = np.sqrt(dx**2 + dy**2 + delta**2)
        sp = sigmoid(k3 * (eps - d))

        total -= k4 * softplus(k3 * (eps - d))

        grad_x += k3 * k4 * sp * (x0 - xi) / d
        grad_y += k3 * k4 * sp * (y0 - yi) / d

    return total, grad_x, grad_y, grad_phi


phi = 0.5
X, Y = np.meshgrid(np.linspace(-10, 10, 30),
                   np.linspace(-10, 10, 30))
U = np.zeros_like(X)
V = np.zeros_like(Y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        val, gx, gy, _ = compute(X[i,j], Y[i,j], phi)
        U[i,j] = gx
        V[i,j] = gy
        Z[i,j] = val

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1)
ax1.quiver(X, Y, U, V)
ax1.scatter(points[:,0], points[:,1])
ax1.set_title("Gradient field")

ax2 = fig.add_subplot(1, 3, 2)
contour = ax2.contourf(X, Y, Z, levels=50)
fig.colorbar(contour, ax=ax2)
ax2.scatter(points[:,0], points[:,1])
ax2.set_title("Objective function")

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X, Y, Z)
ax3.set_title("3D surface")

plt.suptitle("Visualization of the objective function and its gradient")
plt.tight_layout()
plt.savefig("gradient_visualization.png")
plt.show()