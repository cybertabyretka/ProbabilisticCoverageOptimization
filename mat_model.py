import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(42)

a1 = 10.0
a2 = 2.0
k1 = 1.0
k2 = 2.0
k3 = 50.0
k4 = 50.0
eps = 1
delta = 1e-3
n_closest = 1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softplus function
def softplus(x):
    return np.log1p(np.exp(x))

# Projections of points onto the main and perpendicular axes
def compute_projections(x0, y0, phi, pts):
    u_vec = np.array([np.cos(phi), np.sin(phi)])
    n_vec = np.array([-np.sin(phi), np.cos(phi)])
    diff = pts - np.array([x0, y0])
    u = diff @ u_vec
    n = diff @ n_vec
    return u, n

# Capture function by parallel projection
def g_parallel(u):
    return sigmoid(k1 * u) * sigmoid(k1 * (a1 - u))

# Capture function by perpendicular projection
def g_perp(n):
    return sigmoid(k2 * (n + a2)) * sigmoid(k2 * (a2 - n))

# Points generation by functions
def generate_points(func, x_range=(-5, 5), n=100):
    x = np.linspace(x_range[0], x_range[1], n)
    y = func(x)
    return np.stack([x, y], axis=1)

# Optimize function and plot results
def optimize_and_plot(ax, points, title):
    # Objective function for optimization
    def objective(vars):
        x0, y0, phi = vars
        phi = phi % (2 * np.pi)

        u, n = compute_projections(x0, y0, phi, points)
        capture = g_parallel(u) * g_perp(n)
        term1 = np.sum(capture)

        dists = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + delta**2)
        penalty = np.sum(softplus(k3 * (eps - dists)))

        return -(term1 - k4 * penalty)

    # Start point for optimization
    init = np.array([0.0, 0.0, 0.0])
    res = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (0, 2 * np.pi)]
    )

    x0_opt, y0_opt, phi_opt = res.x

    ax.scatter(points[:, 0], points[:, 1], label="Points")

    u_vec = np.array([np.cos(phi_opt), np.sin(phi_opt)])
    origin = np.array([x0_opt, y0_opt])

    main_line = np.array([origin, origin + a1 * u_vec])
    ax.plot(main_line[:, 0], main_line[:, 1], color="red", label="Main")

    n_vec = np.array([-np.sin(phi_opt), np.cos(phi_opt)])
    ax.plot(*(main_line + a2 * n_vec).T, linestyle="--", color="green")
    ax.plot(*(main_line - a2 * n_vec).T, linestyle="--", color="green")

    start = origin
    end = origin + a1 * u_vec

    start_line = np.array([
        start - a2 * n_vec,
        start + a2 * n_vec
    ])
    ax.plot(start_line[:, 0], start_line[:, 1], linestyle="--", color="green")

    end_line = np.array([
        end - a2 * n_vec,
        end + a2 * n_vec
    ])
    ax.plot(end_line[:, 0], end_line[:, 1], linestyle="--", color="green")

    x_grid = np.linspace(-10, 10, 200)
    y_grid = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Compute probabilities on the grid
    u, n = compute_projections(x0_opt, y0_opt, phi_opt, grid_points)
    Z = (g_parallel(u) * g_perp(n)).reshape(X.shape)

    contour = ax.contourf(X, Y, Z, levels=50, alpha=0.3, cmap="viridis")

    distances = np.sqrt((points[:, 0] - x0_opt) ** 2 + (points[:, 1] - y0_opt) ** 2)
    idx = np.argsort(distances)[:n_closest]

    for i in idx:
        pt = points[i]
        dist = distances[i]

        dx = pt[0] - x0_opt
        dy = pt[1] - y0_opt

        ax.arrow(x0_opt, y0_opt, dx, dy,
                 head_width=0.2, length_includes_head=True,
                 color='black')

        ax.text(x0_opt + dx*0.5, y0_opt + dy*0.5,
                f"{dist:.2f}",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.scatter([x0_opt], [y0_opt], color='black', marker='x', s=100, label='Optimal Point')

    ax.set_title(title, fontsize=16)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    return res, contour


functions = [
    lambda x: x,
    lambda x: x**2,
    lambda x: np.sin(x)
]

ranges = [
    (-5, 5),
    (-3, 3),
    (-5, 5)
]

point_sets = [
    generate_points(f, r, 100)
    for f, r in zip(functions, ranges)
]

graph_names = ["Linear", "Quadratic", "Sine", "Random"]

rand_points = np.random.uniform(-5, 5, (100, 2))
point_sets.append(rand_points)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
axes = axes.ravel()

results = []
last_contour = None

for i, (ax, pts) in enumerate(zip(axes, point_sets)):
    res, contour = optimize_and_plot(ax, pts, graph_names[i])
    results.append(res)
    last_contour = contour

fig.suptitle("Optimization of Capture Value", fontsize=20)
fig.colorbar(contour, ax=axes, shrink=0.8, label='Capture Value')
plt.savefig("optimization_results.png")
plt.show()

for i, res in enumerate(results, 1):
    x0, y0, phi = res.x
    print(f"\nGraph {i}")
    print("Optimal:", x0, y0, phi)
    print("Objective:", -res.fun)