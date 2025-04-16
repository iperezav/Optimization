import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0.7, 0.3], dtype = float)
eps = float(1e-13)
Niter = 5500
gamma = float(1e-3)


def obj_func(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def deriv_i(x0, i):
    fx0 = obj_func(x0)
    x1 = x0
    x1[i] = x0[i] + eps
    fx1 = obj_func(x1)
    return (fx1-fx0)/eps

def grad(x0):
    df0 = deriv_i(x0, 0)
    df1 = deriv_i(x0, 1)
    return np.array([df0, df1])

x_list = x0

for i in np.arange(Niter):
    dx = grad(x0)
    x0 = x0 - gamma*dx
    x_list = np.vstack([x_list, x0])


print(x_list)

y_list = np.apply_along_axis(obj_func, 1, x_list)

plt_x_1 = np.linspace(-2, 2, 1000)
plt_x_2 = np.linspace(-2, 2, 1000)


plt_xx_1, plt_xx_2 = np.meshgrid(plt_x_1, plt_x_2)

plt_fx = 100*(plt_xx_2-plt_xx_1**2)**2 + (1-plt_xx_1)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plt_xx_1, plt_xx_2, plt_fx, cmap='viridis', alpha=0.6)

# Add labels and a color bar
ax.set_title("3D Surface Plot of $f(x) = 100(x_2-x_1^2)^2 + (1-x_1)^2$ with Optimization Path")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("f(x)")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot the optimization path as a line
ax.plot(x_list[:, 0], x_list[:, 1], y_list.flatten(), color='red', label='Optimization Path')
ax.legend()

# Show the plot
plt.show()