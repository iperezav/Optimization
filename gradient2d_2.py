import numpy as np
import matplotlib.pyplot as plt

# initial point
x0 = np.array([1.2, 1.5], dtype = float)
# size to compute derivative
eps = float(1e-13)
# learning iterations
Niter = 5500
# learning rate
gamma = float(1e-3)
# error size
beta = float(1e-1)
# equilibrium
eq = np.array([1,1], dtype = float)

# Rosenbrock function
def obj_func(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

# derivative
def deriv_i(x0, i):
    xbh = np.copy(x0)
    xbh[i] = xbh[i] -eps
    fx0 = obj_func(x0)
    xfh = np.copy(x0)
    xfh[i] = xfh[i] + eps
    fx1 = obj_func(xfh)
    return (fx1-fx0)/(2*eps)

# gradient
def grad(x0):
    df0 = deriv_i(x0, 0)
    df1 = deriv_i(x0, 1)
    return np.array([df0, df1])

# initiate the iteration list
x_list = np.copy(x0)

# learning
for i in np.arange(Niter):
    dx = grad(x0)
    x0 = x0 - gamma*dx
    x_list = np.vstack([x_list, x0])
    if np.linalg.norm(eq-x0) < beta: break


print(f'Number of iterations: {i}')
print(x_list)

y_list = np.apply_along_axis(obj_func, 1, x_list)

plt_x_1 = np.linspace(0, 2, 20)
plt_x_2 = np.linspace(0, 2, 20)


plt_xx_1, plt_xx_2 = np.meshgrid(plt_x_1, plt_x_2)

plt_fx = 100*(plt_xx_2-plt_xx_1**2)**2 + (1-plt_xx_1)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plt_xx_1, plt_xx_2, plt_fx, cmap='viridis', alpha=0.65)

# Add labels and a color bar
ax.set_title("3D Surface Plot of $f(x) = 100(x_2-x_1^2)^2 + (1-x_1)^2$ with Optimization Path")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("f(x)")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot the optimization path as a line
ax.plot(x_list[:, 0], x_list[:, 1], y_list.flatten(), color='red', linestyle = '--', marker = 'o',
        markersize = 1, label='Optimization Path')
ax.legend()

# Show the plot
plt.show()