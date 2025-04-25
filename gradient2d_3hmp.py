import numpy as np
import matplotlib.pyplot as plt

# initial point
x0 = np.array([1.2, 1.5], dtype = float)
# size to compute derivative
eps = float(1e-13)
# learning iterations
Niter = 5500
# learning rate
gamma = float(1e-1)
# error size
beta = float(1e-1)
# equilibrium
eq = np.array([0,0], dtype = float)

# 3-hump camel function
def obj_func(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

# derivative
def deriv_i(x0, i):
    xbh = np.copy(x0)
    xbh[i] = xbh[i] - eps
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
x_list = x0

# learning
for i in np.arange(Niter):
    dx = grad(x0)
    x0 = x0 - gamma*dx
    x_list = np.vstack([x_list, x0])
    if np.linalg.norm(eq-x0) < beta: break


print(f'Number of iterations: {i}')
print(x_list)

y_list = np.apply_along_axis(obj_func, 1, x_list)

plt_x_1 = np.linspace(-2, 2, 20)
plt_x_2 = np.linspace(-2, 2, 20)



plt_xx_1, plt_xx_2 = np.meshgrid(plt_x_1, plt_x_2)

plt_fx = 2*plt_xx_1**2 - 1.05*plt_xx_1**4 + plt_xx_1**6/6 + plt_xx_1*plt_xx_2 + plt_xx_2**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot with custom colormap and transparency
surf = ax.plot_surface(plt_xx_1, plt_xx_2, plt_fx, cmap='plasma', alpha=0.65)

# Add labels and color bar
ax.set_title("3D Surface Plot of $f(x)$ with Optimization Path", fontsize=16)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.set_zlabel("$f(x)$", fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot optimization path
ax.plot(x_list[:, 0], x_list[:, 1], y_list.flatten(), color='red', linestyle='--', marker='o', label='Optimization Path')
ax.legend()

# Adjust the view angle for a better perspective
ax.view_init(elev=30, azim=120)

# Add grid for better referencing
ax.grid(True)

# Show the plot
plt.show()