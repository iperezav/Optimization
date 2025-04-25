import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# initial point
x0 = np.array([1.2, 1.5], dtype = float)
# size to compute derivative
eps = float(1e-13)
# learning iterations
Niter = 5000
# learning rate
gamma = float(1)
# error size
beta = float(1e-3)
# learning shrink rate
c_le = 0.5
# equilibrium
eq = np.array([0,0], dtype = float)

# 3-hump camel function
def obj_func(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

# derivative
def deriv_i(x0, i):
    xbh = np.copy(x0)
    xbh[i] = xbh[i] - eps
    fx0 = obj_func(xbh)
    xfh = np.copy(x0)
    xfh[i] = xfh[i] + eps
    fx1 = obj_func(xfh)
    return (fx1 - fx0)/(2*eps)

# gradient
def grad(x0):
    df0 = deriv_i(x0, 0)
    df1 = deriv_i(x0, 1)
    return np.array([df0, df1])

# initiate the iteration list
x_list = np.copy(x0)

# learning
for i in np.arange(Niter):
    fx0 = obj_func(x0)
    dx = grad(x0)
    gamma_temp = gamma
    x_temp = x0 - gamma_temp * dx
    fx_temp = obj_func(x_temp)
    # backtracking
    while fx_temp > fx0 :
        gamma_temp = c_le * gamma_temp
        x_temp = x0 - gamma_temp * dx
        fx_temp = obj_func(x_temp)
        print(f'{x_temp, gamma_temp}')
    x_list = np.vstack([x_list, x_temp])
    # update
    x0 = x_temp
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


# Initialize the optimization path
line, = ax.plot([], [], [], color='red', linestyle='--', marker='o', label='Optimization Path')
ax.legend()

# Function to update animation
def update(frame):
    line.set_data(x_list[:frame, 0], x_list[:frame, 1])
    line.set_3d_properties(y_list[:frame].flatten())
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(x_list), interval=200, blit=False)

# Save the animation as a GIF
ani.save("optimization_path.gif", writer=animation.PillowWriter(fps=3))

# Show the plot
plt.show()

