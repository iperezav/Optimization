import numpy as np
import matplotlib.pyplot as plt



x0 = np.array([0.5,0.3], dtype= float)
eps = float(1e-4)
delta = 1e-1
#N0points = 1000
Niter = 1000
gamma = float(0.01)
#x_1 = np.linspace(x0[0]-delta, x0[0]+delta, N0points)
#x_2 = np.linspace(x0[1]-delta, x0[1]+delta, N0points)
x_1 = np.arange(-0.00001, x0[0]+delta, step = eps)
x_2 = np.arange(-0.00001, x0[1]+delta, step = eps)


xx_1, xx_2 = np.meshgrid(x_1, x_2)

fx = xx_1**2+xx_2**2




#print(fx)

def my_interpolation2(x0,x_1,x_2,fx):
    x_index = np.searchsorted(x_1, x0[0])
    y_index = np.searchsorted(x_2, x0[1])
    return fx[y_index-1, x_index-1]

def my_interpolation(x0, x_1, x_2, fx):
    x_index = np.searchsorted(x_1, x0[0])
    y_index = np.searchsorted(x_2, x0[1])
    length_size_x = x_1[x_index] - x_1[x_index-1]
    length_size_y = x_2[y_index] - x_2[y_index-1]
    alpha_x = (x_1[x_index]-x0[0])/length_size_x
    beta_x = (x0[0]-x_1[x_index-1])/length_size_x
    alpha_y = (x_2[y_index]-x0[1])/length_size_y
    beta_y = (x0[1]-x_2[y_index - 1])/length_size_y
    fx_inter1_x = alpha_x*fx[y_index, x_index]+beta_x*fx[y_index, x_index-1]
    fx_inter1_y = alpha_y*fx[y_index, x_index]+beta_y*fx[y_index-1, x_index]
    fx_inter2_x = alpha_x*fx[y_index-1, x_index] + beta_x*fx[y_index - 1, x_index - 1]
    fx_inter2_y = alpha_y*fx[y_index, x_index - 1] + beta_y*fx[y_index - 1, x_index - 1]
    return 0.25*fx_inter1_x + 0.25*fx_inter1_y + 0.25*fx_inter2_x + 0.25*fx_inter2_y



def deriv_i(x_1, x_2, fx, x0, i):
    fx0 = my_interpolation(x0, x_1, x_2, fx)
    #points = np.array([xx_1.flatten(), xx_2.flatten()]).T
    #values = fx.flatten()
    #fx0 = griddata(points, values, (x0[0], x0[1]), method = 'linear')
    x0i_eps = x0[i] + eps
    x1 = x0
    x1[i] = x0i_eps
    fx0i_eps = my_interpolation(x1, x_1, x_2, fx)
    #fx0i_eps = griddata(points, values, (x1[0], x1[1]), method = 'linear')
    grad_t = (fx0i_eps - fx0) / eps
    return grad_t, fx0

def grad(x_1, x_2, fx, x0):
    d0, fx0 = deriv_i(x_1, x_2, fx, x0, 0)
    d1, fx1 = deriv_i(x_1, x_2, fx, x0, 1)
    return np.array([d0, d1]), fx0



x_list = x0
y_list = my_interpolation(x0, x_1, x_2, fx)

for i in np.arange(Niter):
    dx, fx0 = grad(x_1, x_2, fx, x0)
    x0 = x0 - gamma*dx
    x_list = np.vstack([x_list, x0])
    y_list = np.vstack([y_list, fx0])


print(y_list)



plt_x_1 = np.linspace(-1, 1, 1000)
plt_x_2 = np.linspace(-1, 1, 1000)


plt_xx_1, plt_xx_2 = np.meshgrid(plt_x_1, plt_x_2)

plt_fx = plt_xx_1**2+plt_xx_2**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plt_xx_1, plt_xx_2, plt_fx, cmap='viridis', alpha=0.6)

# Add labels and a color bar
ax.set_title("3D Surface Plot of f(x) = x₁² + x₂² with Optimization Path")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("f(x)")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot the optimization path as a line
ax.plot(x_list[:, 0], x_list[:, 1], y_list.flatten(), color='red', label='Optimization Path')
ax.legend()

# Show the plot
plt.show()


