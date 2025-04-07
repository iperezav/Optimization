import numpy as np
import plotly.graph_objects as go

'''
Parameters
'''

# Initial point of the gradient descent
x0 = float(0.31)
# Epsilon value representing the perturbation in the calculation of the derivative
eps = float(1e-15)
# Size of neighborhood around x0
delta = float(2)
# Number of points in the neighborhood around x0
N0points = 10000
# Number of iterations of gradient descent
Niter = 100
# Learning rate of gradient descent
gamma = float(0.01)

# Neighborhood around the initial point x0 where all the iterations must lie
x = np.linspace(x0-delta, x0+delta, N0points)
# Function to minimize
fx = np.exp(-x)

'''
Functions
'''

# derivative or gradient function
def grad(x, fx, x0):
    x1 = x0+eps
    y0 = np.interp(x0,x,fx)
    y1 = np.interp(x1,x,fx)
    return (y1-y0)/eps

#
#gfx0 = grad(x,fx, 0.45)

'''
Main
'''

# Vector containing all the iteration of the gradient descent
x_list = np.array(x0)

# gradient descent procedure
for i in np.arange(Niter):
    x0 = x0 - gamma*grad(x, fx, x0)
    x_list = np.append(x_list, x0)




print(x0)
print(x_list)

# Vector of the image points of points in the domain produced by gradient descent
y_list = np.interp(x_list, x, fx)

#print(gfx0)
#print(3*0.45**2)



'''
Figures
'''

print(x0)
fig = go.Figure()
fig.add_trace(go.Scatter(x = x, y = fx, name = 'Function'))
fig.add_trace(go.Scatter(x = x_list, y = y_list, name = 'Gradient Descent Iterations'))
fig.update_layout(title = 'Minimization',
                  xaxis_title = '$x$',
                  yaxis_title = '$f(x)$',
                  legend = dict(
                      orientation = 'h',
                      yanchor = 'bottom',
                      y = 1.02,
                      xanchor = 'left',
                      x = 0.0
                  )
                  )

fig.show()
#print(fx)
