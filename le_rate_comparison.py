import numpy as np
import plotly.graph_objects as go

'''
Parameters
'''

# Initial point of the gradient descent
x0 = float(0.3)
# Epsilon value representing the perturbation in the calculation of the derivative
eps = float(1e-15)
# Size of neighborhood around x0
delta = float(2)
# Number of points in the neighborhood around x0
N0points = 10000
# Number of iterations of gradient descent
Niter = 100
# Neighborhood around the initial point x0 where all the iterations must lie
x = np.linspace(x0-delta, x0+delta, N0points)
# Function to minimize
fx = x**2
# Number of learning rates
Nlines = 10
# List of learning rates
gamma_list = np.linspace(0.1, 0.001, Nlines)

# Minimum value previously known
xmin = 0




'''
Functions
'''

# derivative or gradient function
def grad(x, fx, x0):
    x1 = x0+eps
    y0 = np.interp(x0,x,fx)
    y1 = np.interp(x1,x,fx)
    return (y1-y0)/eps

#gfx0 = grad(x,fx, 0.45)

'''
Main
'''


x_list = np.empty((Nlines, Niter))
x_list[:,0] = x0
iter_list = np.ones(Nlines)*Niter

for (i, gamma) in enumerate(gamma_list):
    x1 = x0
    for j in np.arange(1,Niter):
        x1 = x1 - gamma*grad(x, fx, x1)
        x_list[i,j] = x1
        if abs(x1-xmin) < 1e-3 :
            iter_list[i] = j
            break




y_list = np.interp(x_list, x, fx)

'''
Figures
'''

# Plot
fig = go.Figure()
#fig.add_trace(go.Scatter(x = x, y = fx, name = 'Function'))

for (i, gamma) in enumerate(gamma_list):
    fig.add_trace(go.Scatter(y=y_list[i,:], name= f'$\gamma = {gamma: .4f}$ '))

fig.update_layout(title = 'Minimization Error for Different Learning Rates',
                  xaxis_title = 'Iteration Number',
                  yaxis_title = 'Error',
                  legend = dict(
                      orientation = 'h',
                      yanchor = 'bottom',
                      y = 1.02,
                      xanchor = 'left',
                      x = 0.0
                  )
                  )

fig.show()


#fig2 = go.Figure(data=[go.Bar(x=gamma_list, y=iter_list)])

xlabels = [f'{num: .4f}' for num in np.flip(gamma_list)]
xlabels = [str(a) for a in xlabels]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=xlabels, y=np.flip(iter_list)))
fig2.update_xaxes(type='category')
fig2.update_layout(title = 'Number of Iterations vs Learning Rate',
                  xaxis_title = 'Learning Rate',
                  yaxis_title = 'Number of Iterations'
                  )
fig2.show()