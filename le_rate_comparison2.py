import numpy as np
import plotly.graph_objects as go

x0 = 0.45
delta = 2
N0points = 10000
x = np.linspace(x0-delta, x0+delta, N0points)

fx = np.sin(x)


def grad(x, fx, x0):
    x1 = x0+1e-16
    y0 = np.interp(x0,x,fx)
    y1 = np.interp(x1,x,fx)
    return (y1-y0)/(x1-x0)

#gfx0 = grad(x,fx, 0.45)


Nlines = 10
Niter = 100
x_list = np.empty((Nlines, Niter))
iter_list = np.ones(Nlines)*Niter
gamma_list = np.linspace(0.1, 0.001, Nlines)
xmin = 0

for (i, gamma) in enumerate(gamma_list):
    x1 = x0
    for j in np.arange(1,Niter):
        x1 = x1 - gamma*grad(x, fx, x1)
        x_list[i,j] = x1




y_list = np.interp(x_list, x, fx)



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
