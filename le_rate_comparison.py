import numpy as np
import plotly.graph_objects as go

x = np.linspace(-1, 1, 10000)

fx = x**2


def grad(x, fx, x0):
    x1 = x0+1e-16
    y0 = np.interp(x0,x,fx)
    y1 = np.interp(x1,x,fx)
    return (y1-y0)/(x1-x0)

gfx0 = grad(x,fx, 0.45)



x0 = 0.45
Nlines = 10
Npoints = 1000
x_list = np.empty((Nlines, Npoints))
iter_list = np.ones(Nlines)*Npoints
gamma_list = np.linspace(0.1, 0.001, Nlines)
xmin = 0

for (i, gamma) in enumerate(gamma_list):
    x1 = x0
    for j in np.arange(1,Npoints):
        x1 = x1 - gamma*grad(x, fx, x1)
        x_list[i,j] = x1
        if abs(x1-xmin) < 1e-3 :
            iter_list[i] = j
            break




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