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
gamma = 0.1
x_list = np.array(x0)
for i in np.arange(1000):
    x0 = x0 - gamma*grad(x, fx, x0)
    x_list = np.append(x_list, x0)

print(x0)
print(x_list)

y_list = np.interp(x_list, x, fx)

#print(gfx0)
#print(3*0.45**2)

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
