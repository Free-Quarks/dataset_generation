import numpy as np
import matplotlib.pyplot as plt
import json

def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3 ) )
            )( dt * f( t + dt/2, y + dy2/2 ) )
            )( dt * f( t + dt/2, y + dy1/2 ) )
            )( dt * f( t       , y         ) )

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

beta, gamma = 0.2, 1./10 
S0, I0, R0 = 1-1e-6, 1e-6, 0
t = np.linspace(0, 160, 160) 
y0 = [S0, I0, R0] 

sir = RK4(sir_model)
ys = np.array([sir(t[i], ys[i-1], t[i]-t[i-1], beta, gamma) for i in range(1, len(t))])
ys = np.insert(ys, 0, y0, axis=0)

plt.figure()
plt.plot(t, ys[:,0], label='S(t)')
plt.plot(t, ys[:,1], label='I(t)')
plt.plot(t, ys[:,2], label='R(t)')
plt.legend()
plt.show()
