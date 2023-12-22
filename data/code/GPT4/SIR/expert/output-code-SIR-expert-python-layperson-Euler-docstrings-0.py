import numpy as np
import matplotlib.pyplot as plt

def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dydt = [-beta*S*I, beta*S*I - gamma*I, gamma*I]
    return dydt

def euler_method(y0, t, dt, model, *args):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(0, len(t)-1):
        y[i+1] = y[i] + np.array(model(t[i], y[i], *args)) * dt
    return y

beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0
y0 = [S0, I0, R0]
t = np.linspace(0, 100, 1000)
dt = t[1] - t[0]
y = euler_method(y0, t, dt, SIR_model, beta, gamma)

plt.figure()
plt.plot(t, y[:,0], label='S(t)')
plt.plot(t, y[:,1], label='I(t)')
plt.plot(t, y[:,2], label='R(t)')
plt.legend()
plt.show()
