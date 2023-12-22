import numpy as np
import matplotlib.pyplot as plt
import json

def rk4(y, dx, n, F):
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    k1 = dx * F(y)
    k2 = dx * F(y + 0.5*k1)
    k3 = dx * F(y + 0.5*k2)
    k4 = dx * F(y + k3)
    y += (k1 + 2*k2 + 2*k3 + k4)/6
    return y

def SIR_model(y):
    S, I, R = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

beta, gamma = 0.1, 0.05
y = np.array([0.99, 0.01, 0.0])
dt = 0.1
nsteps = 1000
time = np.zeros(nsteps)
SIR = np.zeros((3,nsteps))
SIR[:,0] = y

for j in range(nsteps-1):
    y = rk4(y, dt, 3, SIR_model)
    SIR[:,j+1] = y
    time[j+1] = (j+1)*dt

plt.plot(time, SIR[0,:], label='S')
plt.plot(time, SIR[1,:], label='I')
plt.plot(time, SIR[2,:], label='R')
plt.legend()
plt.show()
